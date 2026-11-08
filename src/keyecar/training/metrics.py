"""Evaluation metrics (pure numpy, no model).

* Command errors: MAE / RMSE of throttle and steering, steering direction
  accuracy, and smoothness, overall and per route condition.
* Steering errors bucketed by how strongly the human steered (center,
  medium, strong, left, right).
* Previous-command sensitivity: how far predictions move when the model's
  motor-state input is set to zero or fed back from its own predictions
  (see "The previous-command problem" in docs/approach.md). These compare
  predictions with predictions, never with the human label, because there is
  no right answer for a situation that never happened.
* Residual closure on DAgger correction targets.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from keyecar.car.control import Condition

METRICS_KIND = "keyecar_policy_robustness_metrics"
METRICS_SCHEMA_VERSION = 1
CENTER_THRESHOLD = 0.15  # |steering| at or below this counts as driving straight
STRONG_THRESHOLD = 0.5
SATURATION_THRESHOLD = 0.95


def format_metric(value: float | None) -> str:
    """Three decimals, or "-" when there was nothing to measure."""
    return "-" if value is None or np.isnan(value) else f"{value:.3f}"


def turn_direction(steering: np.ndarray) -> np.ndarray:
    """Classify steering as 0 = left, 1 = straight, 2 = right."""
    return np.where(steering < -CENTER_THRESHOLD, 0, np.where(steering > CENTER_THRESHOLD, 2, 1))


def _mean(values: np.ndarray) -> float | None:
    return float(values.mean()) if values.size else None


def _bucket_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, object]:
    if len(pred) == 0:
        return {
            "num_samples": 0,
            "steering_mae": None,
            "steering_rmse": None,
            "steering_sign_accuracy": None,
            "steering_magnitude_ratio": None,
            "throttle_mae": None,
            "command_mae": None,
        }
    error = pred - target
    nonzero = np.abs(target[:, 1]) > 1e-8
    target_magnitude = float(np.abs(target[:, 1]).mean())
    return {
        "num_samples": len(pred),
        "steering_mae": float(np.abs(error[:, 1]).mean()),
        "steering_rmse": float(np.sqrt(np.square(error[:, 1]).mean())),
        "steering_sign_accuracy": _mean(
            (np.sign(pred[nonzero, 1]) == np.sign(target[nonzero, 1])).astype(np.float64)
        ),
        "steering_magnitude_ratio": (
            None if target_magnitude <= 1e-8 else float(np.abs(pred[:, 1]).mean() / target_magnitude)
        ),
        "throttle_mae": float(np.abs(error[:, 0]).mean()),
        "command_mae": float(np.abs(error).mean()),
    }


def steering_target_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, dict[str, object]]:
    """Supervised metrics, bucketed by how strongly the human steered."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    steering = target[:, 1]
    magnitude = np.abs(steering)
    masks = {
        "overall": np.ones(len(steering), dtype=bool),
        "center": magnitude <= CENTER_THRESHOLD,
        "medium": (magnitude > CENTER_THRESHOLD) & (magnitude <= STRONG_THRESHOLD),
        "strong": magnitude > STRONG_THRESHOLD,
        "negative": steering < -CENTER_THRESHOLD,
        "positive": steering > CENTER_THRESHOLD,
        "strong_negative": steering < -STRONG_THRESHOLD,
        "strong_positive": steering > STRONG_THRESHOLD,
    }
    return {name: _bucket_metrics(pred[mask], target[mask]) for name, mask in masks.items()}


def _drift_growth(drift: np.ndarray, session_ids: Sequence[str]) -> dict[str, object]:
    """Compare the drift in the first and last quarter of each session."""
    by_session: dict[str, list[float]] = defaultdict(list)
    for session_id, value in zip(session_ids, drift, strict=True):
        by_session[session_id].append(float(value))
    per_session = {}
    for session_id in sorted(by_session):
        values = np.asarray(by_session[session_id])
        quarter = max(1, int(np.ceil(len(values) / 4)))
        first, last = float(values[:quarter].mean()), float(values[-quarter:].mean())
        per_session[session_id] = {
            "num_samples": len(values),
            "first_quarter_steering_drift": first,
            "last_quarter_steering_drift": last,
            "drift_growth": last - first,
        }
    first_mean = float(np.mean([s["first_quarter_steering_drift"] for s in per_session.values()]))
    last_mean = float(np.mean([s["last_quarter_steering_drift"] for s in per_session.values()]))
    return {
        "first_quarter_steering_drift": first_mean,
        "last_quarter_steering_drift": last_mean,
        "steering_drift_growth": last_mean - first_mean,
        "per_session": per_session,
    }


def motor_input_sensitivity_metrics(
    recorded: np.ndarray, counterfactual: np.ndarray, *, session_ids: Sequence[str]
) -> dict[str, object]:
    """How far predictions move when the motor-state input is replaced (no targets involved)."""
    drift = np.abs(counterfactual - recorded)
    recorded_magnitude = float(np.abs(recorded[:, 1]).mean())
    turning = np.abs(recorded[:, 1]) > CENTER_THRESHOLD
    sign_flip = np.sign(counterfactual[turning, 1]) != np.sign(recorded[turning, 1])
    collapse = np.abs(counterfactual[turning, 1]) <= CENTER_THRESHOLD
    return {
        "reference_mode": "recorded_prediction",
        "ground_truth_accuracy_valid": False,
        "num_samples": len(recorded),
        "mean_abs_throttle_drift": float(drift[:, 0].mean()),
        "mean_abs_steering_drift": float(drift[:, 1].mean()),
        "steering_magnitude_retention": (
            None
            if recorded_magnitude <= 1e-8
            else float(np.abs(counterfactual[:, 1]).mean() / recorded_magnitude)
        ),
        "steering_sign_flip_rate": _mean(sign_flip.astype(np.float64)),
        "steering_collapse_rate": _mean(collapse.astype(np.float64)),
        "steering_saturation_rate": float((np.abs(counterfactual[:, 1]) >= SATURATION_THRESHOLD).mean()),
        **_drift_growth(drift[:, 1], session_ids),
    }


METRIC_NAMES = (
    "num_samples",
    "throttle_mae",
    "steering_mae",
    "throttle_rmse",
    "steering_rmse",
    "turn_accuracy",
    "smoothness_throttle",
    "smoothness_steering",
    "command_mae",
)


def command_error_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Errors of (throttle, steering) predictions; ``smoothness`` is the mean frame-to-frame change."""
    if len(pred) == 0:
        return {name: 0 if name == "num_samples" else float("nan") for name in METRIC_NAMES}
    error = pred - target
    rmse = np.sqrt(np.mean(error * error, axis=0))
    smoothness = np.mean(np.abs(np.diff(pred, axis=0)), axis=0) if len(pred) > 1 else np.zeros(2)
    return {
        "num_samples": len(pred),
        "throttle_mae": float(np.abs(error[:, 0]).mean()),
        "steering_mae": float(np.abs(error[:, 1]).mean()),
        "throttle_rmse": float(rmse[0]),
        "steering_rmse": float(rmse[1]),
        "turn_accuracy": float((turn_direction(pred[:, 1]) == turn_direction(target[:, 1])).mean()),
        "smoothness_throttle": float(smoothness[0]),
        "smoothness_steering": float(smoothness[1]),
        "command_mae": float(np.abs(error).mean()),
    }


def bucketed_command_metrics(pred: np.ndarray, target: np.ndarray, conditions: np.ndarray) -> dict[str, dict]:
    """Command errors per route condition, so turns are not averaged away by straight driving."""
    buckets = {
        "overall": np.ones(len(pred), dtype=bool),
        "forward": conditions == Condition.FORWARD,
        "turn": conditions != Condition.FORWARD,
        "left_turn": conditions == Condition.LEFT,
        "right_turn": conditions == Condition.RIGHT,
    }
    return {name: command_error_metrics(pred[mask], target[mask]) for name, mask in buckets.items()}


def masked_command_metrics(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Training metrics where each output only counts where it is supervised (``mask`` is (N, 2) bool)."""
    error = np.abs(pred - target)
    steering = mask[:, 1]
    return {
        "throttle_mae": float(error[mask[:, 0], 0].mean()),
        "steering_mae": float(error[steering, 1].mean()),
        "turn_accuracy": float(
            (turn_direction(pred[steering, 1]) == turn_direction(target[steering, 1])).mean()
        ),
    }


def build_robustness_metrics_payload(
    candidate_id: str, evaluation: dict[str, object], *, residual_closure: dict[str, object] | None = None
) -> dict[str, object]:
    predictions = evaluation["predictions"]
    recorded = predictions["recorded"]
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "kind": METRICS_KIND,
        "candidate_id": candidate_id,
        "supervised_mode": "recorded",
        "ground_truth_motor_state": "recorded_previous_executed_command",
        "supervised_metrics": {
            "bucket_definition": {
                "center": f"abs(target_steering) <= {CENTER_THRESHOLD}",
                "medium": f"{CENTER_THRESHOLD} < abs(target_steering) <= {STRONG_THRESHOLD}",
                "strong": f"abs(target_steering) > {STRONG_THRESHOLD}",
            },
            "buckets": steering_target_metrics(recorded, evaluation["targets"]),
        },
        "sensitivity": {
            mode: motor_input_sensitivity_metrics(
                recorded, predictions[mode], session_ids=evaluation["session_ids"]
            )
            for mode in ("zero", "recursive")
        },
        "residual_closure": residual_closure,
    }


def intervention_residual_closure_metrics(
    baseline: np.ndarray, target: np.ndarray, prediction: np.ndarray, *, regions: Sequence[str], split: str
) -> dict[str, object]:
    """How much of the gap between the recorded model steering and the correction target is closed.

    1 means the prediction hits the target, 0 means it is no better than what
    the model did when the human took over.
    """
    baseline_error = np.abs(baseline - target)
    usable = np.isfinite(baseline) & np.isfinite(target) & np.isfinite(prediction) & (baseline_error > 1e-8)
    regions = np.asarray(regions)

    def summarize(mask: np.ndarray) -> dict[str, object]:
        mask = mask & usable
        if not mask.any():
            return {
                "num_samples": 0,
                "baseline_target_mae": None,
                "prediction_target_mae": None,
                "mean_residual_closure": None,
                "median_residual_closure": None,
                "mean_clipped_residual_closure": None,
            }
        prediction_error = np.abs(prediction[mask] - target[mask])
        closure = 1.0 - prediction_error / baseline_error[mask]
        return {
            "num_samples": int(mask.sum()),
            "baseline_target_mae": float(baseline_error[mask].mean()),
            "prediction_target_mae": float(prediction_error.mean()),
            "mean_residual_closure": float(closure.mean()),
            "median_residual_closure": float(np.median(closure)),
            "mean_clipped_residual_closure": float(np.clip(closure, 0.0, 1.0).mean()),
        }

    return {
        "split": split,
        "selection_eligible": split == "val",
        "provenance_label": "held_out_supporting_evidence" if split == "val" else "training_diagnostic",
        "num_entries": len(baseline),
        "num_usable_entries": int(usable.sum()),
        "overall": summarize(np.ones(len(baseline), dtype=bool)),
        "exact_human": summarize(regions == "human"),
        "pseudo_window": summarize(regions != "human"),
    }
