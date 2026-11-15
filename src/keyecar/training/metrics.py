"""Evaluation metrics (pure numpy, no model).

* Command errors: MAE / RMSE of throttle and steering, steering direction
  accuracy, and smoothness, overall and per route condition.
* Steering errors bucketed by how strongly the human steered (center,
  medium, strong, left, right).
* Trivial baselines: copying the previous command and always predicting zero.
  At 20 Hz the stick barely moves between frames, so copying scores very well
  on the plain errors above. Every model score is read against it.
* Change slices: errors on the few frames where the human command moves away
  from the previous one. Only there does the image have to supply the answer.
* Turn-in lag: per steering step of the human, how much later the model
  reaches half of the step.
* Whole turns: per turn of the human, the model's summed steering as a share
  of the human's. A model that turns a little later, or harder for a shorter
  time, still scores 1.
* Accuracy by motor mode: the three scores above with the model's motor-state
  input set to the recorded command, to zero, or to its own previous output.
  The image and the human label are the same in all three, so the label stays
  valid. It is still open loop: the images come from the human's drive.
* Previous-command sensitivity: how far predictions move between those modes,
  prediction against prediction.
* Residual closure on DAgger correction targets.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from keyecar.car.control import Condition

METRICS_KIND = "keyecar_policy_robustness_metrics"
METRICS_SCHEMA_VERSION = 3
CENTER_THRESHOLD = 0.15  # |steering| at or below this counts as driving straight
STRONG_THRESHOLD = 0.5
SATURATION_THRESHOLD = 0.95
HELD_THRESHOLD = 0.05  # the human command moved at most this far from the previous one
CHANGE_THRESHOLD = CENTER_THRESHOLD  # ... or further than this: a steering decision


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


# ---------------------------------------------------------------------------
# Baselines and change slices
# ---------------------------------------------------------------------------


def skill(model_error: float | None, baseline_error: float | None) -> float | None:
    """``1 - model / baseline``: 1 is perfect, 0 equals the baseline, negative is worse."""
    if model_error is None or baseline_error is None or np.isnan(model_error):
        return None
    if not baseline_error > 1e-12:
        return None
    return float(1.0 - model_error / baseline_error)


def baseline_metrics(previous: np.ndarray, target: np.ndarray) -> dict[str, dict[str, object]]:
    """Scores of two predictors that never look at the image."""
    previous = np.asarray(previous, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    zero = np.zeros_like(target)
    return {
        name: {
            "overall": command_error_metrics(prediction, target),
            "buckets": steering_target_metrics(prediction, target),
        }
        for name, prediction in (("copy_previous", previous), ("always_zero", zero))
    }


def change_slice_metrics(
    pred: np.ndarray, target: np.ndarray, previous: np.ndarray
) -> dict[str, dict[str, object]]:
    """Steering errors split by how far the human command moved away from the previous one.

    ``median_step_ratio`` on the changed frames is the share of the human's
    step that the model follows: 0 keeps the previous command, 1 matches the
    human, negative moves the wrong way.
    """
    pred, target, previous = (np.asarray(a, dtype=np.float64) for a in (pred, target, previous))
    step = target[:, 1] - previous[:, 1]
    moved = np.abs(step)  # also the error of copying the previous command
    error = np.abs(pred[:, 1] - target[:, 1])

    def summarize(mask: np.ndarray) -> dict[str, object]:
        if not mask.any():
            return {
                "num_samples": 0,
                "share": None if len(mask) == 0 else 0.0,
                "steering_mae": None,
                "copy_steering_mae": None,
                "steering_skill_vs_copy": None,
            }
        model_mae, copy_mae = float(error[mask].mean()), float(moved[mask].mean())
        return {
            "num_samples": int(mask.sum()),
            "share": float(mask.mean()),
            "steering_mae": model_mae,
            "copy_steering_mae": copy_mae,
            "steering_skill_vs_copy": skill(model_mae, copy_mae),
        }

    changed = moved > CHANGE_THRESHOLD
    slices = {"held": summarize(moved <= HELD_THRESHOLD), "changed": summarize(changed)}
    ratio = (pred[changed, 1] - previous[changed, 1]) / step[changed]
    slices["changed"]["median_step_ratio"] = float(np.median(ratio)) if ratio.size else None
    slices["changed"]["wrong_way_rate"] = _mean((ratio < 0.0).astype(np.float64))
    return slices


def sequence_breaks(
    session_ids: Sequence[str],
    frame_index: Sequence[int],
    timestamps_ns: Sequence[int] | None = None,
    *,
    max_gap_ns: int | None = None,
) -> np.ndarray:
    """True where a sample does not directly follow the one before it (always true for the first).

    A sample follows if it is the next frame of the same session and, when
    timestamps are given, came at most ``max_gap_ns`` later.
    """
    sessions = np.asarray(session_ids)
    frames = np.asarray(frame_index, dtype=np.int64)
    breaks = np.ones(len(frames), dtype=bool)
    if len(frames) > 1:
        breaks[1:] = (sessions[1:] != sessions[:-1]) | (np.diff(frames) != 1)
        if timestamps_ns is not None:
            gap = np.diff(np.asarray(timestamps_ns, dtype=np.int64))
            breaks[1:] |= gap <= 0
            if max_gap_ns is not None:
                breaks[1:] |= gap > max_gap_ns
    return breaks


# ---------------------------------------------------------------------------
# Turn-in lag
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnInConfig:
    """What counts as a steering step of the human, and how long the model has to follow it."""

    stable_s: float = 0.2  # the command was steady for this long ...
    stable_range: float = 0.1  # ... within this range ...
    onset_delta: float = 0.05  # ... before it moved by more than this in one frame
    settle_start_s: float = 0.3  # the new level is the median over this window after the onset
    settle_end_s: float = 0.6
    min_step: float = 0.4  # smaller steps are not events
    timeout_s: float = 1.0  # a model that needs longer has missed the step


@dataclass(frozen=True)
class TurnInEvent:
    onset: int  # first sample of the step
    stop: int  # one past the last sample within the timeout
    start_level: float
    end_level: float


def turn_in_events(
    steering: np.ndarray,
    timestamps_ns: Sequence[int],
    breaks: np.ndarray,
    config: TurnInConfig | None = None,
) -> list[TurnInEvent]:
    """Find the human's steering steps. Uses the human signal only, never a prediction."""
    config = config or TurnInConfig()
    steering = np.asarray(steering, dtype=np.float64)
    times = np.asarray(timestamps_ns, dtype=np.int64)
    stable_ns, timeout_ns = round(config.stable_s * 1e9), round(config.timeout_s * 1e9)
    settle_start_ns, settle_end_ns = round(config.settle_start_s * 1e9), round(config.settle_end_s * 1e9)

    starts = np.flatnonzero(breaks)
    events: list[TurnInEvent] = []
    for lo, hi in zip(starts, [*starts[1:], len(steering)], strict=True):
        t, s = times[lo:hi], steering[lo:hi]
        i = 1
        while i < len(s):
            steady_from = int(np.searchsorted(t, t[i - 1] - stable_ns, side="left"))
            is_onset = (
                abs(s[i] - s[i - 1]) > config.onset_delta
                and t[i - 1] - t[0] >= stable_ns  # the steady stretch lies inside this run of frames
                and np.ptp(s[steady_from:i]) <= config.stable_range
                and t[-1] >= t[i] + timeout_ns  # ... and so does the whole timeout
            )
            if is_onset:
                settled = s[(t >= t[i] + settle_start_ns) & (t <= t[i] + settle_end_ns)]
                if settled.size and abs(np.median(settled) - s[i - 1]) >= config.min_step:
                    stop = int(np.searchsorted(t, t[i] + timeout_ns, side="right"))
                    events.append(
                        TurnInEvent(
                            onset=int(lo + i),
                            stop=int(lo + stop),
                            start_level=float(s[i - 1]),
                            end_level=float(np.median(settled)),
                        )
                    )
                    i = int(np.searchsorted(t, t[i] + settle_end_ns, side="right"))
                    continue
            i += 1
    return events


def _bootstrap_median_ci(values: np.ndarray, *, resamples: int = 1000, seed: int = 0) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(values, size=(resamples, len(values)), replace=True), axis=1)
    return [float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))]


def turn_in_lag_metrics(
    pred_steering: np.ndarray,
    target_steering: np.ndarray,
    timestamps_ns: Sequence[int],
    events: Sequence[TurnInEvent],
) -> dict[str, object]:
    """How much later than the human the model covers half of each steering step.

    An event is ``missed`` if the model does not get there within the timeout,
    and ``already_past`` if it was beyond the half-way level before the human
    moved (then there is no lag to measure). ``median_peak_ratio`` compares how
    far model and human get within the timeout, as a share of the step.
    """
    pred = np.asarray(pred_steering, dtype=np.float64)
    target = np.asarray(target_steering, dtype=np.float64)
    times = np.asarray(timestamps_ns, dtype=np.int64)
    lags_ms: list[float] = []
    peak_ratios: list[float] = []
    missed = already_past = usable = 0
    for event in events:
        half = 0.5 * abs(event.end_level - event.start_level)
        direction = np.sign(event.end_level - event.start_level)
        window = slice(event.onset, event.stop)
        human = (target[window] - event.start_level) * direction
        model = (pred[window] - event.start_level) * direction
        human_reached = np.flatnonzero(human >= half)
        if human_reached.size == 0:
            continue
        usable += 1
        peak_ratios.append(float(model.max() / human.max()))
        if (pred[event.onset - 1] - event.start_level) * direction >= half:
            already_past += 1
            continue
        model_reached = np.flatnonzero(model >= half)
        if model_reached.size == 0:
            missed += 1
            continue
        lag_ns = times[event.onset + model_reached[0]] - times[event.onset + human_reached[0]]
        lags_ms.append(float(lag_ns) / 1e6)

    lags = np.asarray(lags_ms)
    return {
        "num_events": usable,
        "num_lag_measured": len(lags_ms),
        "missed_share": missed / usable if usable else None,
        "already_past_share": already_past / usable if usable else None,
        "median_lag_ms": float(np.median(lags)) if lags.size else None,
        "p90_lag_ms": float(np.percentile(lags, 90)) if lags.size else None,
        "median_lag_ci95_ms": _bootstrap_median_ci(lags),
        "median_peak_ratio": float(np.median(peak_ratios)) if peak_ratios else None,
    }


# ---------------------------------------------------------------------------
# Whole turns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnConfig:
    """What counts as one turn of the human, and as following it."""

    threshold: float = CENTER_THRESHOLD  # a turn lasts while the steering stays beyond this on one side
    min_peak: float = 0.4  # smaller turns are not counted
    min_total: float = 0.2  # ... nor are turns with less summed steering (steering x seconds)
    pad_s: float = 0.5  # straight driving before and after the turn that is summed as well
    followed_low: float = 0.8  # the model follows a turn if its summed steering is
    followed_high: float = 1.25  # between these shares of the human's


@dataclass(frozen=True)
class Turn:
    start: int  # first sample of the turn
    stop: int  # one past the last
    window_start: int  # the same, with the straight driving around the turn
    window_stop: int
    direction: float  # sign of the steering


def _summed_steering(steering: np.ndarray, seconds: np.ndarray, turn: Turn) -> float:
    """Steering in the direction of the turn, integrated over its window (steering x seconds)."""
    window = slice(turn.window_start, turn.window_stop)
    values, times = steering[window], seconds[window]
    return float(turn.direction * 0.5 * ((values[1:] + values[:-1]) * np.diff(times)).sum())


def turn_segments(
    steering: np.ndarray,
    timestamps_ns: Sequence[int],
    breaks: np.ndarray,
    config: TurnConfig | None = None,
) -> list[Turn]:
    """Find the human's turns. Uses the human signal only, never a prediction.

    A turn is a stretch where the steering stays beyond the straight band on
    one side. It must begin and end inside a run of frames, so that it is
    whole. Its window adds the straight driving just before and after, where
    a model that steers earlier or later than the human still counts.
    """
    config = config or TurnConfig()
    steering = np.asarray(steering, dtype=np.float64)
    seconds = np.asarray(timestamps_ns, dtype=np.int64) / 1e9
    side = np.sign(steering) * (np.abs(steering) > config.threshold)  # -1, 0 (straight), or 1

    starts = np.flatnonzero(breaks)
    turns: list[Turn] = []
    for lo, hi in zip(starts, [*starts[1:], len(steering)], strict=True):
        start = lo
        while start < hi:
            stop = start + 1
            while stop < hi and side[stop] == side[start]:
                stop += 1
            whole = start > lo and stop < hi
            if side[start] != 0 and whole and np.abs(steering[start:stop]).max() >= config.min_peak:
                first, last = start, stop
                while (
                    first > lo
                    and side[first - 1] == 0
                    and seconds[start] - seconds[first - 1] <= config.pad_s
                ):
                    first -= 1
                while last < hi and side[last] == 0 and seconds[last] - seconds[stop - 1] <= config.pad_s:
                    last += 1
                turn = Turn(int(start), int(stop), int(first), int(last), float(side[start]))
                if _summed_steering(steering, seconds, turn) >= config.min_total:
                    turns.append(turn)
            start = stop
    return turns


def turn_following_metrics(
    pred_steering: np.ndarray,
    target_steering: np.ndarray,
    timestamps_ns: Sequence[int],
    turns: Sequence[Turn],
    *,
    conditions: np.ndarray | None = None,
    config: TurnConfig | None = None,
) -> dict[str, object]:
    """Per turn of the human, the model's summed steering as a share of the human's.

    Summed steering is roughly the change of heading, so the ratio does not
    depend on when exactly the model steers within the turn's window: 1 is as
    much as the human, below 1 is understeer, above 1 is oversteer. A copy of
    the previous command scores 1, so this is read with the model's own output
    fed back. With ``conditions`` the turns are also split by the route
    command that held for most of the turn.
    """
    config = config or TurnConfig()
    pred = np.asarray(pred_steering, dtype=np.float64)
    target = np.asarray(target_steering, dtype=np.float64)
    seconds = np.asarray(timestamps_ns, dtype=np.int64) / 1e9
    ratios = np.asarray(
        [_summed_steering(pred, seconds, turn) / _summed_steering(target, seconds, turn) for turn in turns]
    )

    def summarize(mask: np.ndarray) -> dict[str, object]:
        selected = ratios[mask]
        if selected.size == 0:
            return {
                "num_turns": 0,
                "median_steering_ratio": None,
                "p10_steering_ratio": None,
                "followed_share": None,
                "understeered_share": None,
                "oversteered_share": None,
            }
        return {
            "num_turns": int(selected.size),
            "median_steering_ratio": float(np.median(selected)),
            "p10_steering_ratio": float(np.percentile(selected, 10)),
            "followed_share": float(
                ((selected >= config.followed_low) & (selected <= config.followed_high)).mean()
            ),
            "understeered_share": float((selected < config.followed_low).mean()),
            "oversteered_share": float((selected > config.followed_high).mean()),
        }

    metrics = summarize(np.ones(len(ratios), dtype=bool))
    if conditions is not None:
        conditions = np.asarray(conditions, dtype=np.int64)
        main_condition = np.asarray(
            [np.bincount(conditions[turn.start : turn.stop]).argmax() for turn in turns], dtype=np.int64
        )
        metrics["by_condition"] = {
            name: summarize(main_condition == condition)
            for name, condition in (
                ("forward", Condition.FORWARD),
                ("left_turn", Condition.LEFT),
                ("right_turn", Condition.RIGHT),
            )
        }
    return metrics


# ---------------------------------------------------------------------------
# Accuracy with the motor-state input replaced
# ---------------------------------------------------------------------------


def motor_mode_accuracy_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    previous: np.ndarray,
    *,
    session_ids: Sequence[str],
    timestamps_ns: Sequence[int],
    events: Sequence[TurnInEvent],
    turns: Sequence[Turn],
    conditions: np.ndarray | None = None,
    turn_config: TurnConfig | None = None,
) -> dict[str, object]:
    """Errors against the human label for one motor-state mode, overall and per session.

    ``previous`` is always the recorded previous command: it defines the
    slices and the copy baseline, whatever the model was fed.
    """
    pred, target, previous = (np.asarray(a, dtype=np.float64) for a in (pred, target, previous))
    sessions = np.asarray(session_ids)
    overall = command_error_metrics(pred, target)
    copy_mae = float(np.abs(previous[:, 1] - target[:, 1]).mean()) if len(target) else None

    per_session = {}
    for session_id in sorted(set(sessions.tolist())):
        mask = sessions == session_id
        changed = change_slice_metrics(pred[mask], target[mask], previous[mask])["changed"]
        turn_in = turn_in_lag_metrics(
            pred[:, 1], target[:, 1], timestamps_ns, [event for event in events if mask[event.onset]]
        )
        whole_turns = turn_following_metrics(
            pred[:, 1],
            target[:, 1],
            timestamps_ns,
            [turn for turn in turns if mask[turn.start]],
            config=turn_config,
        )
        per_session[session_id] = {
            "num_samples": int(mask.sum()),
            "steering_mae": float(np.abs(pred[mask, 1] - target[mask, 1]).mean()),
            "copy_steering_mae": float(np.abs(previous[mask, 1] - target[mask, 1]).mean()),
            "changed_num_samples": changed["num_samples"],
            "changed_steering_mae": changed["steering_mae"],
            "changed_median_step_ratio": changed["median_step_ratio"],
            "turn_in_num_events": turn_in["num_events"],
            "turn_in_median_lag_ms": turn_in["median_lag_ms"],
            "turn_in_missed_share": turn_in["missed_share"],
            "turn_in_median_peak_ratio": turn_in["median_peak_ratio"],
            "num_turns": whole_turns["num_turns"],
            "turn_median_steering_ratio": whole_turns["median_steering_ratio"],
            "turn_followed_share": whole_turns["followed_share"],
        }
    return {
        "overall": overall,
        "steering_skill_vs_copy": skill(overall["steering_mae"], copy_mae),
        "buckets": steering_target_metrics(pred, target),
        "change_slices": change_slice_metrics(pred, target, previous),
        "turn_in": turn_in_lag_metrics(pred[:, 1], target[:, 1], timestamps_ns, events),
        "turns": turn_following_metrics(
            pred[:, 1], target[:, 1], timestamps_ns, turns, conditions=conditions, config=turn_config
        ),
        "per_session": per_session,
    }


def build_robustness_metrics_payload(
    candidate_id: str,
    evaluation: dict[str, object],
    *,
    residual_closure: dict[str, object] | None = None,
    turn_in_config: TurnInConfig | None = None,
    turn_config: TurnConfig | None = None,
) -> dict[str, object]:
    """Assemble ``robustness_metrics.json`` from the output of ``evaluate_policy_motor_modes``."""
    turn_in_config = turn_in_config or TurnInConfig()
    turn_config = turn_config or TurnConfig()
    predictions = evaluation["predictions"]
    recorded = predictions["recorded"]
    targets, previous = evaluation["targets"], evaluation["previous"]
    events = turn_in_events(targets[:, 1], evaluation["timestamps_ns"], evaluation["breaks"], turn_in_config)
    turns = turn_segments(targets[:, 1], evaluation["timestamps_ns"], evaluation["breaks"], turn_config)
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
        "baselines": baseline_metrics(previous, targets),
        "accuracy_by_motor_mode": {
            "reference": "human_label",
            "interpretation": (
                "Open loop: the images come from the human's drive, so errors cannot compound. "
                "'recursive' feeds the model its own previous output, as on the car."
            ),
            "change_thresholds": {"held": HELD_THRESHOLD, "changed": CHANGE_THRESHOLD},
            "turn_in_config": asdict(turn_in_config),
            "turn_config": asdict(turn_config),
            "modes": {
                mode: motor_mode_accuracy_metrics(
                    predictions[mode],
                    targets,
                    previous,
                    session_ids=evaluation["session_ids"],
                    timestamps_ns=evaluation["timestamps_ns"],
                    events=events,
                    turns=turns,
                    conditions=evaluation.get("conditions"),
                    turn_config=turn_config,
                )
                for mode in ("recorded", "zero", "recursive")
            },
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
    baseline: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    regions: Sequence[str],
    split: str,
    first_human: np.ndarray | None = None,
) -> dict[str, object]:
    """How much of the gap between the recorded model steering and the correction target is closed.

    1 means the prediction hits the target, 0 means it is no better than what
    the model did when the human took over.

    ``first_human`` marks the first human frame of each takeover. It is the
    cleanest sample: the motor-state input is still the model's own command,
    while the target is already the human's.
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
                "share_moved_toward_target": None,
                "share_closed_at_least_half": None,
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
            "share_moved_toward_target": float((closure > 0.0).mean()),
            "share_closed_at_least_half": float((closure >= 0.5).mean()),
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
        "first_human_frame": None if first_human is None else summarize(np.asarray(first_human, dtype=bool)),
    }
