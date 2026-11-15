"""Evaluate a trained checkpoint on the validation sessions.

    python -m keyecar.training.eval runs/rgb_history_dagger_v5/best.pt
    python -m keyecar.training.eval runs/rgb_history_dagger_v5/best.pt --videos runs/eval_videos

Writes, next to the checkpoint (or to ``--out-dir``):

* ``eval.json`` and ``eval_buckets.csv``: command errors overall and per route
  condition;
* ``robustness_metrics.json``: everything that asks whether the model reads
  the image or copies its previous-command input.

Each validation frame is predicted in three motor-state modes: ``recorded``
(the human's previous command, from the recording), ``zero``, and
``recursive`` (the model's own previous prediction, frame by frame through
each session, as on the car). All three are scored against the human label,
next to a baseline that only copies the previous command: overall, on the
frames where the human command changes, as the lag behind the human's
steering steps, and as the summed steering over each of the human's turns.
This is open loop. The images come from the human's drive, so
it can show that a model is bad, not that it drives well.

Optionally it also scores the reviewed correction targets (``--correction-manifest``
and ``--correction-split``; use ``val`` for held-out sessions) and renders one
annotated MP4 per validation session.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from keyecar.car.control import Condition
from keyecar.cli import add_checkpoint, add_keep_frames, add_recordings_root, make_parser

from .checkpoint import load_checkpoint, model_from_checkpoint, variant_from_checkpoint
from .dagger import (
    EXACT_SOURCE,
    PSEUDO_SOURCE,
    CorrectionTarget,
    CorrectionWindowDataset,
    load_accepted_correction_sessions,
)
from .dataset import KeyecarSessionDataset, discover_session_split
from .files import sha256_file, write_json
from .metrics import (
    METRIC_NAMES,
    bucketed_command_metrics,
    build_robustness_metrics_payload,
    command_error_metrics,
    format_metric,
    intervention_residual_closure_metrics,
    sequence_breaks,
)
from .model import RegressionPolicy, batch_inputs, default_device, forward_batch
from .tools import drawing
from .tools.video import encode_mp4, frames_directory, infer_fps_from_monotonic_ns

# ---------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------


def evaluate_policy_motor_modes(
    model: RegressionPolicy,
    dataset: KeyecarSessionDataset,
    *,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, object]:
    """Predict every sample with recorded, zero, and recursive motor state.

    The dataset must be in time order within each session: in recursive mode
    each prediction becomes the next sample's motor state. That state starts
    again from zero wherever frames are missing. Stops are not samples, and
    after a stop the previous command is zero on the car too.
    """
    model.to(device).eval()
    frames = dataset.df
    session_column = frames["session_id"].to_numpy()
    frame_index = frames["frame_index"].to_numpy()
    timestamps_ns = frames["timestamp_monotonic_ns"].to_numpy(dtype=np.int64)
    missing_frames = sequence_breaks(session_column, frame_index)

    predictions: dict[str, list[np.ndarray]] = defaultdict(list)
    targets: list[np.ndarray] = []
    previous: list[np.ndarray] = []
    session_ids: list[str] = []
    recursive_state = torch.zeros(1, 2, device=device)
    sample = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="robustness"):
            image, recorded_state, condition, history = batch_inputs(model, batch, device)
            features = model.encode_image(image)  # shared by all three modes

            predictions["recorded"].append(
                model.head(features, recorded_state, condition, history).cpu().numpy()
            )
            zero_state = torch.zeros_like(recorded_state)
            predictions["zero"].append(model.head(features, zero_state, condition, history).cpu().numpy())
            for i in range(len(batch["session_id"])):
                if missing_frames[sample]:
                    # A new tensor, not zero_(): the stored numpy outputs share memory with the old one.
                    recursive_state = torch.zeros(1, 2, device=device)
                sample += 1
                row = slice(i, i + 1)
                recursive_state = model.head(
                    features[row], recursive_state, condition[row], None if history is None else history[row]
                )
                predictions["recursive"].append(recursive_state.cpu().numpy())

            targets.append(batch["target_command"].numpy())
            previous.append(batch["motor_state"].numpy())
            session_ids.extend(batch["session_id"])
    return {
        "predictions": {
            mode: np.concatenate(values).astype(np.float64) for mode, values in predictions.items()
        },
        "targets": np.concatenate(targets).astype(np.float64),
        "previous": np.concatenate(previous).astype(np.float64),
        "session_ids": session_ids,
        "timestamps_ns": timestamps_ns,
        "conditions": frames["condition"].to_numpy(dtype=np.int64),
        # For the turn-in events and the turns a late frame also ends a run of frames.
        "breaks": sequence_breaks(
            session_column, frame_index, timestamps_ns, max_gap_ns=dataset.history_max_gap_ns
        ),
    }


def first_human_frames(targets: Sequence[CorrectionTarget]) -> list[bool]:
    """Mark the earliest human frame of each takeover."""
    earliest: dict[str, int] = {}
    for index, target in enumerate(targets):
        if target.region != "human":
            continue
        best = earliest.get(target.intervention_key)
        if best is None or target.base_index < targets[best].base_index:
            earliest[target.intervention_key] = index
    marked = set(earliest.values())
    return [index in marked for index in range(len(targets))]


def evaluate_intervention_residual_datasets(
    model: RegressionPolicy,
    datasets: Iterable[CorrectionWindowDataset],
    *,
    split: str,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, object]:
    """Residual closure of the exact-human and pseudo-window correction targets."""
    model.to(device).eval()
    baseline, target, prediction, regions, first_human = [], [], [], [], []
    with torch.no_grad():
        for dataset in datasets:
            for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
                prediction.extend(forward_batch(model, batch, device)[:, 1].cpu().tolist())
                target.extend(batch["target_command"][:, 1].tolist())
            baseline.extend(t.baseline_steering for t in dataset.targets)
            regions.extend(t.region for t in dataset.targets)
            first_human.extend(first_human_frames(dataset.targets))
    return intervention_residual_closure_metrics(
        np.asarray(baseline),
        np.asarray(target),
        np.asarray(prediction),
        regions=regions,
        split=split,
        first_human=np.asarray(first_human, dtype=bool),
    )


# ---------------------------------------------------------------------------
# Videos
# ---------------------------------------------------------------------------


def render_eval_frame(
    *,
    image_path: Path,
    output_path: Path,
    condition: int,
    target_command: np.ndarray,
    pred_command: np.ndarray,
    frame_index: int,
    image_size: int = 640,
) -> Path:
    """Draw one validation frame with the human target and the prediction below it."""
    canvas, draw = drawing.hud_canvas(Image.open(image_path).convert("RGB"), image_size, image_size, 132)
    font, small = drawing.load_font(15), drawing.load_font(13)
    y = image_size

    (target_t, target_s), (pred_t, pred_s) = map(tuple, (target_command, pred_command))
    err_t, err_s = abs(pred_t - target_t), abs(pred_s - target_s)
    mean_err = (err_t + err_s) / 2.0
    err_color = drawing.GREEN if mean_err < 0.10 else drawing.YELLOW if mean_err < 0.25 else drawing.RED
    draw.text(
        (14, y + 10),
        f"frame={frame_index:06d}  cond={Condition(condition).name}",
        fill=drawing.TEXT,
        font=font,
    )
    draw.text((14, y + 42), f"GT  t={target_t:+.2f}  s={target_s:+.2f}", fill=drawing.BLUE, font=font)
    draw.text((14, y + 72), f"PR  t={pred_t:+.2f}  s={pred_s:+.2f}", fill=drawing.ORANGE, font=font)
    draw.text((14, y + 102), f"err t={err_t:.2f}  s={err_s:.2f}", fill=err_color, font=font)

    for label, row_y, low, values in (
        ("throttle", 42, 0.0, (target_t, pred_t)),
        ("steering", 74, -1.0, (target_s, pred_s)),
    ):
        draw.text((240, y + row_y - 2), label, fill=drawing.TEXT, font=small)
        for x, value, color in zip((320, 460), values, (drawing.BLUE, drawing.ORANGE), strict=True):
            drawing.draw_bar(
                draw, x=x, y=y + row_y, width=130, height=16, value=value, low=low, high=1.0, color=color
            )
    draw.text((320, y + 104), "GT", fill=drawing.BLUE, font=small)
    draw.text((460, y + 104), "PR", fill=drawing.ORANGE, font=small)
    return drawing.save_frame(canvas, output_path)


def write_eval_videos(
    *,
    dataset: KeyecarSessionDataset,
    output_dir: Path,
    predictions: np.ndarray,
    targets: np.ndarray,
    fps: float | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    keep_frames: bool = False,
) -> dict[str, tuple[Path, int]]:
    """Encode one annotated MP4 per session to ``output_dir/<session_id>/eval.mp4``.

    ``predictions[i]`` and ``targets[i]`` belong to row ``i`` of ``dataset.df``.
    """
    df = dataset.df
    results = {}
    for session_id, session_rows in df.groupby("session_id", sort=False):
        rows = session_rows.sort_values("frame_index", kind="stable").index.to_numpy()[::stride][:max_frames]
        session_dir = output_dir / str(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        output_path = session_dir / "eval.mp4"
        # The timestamps of the selected frames already reflect the stride.
        session_fps = fps or infer_fps_from_monotonic_ns(df.loc[rows, "timestamp_monotonic_ns"].to_numpy())
        with frames_directory(session_dir / "frames" if keep_frames else None) as frames_dir:
            for i, row in enumerate(rows):
                render_eval_frame(
                    image_path=dataset.image_paths[row],
                    output_path=frames_dir / f"frame_{i:06d}.jpg",
                    condition=int(df.at[row, "condition"]),
                    target_command=targets[row],
                    pred_command=predictions[row],
                    frame_index=int(df.at[row, "frame_index"]),
                )
            encode_mp4(frames_dir=frames_dir, fps=max(0.1, session_fps), output_path=output_path)
        results[str(session_id)] = (output_path, len(rows))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_bucket_metrics_csv(path: Path, bucket_metrics: dict[str, dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["bucket", *METRIC_NAMES])
        writer.writeheader()
        writer.writerows({"bucket": bucket, **metrics} for bucket, metrics in bucket_metrics.items())


def print_robustness(payload: dict[str, object]) -> None:
    buckets = payload["supervised_metrics"]["buckets"]
    print("\nSteering by target bucket (recorded motor state):")
    for name in ("center", "medium", "strong", "negative", "positive"):
        metrics = buckets[name]
        print(
            f"[target_{name}] samples={metrics['num_samples']} "
            f"steer_mae={format_metric(metrics['steering_mae'])} "
            f"sign_acc={format_metric(metrics['steering_sign_accuracy'])} "
            f"magnitude_ratio={format_metric(metrics['steering_magnitude_ratio'])}"
        )
    print("Motor-state sensitivity (change relative to the recorded-input prediction, not accuracy):")
    for mode in ("zero", "recursive"):
        metrics = payload["sensitivity"][mode]
        print(
            f"[{mode}] steer_drift={format_metric(metrics['mean_abs_steering_drift'])} "
            f"throttle_drift={format_metric(metrics['mean_abs_throttle_drift'])} "
            f"retention={format_metric(metrics['steering_magnitude_retention'])} "
            f"sign_flip={format_metric(metrics['steering_sign_flip_rate'])} "
            f"collapse={format_metric(metrics['steering_collapse_rate'])} "
            f"drift_growth={format_metric(metrics['steering_drift_growth'])}"
        )
    copy = payload["baselines"]["copy_previous"]
    print(
        "\nAgainst the human label, by motor-state input (open loop). "
        f"Copying the previous command: steer_mae={format_metric(copy['overall']['steering_mae'])} "
        f"strong={format_metric(copy['buckets']['strong']['steering_mae'])}"
    )
    for mode, metrics in payload["accuracy_by_motor_mode"]["modes"].items():
        changed, turn_in, turns = metrics["change_slices"]["changed"], metrics["turn_in"], metrics["turns"]
        print(
            f"[{mode}] steer_mae={format_metric(metrics['overall']['steering_mae'])} "
            f"skill_vs_copy={format_metric(metrics['steering_skill_vs_copy'])} | "
            f"changed frames n={changed['num_samples']} "
            f"steer_mae={format_metric(changed['steering_mae'])} "
            f"(copy {format_metric(changed['copy_steering_mae'])}) "
            f"step_ratio={format_metric(changed['median_step_ratio'])} | "
            f"turn-in events={turn_in['num_events']} "
            f"lag_ms={format_metric(turn_in['median_lag_ms'])} "
            f"missed={format_metric(turn_in['missed_share'])} "
            f"peak_ratio={format_metric(turn_in['median_peak_ratio'])} | "
            f"whole turns n={turns['num_turns']} "
            f"steering_ratio={format_metric(turns['median_steering_ratio'])} "
            f"followed={format_metric(turns['followed_share'])} "
            f"understeered={format_metric(turns['understeered_share'])}"
        )
    residual = payload["residual_closure"]
    if residual is not None:
        first = residual["first_human_frame"]
        print(
            f"[residual_closure:{residual['split']}] samples={residual['overall']['num_samples']} "
            f"mean={format_metric(residual['overall']['mean_residual_closure'])} | "
            f"first human frame of each takeover n={first['num_samples']} "
            f"clipped_mean={format_metric(first['mean_clipped_residual_closure'])} "
            f"moved_toward={format_metric(first['share_moved_toward_target'])} "
            f"closed_half={format_metric(first['share_closed_at_least_half'])}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser(__doc__)
    add_checkpoint(parser)
    add_recordings_root(parser)
    parser.add_argument("--batch-size", type=int, default=64, help="samples per forward pass")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="data loader workers (0 avoids shared-memory hangs on macOS)",
    )
    parser.add_argument("--device", default=default_device(allow_mps=False), help="torch device")
    parser.add_argument(
        "--all-frames", action="store_true", help="evaluate every frame, not only training targets"
    )
    parser.add_argument("--no-robustness", action="store_true", help="skip the motor-state checks")
    parser.add_argument(
        "--out-dir", type=Path, help="write the results here (default: next to the checkpoint)"
    )

    corrections = parser.add_argument_group("correction targets (optional, give both)")
    corrections.add_argument("--correction-manifest", type=Path, help="reviewed correction manifest to score")
    corrections.add_argument(
        "--correction-split", choices=["train", "val"], help="split of the correction sessions"
    )

    videos = parser.add_argument_group("videos")
    videos.add_argument("--videos", type=Path, help="write one annotated MP4 per validation session here")
    videos.add_argument("--video-fps", type=float, help="playback rate (default: the recorded rate)")
    videos.add_argument("--video-stride", type=int, default=1, help="render every Nth frame")
    videos.add_argument("--video-max-frames", type=int, help="limit the frames per video")
    add_keep_frames(videos)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.correction_manifest is None) != (args.correction_split is None):
        parser.error("--correction-manifest and --correction-split go together")

    state = load_checkpoint(args.checkpoint)
    variant = variant_from_checkpoint(state)
    metadata = state["metadata"]
    device = torch.device(args.device)
    model = model_from_checkpoint(state).to(device).eval()
    data_options = {
        "image_size": int(metadata["image_size"]),
        "history_length": variant.history_length,
        "history_max_gap_s": float(metadata["temporal_context"]["history_max_gap_s"]),
        "augment": False,
    }

    _, val_sessions = discover_session_split(args.recordings_root)
    print(f"Val sessions: {[p.name for p in val_sessions]}")
    print(f"Variant: {variant.name} (history={variant.history_length})")
    dataset = KeyecarSessionDataset(val_sessions, filter_targets=not args.all_frames, **data_options)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            predictions.append(forward_batch(model, batch, device).cpu().numpy())
            targets.append(batch["target_command"].numpy())
    predictions, targets = np.concatenate(predictions), np.concatenate(targets)
    conditions = dataset.df["condition"].to_numpy()

    command_metrics = command_error_metrics(predictions, targets)
    bucket_metrics = bucketed_command_metrics(predictions, targets, conditions)
    print(
        f"\ncommand: throttle_mae={command_metrics['throttle_mae']:.3f} "
        f"steering_mae={command_metrics['steering_mae']:.3f} turn_acc={command_metrics['turn_accuracy']:.3f}"
    )
    for name in ("forward", "left_turn", "right_turn"):
        metrics = bucket_metrics[name]
        print(
            f"[{name}] samples={metrics['num_samples']} thr_mae={metrics['throttle_mae']:.3f} "
            f"steer_mae={metrics['steering_mae']:.3f} turn_acc={metrics['turn_accuracy']:.3f}"
        )

    out_dir = args.out_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    robustness = None
    if not args.no_robustness:
        residual_closure = None
        if args.correction_manifest is not None:
            sessions, _ = load_accepted_correction_sessions(
                args.correction_manifest, split=args.correction_split
            )
            datasets = [
                CorrectionWindowDataset(sessions, source=source, **data_options)
                for source in (EXACT_SOURCE, PSEUDO_SOURCE)
            ]
            residual_closure = evaluate_intervention_residual_datasets(
                model, datasets, split=args.correction_split, device=device
            )
        evaluation = evaluate_policy_motor_modes(
            model, dataset, device=device, batch_size=args.batch_size, num_workers=args.num_workers
        )
        robustness = build_robustness_metrics_payload(
            args.checkpoint.parent.name, evaluation, residual_closure=residual_closure
        )
        robustness["evaluation_provenance"] = {
            "checkpoint_path": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "recordings_root": str(args.recordings_root),
            "validation_sessions": [str(path) for path in val_sessions],
            "variant": variant.to_dict(),
            "correction_manifest": None
            if args.correction_manifest is None
            else str(args.correction_manifest),
            "correction_split": args.correction_split,
        }
        print_robustness(robustness)
        write_json(out_dir / "robustness_metrics.json", robustness, sort_keys=True)
        print(f"Wrote {out_dir / 'robustness_metrics.json'}")

    summary = {
        "variant": variant.to_dict(),
        "command_metrics": command_metrics,
        "bucket_metrics": bucket_metrics,
        "robustness": robustness,
    }
    write_json(out_dir / "eval.json", summary)
    write_bucket_metrics_csv(out_dir / "eval_buckets.csv", bucket_metrics)
    print(f"Wrote {out_dir / 'eval.json'} and {out_dir / 'eval_buckets.csv'}")

    if args.videos is not None:
        videos = write_eval_videos(
            dataset=dataset,
            output_dir=args.videos,
            predictions=predictions,
            targets=targets,
            fps=args.video_fps,
            stride=args.video_stride,
            max_frames=args.video_max_frames,
            keep_frames=args.keep_frames,
        )
        frames = sum(count for _, count in videos.values())
        print(f"Wrote {len(videos)} eval videos ({frames} frames) to {args.videos}")


if __name__ == "__main__":
    main()
