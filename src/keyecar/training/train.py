"""Train a driving policy on the human demonstrations plus reviewed DAgger corrections.

    python -m keyecar.training.train --variant rgb_history --out-dir runs/rgb_history_dagger_v5
    python -m keyecar.training.train --quick --epochs 1 --out-dir runs/smoke   # smoke test

The defaults reproduce the shipped ``*_dagger_v4`` models: AdamW (lr 1e-4,
weight decay 1e-3, cosine schedule) for 100 epochs; batches of 36 demonstration,
20 exact-human, and 8 pseudo-window samples; Smooth L1 loss with steering
weighted 5x; the motor-state input hidden for 25% of samples; photometric and
mirror augmentation. ``best.pt`` is the epoch with the lowest validation loss,
computed with the recorded motor state (the only setting with a ground truth).

Expected layout (sessions are assigned to splits by hand, never by frame)::

    recordings/train/<session>/   recordings/val/<session>/
    recordings/correction_sessions.json   (reviewed DAgger manifest)
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from keyecar.car.inference import IMAGENET_MEAN, IMAGENET_STD
from keyecar.cli import add_recordings_root, make_parser

from .dagger import (
    EXACT_SOURCE,
    PSEUDO_SOURCE,
    SEED_SOURCE,
    CorrectionWindowDataset,
    DaggerBatchSampler,
    DaggerMixedDataset,
    correction_exclusion_counts,
    load_accepted_correction_sessions,
)
from .dataset import KeyecarSessionDataset, discover_session_split
from .files import write_json
from .metrics import format_metric, masked_command_metrics, steering_target_metrics
from .model import RegressionPolicy, count_trainable_parameters, default_device, forward_batch
from .variants import DEFAULT_HISTORY_LENGTH, VARIANT_NAMES, resolve_variant


@dataclass
class TrainConfig:
    variant: str
    history_length: int
    recordings_root: Path
    correction_manifest: Path
    out_dir: Path
    epochs: int
    seed_per_batch: int
    exact_per_batch: int
    pseudo_per_batch: int
    steps_per_epoch: int | None
    lr: float
    weight_decay: float
    image_size: int
    motor_state_dropout: float
    history_max_gap_s: float
    throttle_loss_coef: float
    steering_loss_coef: float
    augment: bool
    num_workers: int
    seed: int
    device: str
    quick: bool

    @property
    def batch_size(self) -> int:
        return self.seed_per_batch + self.exact_per_batch + self.pseudo_per_batch

    @property
    def loss_coefficients(self) -> tuple[float, float]:
        return (self.throttle_loss_coef, self.steering_loss_coef)

    def to_json(self) -> dict[str, object]:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser(__doc__)
    model = parser.add_argument_group("model")
    model.add_argument("--variant", choices=VARIANT_NAMES, default="rgb_history", help="policy variant")
    model.add_argument(
        "--history-length", type=int, default=DEFAULT_HISTORY_LENGTH, help="frames seen by rgb_history"
    )

    data = parser.add_argument_group("data")
    add_recordings_root(data)
    data.add_argument(
        "--correction-manifest",
        type=Path,
        default=Path("recordings/correction_sessions.json"),
        help="reviewed DAgger manifest",
    )
    data.add_argument("--image-size", type=int, default=160, help="side of the letterboxed square input")
    data.add_argument(
        "--history-max-gap-s", type=float, default=0.2, help="largest time gap between history frames"
    )
    data.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="augment images")

    batches = parser.add_argument_group("batch mix")
    batches.add_argument("--seed-per-batch", type=int, default=36, help="demonstration samples per batch")
    batches.add_argument("--exact-per-batch", type=int, default=20, help="human takeover samples per batch")
    batches.add_argument(
        "--pseudo-per-batch", type=int, default=8, help="pre/post-takeover samples per batch"
    )
    batches.add_argument("--steps-per-epoch", type=int, help="default: one pass over the demonstrations")

    optimization = parser.add_argument_group("optimization and loss")
    optimization.add_argument("--epochs", type=int, default=100, help="training epochs")
    optimization.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate")
    optimization.add_argument("--weight-decay", type=float, default=1e-3, help="AdamW weight decay")
    optimization.add_argument(
        "--motor-state-dropout",
        type=float,
        default=0.25,
        help="share of samples whose previous-command input is hidden",
    )
    optimization.add_argument("--throttle-loss-coef", type=float, default=1.0, help="throttle loss weight")
    optimization.add_argument("--steering-loss-coef", type=float, default=5.0, help="steering loss weight")

    run = parser.add_argument_group("run")
    run.add_argument("--out-dir", type=Path, default=Path("runs/policy"), help="checkpoints and logs go here")
    run.add_argument("--num-workers", type=int, default=4, help="data loader workers")
    run.add_argument("--seed", type=int, default=42, help="random seed")
    run.add_argument("--device", default=default_device(), help="torch device")
    run.add_argument("--quick", action="store_true", help="200 train / 100 val samples, 5 steps per epoch")
    return parser


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------


def weighted_loss_terms(
    per_output_loss: torch.Tensor, batch: dict, coefficients: tuple[float, float]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the weighted loss sum and the total weight of a batch.

    Each output of each sample is weighted by its supervision mask (pseudo
    targets supervise steering only), its per-sample weight (alpha for pseudo
    targets), and the output coefficient (steering counts 5x by default).
    """
    device = per_output_loss.device
    weights = (
        batch["supervision_mask"].to(device)
        * batch["output_weights"].to(device)
        * per_output_loss.new_tensor(coefficients).reshape(1, 2)
    )
    return (per_output_loss * weights).sum(), weights.sum()


def run_epoch(
    model: RegressionPolicy,
    loader: DataLoader,
    *,
    cfg: TrainConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    """Train for one epoch if an optimizer is given, otherwise evaluate. Returns the metrics."""
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.SmoothL1Loss(reduction="none")
    loss_sum = weight_sum = 0.0
    predictions, targets, masks = [], [], []

    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, leave=False, desc="train" if training else "val"):
            if training and cfg.motor_state_dropout > 0.0:
                # Hide the previous command for some samples, so the model cannot just copy it.
                motor_state = batch["motor_state"]
                keep = (torch.rand(motor_state.size(0), 1, device=device) >= cfg.motor_state_dropout).float()
                batch = {**batch, "motor_state": motor_state.to(device) * keep}

            output = forward_batch(model, batch, device)
            target = batch["target_command"].to(device)
            batch_loss, batch_weight = weighted_loss_terms(
                loss_fn(output, target), batch, cfg.loss_coefficients
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                (batch_loss / batch_weight.clamp_min(1e-8)).backward()
                optimizer.step()

            loss_sum += float(batch_loss.detach())
            weight_sum += float(batch_weight)
            predictions.append(output.detach().cpu().numpy())
            targets.append(target.cpu().numpy())
            masks.append(batch["supervision_mask"].numpy().astype(bool))

    pred, target, mask = np.concatenate(predictions), np.concatenate(targets), np.concatenate(masks)
    return {
        "loss": loss_sum / max(weight_sum, 1e-8),
        "num_samples": len(pred),
        **masked_command_metrics(pred, target, mask),
        "steering_buckets": steering_target_metrics(pred, target),
    }


def _summary(metrics: dict[str, object]) -> str:
    return (
        f"loss={metrics['loss']:.3f} thr_mae={metrics['throttle_mae']:.3f} "
        f"steer_mae={metrics['steering_mae']:.3f} turn_acc={metrics['turn_accuracy']:.3f}"
    )


def _bucket_summary(metrics: dict[str, object]) -> str:
    buckets = metrics["steering_buckets"]

    def mae(name: str) -> str:
        return format_metric(buckets[name]["steering_mae"])

    return (
        f"steering MAE by target: center={mae('center')} medium={mae('medium')} strong={mae('strong')} "
        f"| left={mae('negative')} right={mae('positive')}"
    )


def write_history(history: list[dict[str, object]], out_dir: Path) -> None:
    """Write ``history.json`` and a flat ``history.csv`` with one row per epoch."""
    write_json(out_dir / "history.json", history)
    rows = [
        {
            "epoch": record["epoch"],
            "lr": record["lr"],
            **{
                f"{split}_{key}": value
                for split in ("train", "val")
                for key, value in record[split].items()
                if isinstance(value, int | float)
            },
        }
        for record in history
    ]
    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    cfg = TrainConfig(**vars(build_parser().parse_args(argv)))
    variant = resolve_variant(cfg.variant, history_length=cfg.history_length)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)

    train_sessions, val_sessions = discover_session_split(cfg.recordings_root)
    print(f"Train sessions: {[p.name for p in train_sessions]}")
    print(f"Val sessions:   {[p.name for p in val_sessions]}")
    print(f"Variant: {variant.name} (history={variant.history_length})")

    data_options = {
        "image_size": cfg.image_size,
        "history_length": variant.history_length,
        "history_max_gap_s": cfg.history_max_gap_s,
    }
    seed_ds = KeyecarSessionDataset(train_sessions, augment=cfg.augment, seed=cfg.seed, **data_options)
    val_ds = KeyecarSessionDataset(val_sessions, augment=False, seed=cfg.seed, **data_options)
    if cfg.quick:
        seed_ds.subset_first(200)
        val_ds.subset_first(100)

    correction_sessions, correction_provenance = load_accepted_correction_sessions(cfg.correction_manifest)
    exact_ds = CorrectionWindowDataset(
        correction_sessions, source=EXACT_SOURCE, augment=cfg.augment, seed=cfg.seed + 1, **data_options
    )
    pseudo_ds = CorrectionWindowDataset(
        correction_sessions, source=PSEUDO_SOURCE, augment=cfg.augment, seed=cfg.seed + 2, **data_options
    )
    train_ds = DaggerMixedDataset(seed_ds, exact_ds, pseudo_ds)
    batch_sampler = DaggerBatchSampler(
        train_ds,
        batch_counts={
            SEED_SOURCE: cfg.seed_per_batch,
            EXACT_SOURCE: cfg.exact_per_batch,
            PSEUDO_SOURCE: cfg.pseudo_per_batch,
        },
        steps_per_epoch=5 if cfg.quick else cfg.steps_per_epoch,
        seed=cfg.seed,
    )
    print(f"Train samples per source: {train_ds.source_counts}  Val samples: {len(val_ds)}")
    print(f"Demonstrations per condition (FORWARD, LEFT, RIGHT): {seed_ds.condition_counts}")

    loader_options = {
        "num_workers": cfg.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": cfg.num_workers > 0,
    }
    train_loader = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        generator=torch.Generator().manual_seed(cfg.seed),
        **loader_options,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, **loader_options)

    model = RegressionPolicy(
        input_channels=variant.input_channels, condition_history_length=variant.condition_history_length
    ).to(device)
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    metadata = {
        "schema_version": 1,
        "artifact": "keyecar_regression_policy",
        "variant": variant.to_dict(),
        "model_config": model.export_config(),
        "output": {
            "names": ["throttle", "steering"],
            "ranges": {"throttle": [0.0, 1.0], "steering": [-1.0, 1.0]},
        },
        "image_size": cfg.image_size,
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
        "recordings_root": str(cfg.recordings_root),
        "train_sessions": [str(p) for p in train_sessions],
        "val_sessions": [str(p) for p in val_sessions],
        "augment": cfg.augment,
        "augment_config": asdict(seed_ds.augment_config),
        "training_config": {
            **cfg.to_json(),
            "batch_size": cfg.batch_size,
            "steps_per_epoch": len(batch_sampler),
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "loss": "SmoothL1",
            "checkpoint_selection": "lowest validation loss (recorded motor state)",
        },
        "temporal_context": {
            "history_length": variant.history_length,
            "history_order": "newest_first",
            "history_max_gap_s": cfg.history_max_gap_s,
            "gap_padding": "repeat_oldest_consecutive_frame",
            "motor_state": "previous_executed_command",
        },
        "dagger_training": {
            **correction_provenance,
            "source_sample_counts": train_ds.source_counts,
            "batch_composition": batch_sampler.batch_counts,
            "correction_sampling": "uniform_intervention_then_uniform_frame",
            "exact_human": exact_ds.stats(),
            "pseudo_window": pseudo_ds.stats(),
            "exclusion_counts": correction_exclusion_counts(correction_sessions),
        },
    }
    write_json(cfg.out_dir / "policy_metadata.json", metadata)

    history = []
    best_loss = math.inf
    for epoch in range(cfg.epochs):
        batch_sampler.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, cfg=cfg, device=device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_loader, cfg=cfg, device=device)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        write_history(history, cfg.out_dir)
        print(f"epoch {epoch:02d}  train {_summary(train_metrics)}  |  val {_summary(val_metrics)}")
        print(f"  train {_bucket_summary(train_metrics)}")
        print(f"  val   {_bucket_summary(val_metrics)}")

        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "config": cfg.to_json(),
            "variant": variant.to_dict(),
            "metadata": metadata,
        }
        torch.save(checkpoint, cfg.out_dir / "last.pt")
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            torch.save(checkpoint, cfg.out_dir / "best.pt")

    print(f"Saved best checkpoint to {cfg.out_dir / 'best.pt'} (validation loss {best_loss:.3f})")


if __name__ == "__main__":
    main()
