"""Training dataset over recorded sessions (recording schema v2).

Each session is a folder written by :class:`keyecar.car.recording.SessionRecorder`
(``records.csv``, ``metadata.json``, ``images/``). A sample is:

* the current frame, or the last ``history_length`` frames (newest first) stacked
  along the channel axis;
* the route condition, and the conditions of the history frames;
* the previously executed command ("motor state");
* the target command: what the human did, or what was executed if no human
  command was recorded.

Images are letterboxed exactly as on the car (see :func:`keyecar.car.inference.letterbox`).
Augmentation is photometric only (color, shadows, blur, noise, JPEG artifacts)
plus horizontal mirroring, which also negates steering and swaps LEFT/RIGHT.
One set of augmentation parameters is drawn per sample and applied to every
frame of its history. Geometric augmentation (rotation, shifts) is deliberately
left out, because it changes the road geometry without changing the label.

Print split statistics and write an augmentation preview::

    python -m keyecar.training.dataset --recordings-root recordings --preview runs/augment_preview.jpg
"""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset, get_worker_info
from torchvision import transforms
from torchvision.transforms import functional as F

from keyecar.car.control import Condition, mirror_condition
from keyecar.car.inference import IMAGENET_MEAN, IMAGENET_STD, letterbox
from keyecar.cli import add_recordings_root, make_parser


@dataclass(frozen=True)
class AugmentConfig:
    """Photometric augmentation strengths and probabilities."""

    flip_prob: float = 0.5
    brightness: float = 0.45
    contrast: float = 0.40
    saturation: float = 0.25
    hue: float = 0.05
    shadow_prob: float = 0.55
    shadow_alpha_min: float = 0.25
    shadow_alpha_max: float = 0.60
    blur_prob: float = 0.45
    blur_radius_min: float = 0.35
    blur_radius_max: float = 1.50
    noise_prob: float = 0.30
    noise_std: float = 0.035
    jpeg_prob: float = 0.25
    jpeg_quality_min: int = 45
    jpeg_quality_max: int = 85


@dataclass(frozen=True)
class _PhotometricParams:
    """Augmentation drawn once per sample and applied to every history frame."""

    brightness: float
    contrast: float
    saturation: float
    hue: float
    shadow_points: tuple[tuple[int, int], ...]
    shadow_alpha: float
    blur_radius: float
    jpeg_quality: int
    noise_seed: int


# ---------------------------------------------------------------------------
# Loading sessions
# ---------------------------------------------------------------------------


def _session_dirs(split_dir: Path) -> list[Path]:
    if not split_dir.is_dir():
        raise ValueError(
            f"Expected a split directory at {split_dir} "
            "(recordings/train/<session>/ and recordings/val/<session>/)."
        )
    return sorted(child for child in split_dir.iterdir() if (child / "records.csv").is_file())


def discover_session_split(recordings_root: Path) -> tuple[list[Path], list[Path]]:
    """Return the train and val sessions. Sessions are assigned to splits by hand, never by frame."""
    root = Path(recordings_root)
    train_sessions, val_sessions = _session_dirs(root / "train"), _session_dirs(root / "val")
    if not train_sessions:
        raise ValueError(f"No training sessions found in {root / 'train'}.")
    if not val_sessions:
        raise ValueError(f"No validation sessions found in {root / 'val'}.")
    return train_sessions, val_sessions


def _flag(column: pd.Series) -> pd.Series:
    """Read a 0/1 column; empty cells count as 0."""
    return pd.to_numeric(column, errors="coerce").fillna(0) != 0


def load_session_frames(session_dir: Path) -> pd.DataFrame:
    """Read ``records.csv`` and add the target and previous-command columns."""
    df = pd.read_csv(session_dir / "records.csv", dtype={"session_id": str, "frame_id": str})
    if not (df["schema_version"] == 2).all():
        raise ValueError(f"{session_dir} is not a schema-v2 recording")

    human_throttle = pd.to_numeric(df["human_throttle"], errors="coerce")
    human_steering = pd.to_numeric(df["human_steering"], errors="coerce")
    executed_throttle = pd.to_numeric(df["executed_throttle"], errors="coerce").fillna(0.0)
    executed_steering = pd.to_numeric(df["executed_steering"], errors="coerce").fillna(0.0)
    # The target is the human command where there is one, otherwise the executed command.
    df["throttle"] = human_throttle.fillna(executed_throttle).astype(np.float32)
    df["steering"] = human_steering.fillna(executed_steering).astype(np.float32)
    # The motor-state input is the command executed one frame earlier.
    df["prev_throttle"] = executed_throttle.shift(1).fillna(0.0).astype(np.float32)
    df["prev_steering"] = executed_steering.shift(1).fillna(0.0).astype(np.float32)
    df["session_dir"] = str(session_dir)
    return df


def _training_frames(df: pd.DataFrame) -> pd.DataFrame:
    """Keep training-eligible human frames, without stops and without reversing."""
    keep = _flag(df["training_eligible"]) & (df["label_source"] == "human") & (df["throttle"] >= 0.0)
    for column in ("stop_pressed", "safety_emergency_stop", "ultrasonic_emergency_stop"):
        keep &= ~_flag(df[column])
    return df[keep]


def _jitter_factor(rng: np.random.Generator, amount: float) -> float:
    if amount <= 0.0:
        return 1.0
    return float(rng.uniform(max(0.0, 1.0 - amount), 1.0 + amount))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class KeyecarSessionDataset(Dataset):
    """Regression samples from one or more recorded sessions.

    With ``filter_targets=True`` (human demonstrations) only training-eligible
    human frames become samples. The DAgger correction datasets pass
    ``filter_targets=False`` and pick their frames from the reviewed sidecars.

    History frames always come from the full, unfiltered session. They must be
    consecutive (``frame_index`` + 1, at most ``history_max_gap_s`` apart); at a
    gap, the oldest consecutive frame is repeated.
    """

    def __init__(
        self,
        session_dirs: Iterable[Path],
        *,
        image_size: int = 160,
        history_length: int = 1,
        history_max_gap_s: float = 0.2,
        augment: bool = True,
        augment_config: AugmentConfig | None = None,
        filter_targets: bool = True,
        seed: int | None = None,
    ):
        if history_length <= 0:
            raise ValueError("history_length must be positive")
        if history_max_gap_s <= 0.0:
            raise ValueError("history_max_gap_s must be positive")
        self.image_size = int(image_size)
        self.history_length = int(history_length)
        self.history_max_gap_ns = round(history_max_gap_s * 1e9)
        self.augment = augment
        self.augment_config = augment_config or AugmentConfig()

        self._sessions: dict[str, pd.DataFrame] = {}
        samples: list[pd.DataFrame] = []
        for session_dir in map(Path, session_dirs):
            frames = load_session_frames(session_dir)
            frames["_raw_row"] = np.arange(len(frames))
            self._sessions[str(session_dir)] = frames
            samples.append(_training_frames(frames) if filter_targets else frames)
        if not samples:
            raise ValueError("No session directories were given")
        self.df = pd.concat(samples, ignore_index=True)
        if self.df.empty:
            raise ValueError("No training samples in the given sessions")
        self.image_paths = [
            Path(d) / p for d, p in zip(self.df["session_dir"], self.df["image_path"], strict=True)
        ]

        self._normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self._seed = seed
        self._worker_seed: int | None = None
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def condition_counts(self) -> list[int]:
        return np.bincount(self.df["condition"].to_numpy(dtype=np.int64), minlength=3).tolist()

    def stats(self) -> dict[str, object]:
        return {
            "num_samples": len(self.df),
            "condition_counts": self.condition_counts,
            "history_length": self.history_length,
            "history_max_gap_s": self.history_max_gap_ns / 1e9,
        }

    def subset_first(self, n: int) -> None:
        """Keep only the first ``n`` samples (for quick smoke runs)."""
        self.df = self.df.iloc[:n].reset_index(drop=True)
        self.image_paths = self.image_paths[:n]

    # -- augmentation -------------------------------------------------------

    def _ensure_worker_rng(self) -> None:
        """Give each DataLoader worker its own reproducible random stream."""
        worker = get_worker_info()
        if worker is None or self._worker_seed == worker.seed:
            return
        base_seed = 0 if self._seed is None else self._seed
        self._rng = np.random.default_rng((worker.seed + base_seed) % (2**63 - 1))
        self._worker_seed = worker.seed

    def _sample_photometric_params(self) -> _PhotometricParams:
        cfg, rng = self.augment_config, self._rng
        shadow_points: tuple[tuple[int, int], ...] = ()
        shadow_alpha = 0.0
        if cfg.shadow_prob > 0.0 and rng.random() < cfg.shadow_prob:
            shadow_points = tuple(
                (int(rng.integers(0, self.image_size)), int(rng.integers(0, self.image_size)))
                for _ in range(int(rng.integers(3, 6)))
            )
            shadow_alpha = float(rng.uniform(cfg.shadow_alpha_min, cfg.shadow_alpha_max))
        blur_radius = 0.0
        if cfg.blur_prob > 0.0 and rng.random() < cfg.blur_prob:
            blur_radius = float(rng.uniform(cfg.blur_radius_min, cfg.blur_radius_max))
        jpeg_quality = 0
        if cfg.jpeg_prob > 0.0 and rng.random() < cfg.jpeg_prob:
            jpeg_quality = int(rng.integers(cfg.jpeg_quality_min, cfg.jpeg_quality_max + 1))
        noise_seed = int(rng.integers(0, 2**32 - 1))
        return _PhotometricParams(
            brightness=_jitter_factor(rng, cfg.brightness),
            contrast=_jitter_factor(rng, cfg.contrast),
            saturation=_jitter_factor(rng, cfg.saturation),
            hue=float(rng.uniform(-cfg.hue, cfg.hue)) if cfg.hue > 0.0 else 0.0,
            shadow_points=shadow_points,
            shadow_alpha=shadow_alpha,
            blur_radius=blur_radius,
            jpeg_quality=jpeg_quality,
            noise_seed=noise_seed,
        )

    def _augment_photometric(self, image: Image.Image, params: _PhotometricParams) -> Image.Image:
        cfg = self.augment_config
        image = F.adjust_brightness(image, params.brightness)
        image = F.adjust_contrast(image, params.contrast)
        image = F.adjust_saturation(image, params.saturation)
        if params.hue != 0.0:
            image = F.adjust_hue(image, params.hue)
        if params.shadow_points:
            overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
            shadow = (0, 0, 0, int(255 * params.shadow_alpha))
            ImageDraw.Draw(overlay).polygon(params.shadow_points, fill=shadow)
            image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        if params.blur_radius > 0.0:
            image = image.filter(ImageFilter.GaussianBlur(radius=params.blur_radius))
        if cfg.noise_prob > 0.0 and cfg.noise_std > 0.0:
            # Seeded per sample, so every frame of a history gets the same noise.
            noise_rng = np.random.default_rng(params.noise_seed)
            if noise_rng.random() < cfg.noise_prob:
                array = np.asarray(image, dtype=np.float32) / 255.0
                array = np.clip(array + noise_rng.normal(0.0, cfg.noise_std, size=array.shape), 0.0, 1.0)
                image = Image.fromarray((array * 255.0).astype(np.uint8), mode="RGB")
        if params.jpeg_quality > 0:
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=params.jpeg_quality)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
        return image

    def _prepare_image(
        self, image: Image.Image, *, flipped: bool, photometric: _PhotometricParams | None
    ) -> Image.Image:
        image = letterbox(image, self.image_size)
        if flipped:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if photometric is not None:
            image = self._augment_photometric(image, photometric)
        return image

    # -- samples ------------------------------------------------------------

    def _history_rows(self, sample: pd.Series) -> list[pd.Series]:
        """Return the sample's frame followed by up to ``history_length - 1`` earlier frames."""
        frames = self._sessions[sample["session_dir"]]
        position = int(sample["_raw_row"])
        rows = [frames.iloc[position]]
        while len(rows) < self.history_length and position > 0:
            newer, older = frames.iloc[position], frames.iloc[position - 1]
            gap_ns = newer["timestamp_monotonic_ns"] - older["timestamp_monotonic_ns"]
            if newer["frame_index"] - older["frame_index"] != 1 or not 0 < gap_ns <= self.history_max_gap_ns:
                break
            rows.append(older)
            position -= 1
        rows += [rows[-1]] * (self.history_length - len(rows))
        return rows

    def _image_tensor(
        self, row: pd.Series, *, flipped: bool, photometric: _PhotometricParams | None
    ) -> torch.Tensor:
        image = Image.open(Path(row["session_dir"]) / row["image_path"]).convert("RGB")
        image = self._prepare_image(image, flipped=flipped, photometric=photometric)
        return self._normalize(F.to_tensor(image))

    def get(self, index: int, *, target: tuple[float, float] | None = None) -> dict[str, object]:
        """Return sample ``index``; ``target`` replaces the recorded (throttle, steering) target."""
        self._ensure_worker_rng()
        rows = self._history_rows(self.df.iloc[index])
        current = rows[0]

        flipped = bool(self.augment and self._rng.random() < self.augment_config.flip_prob)
        photometric = self._sample_photometric_params() if self.augment else None
        image = torch.cat(
            [self._image_tensor(row, flipped=flipped, photometric=photometric) for row in rows], dim=0
        )

        throttle, steering = target or (float(current["throttle"]), float(current["steering"]))
        prev_steering = float(current["prev_steering"])
        conditions = [Condition(int(row["condition"])) for row in rows]
        if flipped:
            steering, prev_steering = -steering, -prev_steering
            conditions = [mirror_condition(condition) for condition in conditions]

        return {
            "image": image,
            "motor_state": torch.tensor(
                [float(current["prev_throttle"]), prev_steering], dtype=torch.float32
            ),
            "condition": torch.tensor(int(conditions[0]), dtype=torch.long),
            "condition_history": torch.tensor([int(c) for c in conditions], dtype=torch.long),
            "target_command": torch.tensor([throttle, steering], dtype=torch.float32),
            # Per-output loss mask and weight; correction samples override these.
            "supervision_mask": torch.ones(2),
            "output_weights": torch.ones(2),
            "session_id": str(current["session_id"]),
            "frame_id": str(current["frame_id"]),
        }

    def __getitem__(self, index: int) -> dict[str, object]:
        return self.get(index)

    def augmentation_preview(self, index: int = 0, num_variants: int = 8) -> Image.Image:
        """The letterboxed frame of sample ``index`` and augmented versions of it, as one image."""
        source = Image.open(self.image_paths[index]).convert("RGB")
        variants = [letterbox(source, self.image_size)]
        for _ in range(num_variants):
            flipped = bool(self._rng.random() < self.augment_config.flip_prob)
            photometric = self._sample_photometric_params()
            variants.append(self._prepare_image(source, flipped=flipped, photometric=photometric))

        cols = min(3, len(variants))
        rows = -(-len(variants) // cols)
        pad, size = 8, self.image_size
        sheet = Image.new(
            "RGB", (cols * size + (cols + 1) * pad, rows * size + (rows + 1) * pad), (24, 24, 24)
        )
        for i, image in enumerate(variants):
            sheet.paste(image, (pad + (i % cols) * (size + pad), pad + (i // cols) * (size + pad)))
        return sheet


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("Print training-split stats and write an augmentation preview.")
    add_recordings_root(parser)
    parser.add_argument("--image-size", type=int, default=160, help="side of the letterboxed square input")
    parser.add_argument("--history-length", type=int, default=1, help="frames per sample")
    parser.add_argument(
        "--preview", type=Path, default=Path("runs/augment_preview.jpg"), help="where to write the preview"
    )
    parser.add_argument("--sample-index", type=int, default=0, help="training sample to preview")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_sessions, _ = discover_session_split(args.recordings_root)
    dataset = KeyecarSessionDataset(
        train_sessions, image_size=args.image_size, history_length=args.history_length
    )
    print(json.dumps(dataset.stats(), indent=2))
    sample = dataset[0]
    print(
        f"first sample: image={tuple(sample['image'].shape)} motor_state={sample['motor_state'].tolist()} "
        f"condition={int(sample['condition'])} target={sample['target_command'].tolist()}"
    )
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    dataset.augmentation_preview(args.sample_index).save(args.preview)
    print(f"Wrote augmentation preview: {args.preview}")


if __name__ == "__main__":
    main()
