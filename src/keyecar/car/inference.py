"""On-Pi runtime for an exported driving policy.

Loads an ONNX bundle written by :mod:`keyecar.training.export` (``policy.onnx``
or ``policy.int8.onnx`` next to ``policy_metadata.json``) and turns each camera
frame into a :class:`DriveCommand`.

The model inputs are the camera frame(s), the route condition, and the
previously executed command ("motor state"). History policies also get the last
N frames stacked along the channel axis plus the matching condition history.
The history is kept here, so the live input matches what the model saw in
training. During a manual takeover :meth:`PolicyRunner.observe` keeps adding
frames without running the model.

The preprocessing in this module (:func:`letterbox`, ImageNet normalization) is
also used by the training dataset, so training and the car cannot drift apart.

Benchmark on the Pi::

    python -m keyecar.car.inference models/rgb_history_dagger_v4
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from keyecar.cli import make_parser

from .control import Condition, DriveCommand

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox(image: Image.Image, size: int) -> Image.Image:
    """Resize to fit a ``size`` x ``size`` square without distortion, padding with black."""
    w, h = image.size
    scale = size / max(w, h)
    new_w, new_h = round(w * scale), round(h * scale)
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


@dataclass(frozen=True)
class PolicyMetadata:
    """The input layout and preprocessing stored in ``policy_metadata.json``."""

    variant_name: str
    image_size: int
    mean: np.ndarray
    std: np.ndarray
    history_length: int
    condition_history_length: int

    @classmethod
    def from_file(cls, path: str | Path) -> PolicyMetadata:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        variant = data["variant"]
        model_config = data["model_config"]
        history_length = int(variant["history_length"])
        if int(model_config["input_channels"]) != 3 * history_length:
            raise ValueError(f"{path}: input_channels does not match {history_length} RGB frames")
        return cls(
            variant_name=str(variant["name"]),
            image_size=int(data["image_size"]),
            mean=np.array(data["imagenet_mean"], dtype=np.float32).reshape(3, 1, 1),
            std=np.array(data["imagenet_std"], dtype=np.float32).reshape(3, 1, 1),
            history_length=history_length,
            condition_history_length=int(model_config["condition_history_length"]),
        )


@dataclass(frozen=True)
class InferenceDiagnostics:
    """Timing of the most recent policy pass."""

    timestamp_monotonic_ns: int
    total_ms: float
    policy_onnx_ms: float


def resolve_bundle(path: str | Path, *, int8: bool = False) -> tuple[Path, Path]:
    """Return ``(model_path, metadata_path)`` for a bundle directory or an ``.onnx`` file."""
    path = Path(path)
    if path.is_dir():
        model_path = path / ("policy.int8.onnx" if int8 else "policy.onnx")
        metadata_path = path / "policy_metadata.json"
    else:
        model_path = path
        metadata_path = path.with_name("policy_metadata.json")
    for required in (model_path, metadata_path):
        if not required.is_file():
            raise FileNotFoundError(f"Missing policy bundle file: {required}")
    return model_path, metadata_path


class PolicyRunner:
    """Runs the policy once per control tick and keeps its frame history."""

    def __init__(
        self,
        *,
        model_path: str | Path,
        metadata_path: str | Path,
        ema_alpha: float = 0.0,
        intra_op_num_threads: int = 4,
        diagnostics: bool = False,
    ):
        """``ema_alpha`` > 0 smooths the output but delays steering by about
        1 / (1 - alpha) ticks, which shows up as turning in late. The car drives
        with 0.
        """
        import onnxruntime as ort  # imported here so training code can use letterbox() without it

        self.metadata = PolicyMetadata.from_file(metadata_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_num_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.ema_alpha = float(ema_alpha)
        self.diagnostics = diagnostics
        self.last_diagnostics: InferenceDiagnostics | None = None
        self._smoothed: tuple[float, float] | None = None
        self._last_condition: Condition | None = None
        self._frames: list[np.ndarray] = []  # newest first
        self._conditions: list[int] = []  # newest first

    def reset(self) -> None:
        """Forget the frame history and smoothing (after a stop)."""
        self.reset_output_smoothing()
        self._last_condition = None
        self._frames.clear()
        self._conditions.clear()

    def reset_output_smoothing(self) -> None:
        """Forget the smoothed output but keep the frame history."""
        self._smoothed = None
        self.last_diagnostics = None

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        """Letterbox and normalize one frame to a CHW float32 array."""
        array = np.asarray(letterbox(image.convert("RGB"), self.metadata.image_size), dtype=np.float32)
        array = array.transpose(2, 0, 1) / 255.0
        return ((array - self.metadata.mean) / self.metadata.std).astype(np.float32)

    def _push(self, image: Image.Image, condition: Condition) -> None:
        """Add a frame to the history. The first frame fills the whole history."""
        if condition != self._last_condition:
            self._smoothed = None
            self._last_condition = condition
        frame = self._preprocess(image)
        length = self.metadata.history_length
        if self._frames:
            self._frames = [frame, *self._frames[: length - 1]]
            self._conditions = [int(condition), *self._conditions[: length - 1]]
        else:
            self._frames = [frame] * length
            self._conditions = [int(condition)] * length

    def observe(self, *, image: Image.Image, condition: Condition) -> None:
        """Add a frame without running the model (used while the human drives).

        This way the first prediction after a takeover sees the same human-driven
        history that the correction samples contain in training.
        """
        self.last_diagnostics = None
        self._push(image, condition)

    def predict(
        self, *, image: Image.Image, motor_state: tuple[float, float], condition: Condition
    ) -> DriveCommand:
        """Return the policy command for one frame.

        ``motor_state`` is the (throttle, steering) executed on the previous tick.
        """
        self.last_diagnostics = None
        timestamp_ns = time.monotonic_ns()
        started = time.perf_counter()
        self._push(image, condition)
        feed = {
            "image": np.concatenate(self._frames, axis=0)[None],
            "motor_state": np.array([motor_state], dtype=np.float32),
            "condition": np.array([int(condition)], dtype=np.int64),
        }
        if self.metadata.condition_history_length > 0:
            feed["condition_history"] = np.array([self._conditions], dtype=np.int64)

        policy_started = time.perf_counter()
        throttle, steering = self.session.run(None, feed)[0][0]
        policy_ms = (time.perf_counter() - policy_started) * 1000.0
        target = (float(np.clip(throttle, 0.0, 1.0)), float(np.clip(steering, -1.0, 1.0)))

        if self._smoothed is None:
            self._smoothed = target
        else:
            a = self.ema_alpha
            self._smoothed = (
                a * self._smoothed[0] + (1.0 - a) * target[0],
                a * self._smoothed[1] + (1.0 - a) * target[1],
            )

        if self.diagnostics:
            self.last_diagnostics = InferenceDiagnostics(
                timestamp_monotonic_ns=timestamp_ns,
                total_ms=(time.perf_counter() - started) * 1000.0,
                policy_onnx_ms=policy_ms,
            )
        return DriveCommand(*self._smoothed)


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser("Time preprocessing + inference on random 640x480 frames.")
    parser.add_argument("bundle", help="bundle folder or .onnx file")
    parser.add_argument("--int8", action="store_true", help="use policy.int8.onnx from a bundle folder")
    parser.add_argument("--iters", type=int, default=200, help="timed frames (after 10 warm-up frames)")
    parser.add_argument("--threads", type=int, default=4, help="ONNX Runtime intra-op threads")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_path, metadata_path = resolve_bundle(args.bundle, int8=args.int8)
    runner = PolicyRunner(
        model_path=model_path, metadata_path=metadata_path, intra_op_num_threads=args.threads
    )
    image = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    for _ in range(10):  # warm-up
        runner.predict(image=image, motor_state=(0.0, 0.0), condition=Condition.FORWARD)
    started = time.perf_counter()
    for _ in range(args.iters):
        runner.predict(image=image, motor_state=(0.0, 0.0), condition=Condition.FORWARD)
    ms = (time.perf_counter() - started) / args.iters * 1000
    print(f"{model_path}: {ms:.2f} ms per frame ({1000.0 / ms:.1f} Hz) over {args.iters} frames")


if __name__ == "__main__":
    main()
