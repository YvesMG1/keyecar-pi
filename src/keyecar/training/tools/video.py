"""Encode rendered frames (``frame_000000.jpg``, ...) to MP4 or GIF with ffmpeg."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class VideoEncodeError(RuntimeError):
    """Raised when ffmpeg is missing or fails."""


@contextmanager
def frames_directory(keep_dir: Path | None = None) -> Iterator[Path]:
    """Yield a folder for rendered frames: ``keep_dir`` (emptied first) or a temporary one."""
    if keep_dir is None:
        with tempfile.TemporaryDirectory(prefix="keyecar_frames_") as tmp:
            yield Path(tmp)
    else:
        shutil.rmtree(keep_dir, ignore_errors=True)
        keep_dir.mkdir(parents=True)
        yield keep_dir


def encode_mp4(*, frames_dir: Path, fps: float, output_path: Path) -> None:
    """Encode ``frame_%06d.jpg`` files as H.264."""
    inputs = ["-framerate", f"{fps:.3f}", "-i", str(frames_dir / "frame_%06d.jpg")]
    h264 = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    _run(["ffmpeg", "-y", *inputs, *h264, str(output_path)])


def encode_gif(*, frames_dir: Path, fps: float, output_path: Path, max_colors: int = 128) -> None:
    """Encode ``frame_%06d.png`` files as a looping GIF in two passes.

    A palette computed over all frames (``stats_mode=full``) keeps the colors of
    static overlays exact, and no dithering keeps the file small.
    """
    ffmpeg = ["ffmpeg", "-y", "-v", "error"]
    inputs = ["-framerate", f"{fps:.3f}", "-i", str(frames_dir / "frame_%06d.png")]
    palette = str(frames_dir / "palette.png")
    _run([*ffmpeg, *inputs, "-vf", f"palettegen=max_colors={max_colors}:stats_mode=full", palette])
    paletteuse = "paletteuse=dither=none:diff_mode=rectangle"
    _run([*ffmpeg, *inputs, "-i", palette, "-lavfi", paletteuse, "-loop", "0", str(output_path)])


def _run(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise VideoEncodeError(
            "ffmpeg not found. Install it with `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg`."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise VideoEncodeError(f"ffmpeg failed with exit code {exc.returncode}: {' '.join(command)}") from exc


def recorded_fps(timestamps_ns, metadata: dict[str, object]) -> float:
    """The rate a session was recorded at, falling back to the camera fps in its metadata."""
    camera = metadata.get("camera", {})
    return infer_fps_from_monotonic_ns(timestamps_ns, default=float(camera.get("fps", 20.0)))


def infer_fps_from_monotonic_ns(timestamps_ns, *, default: float = 20.0) -> float:
    """Playback rate from frame timestamps, using the median gap so dropped frames don't skew it."""
    timestamps = np.asarray(timestamps_ns, dtype=np.float64)
    deltas = np.diff(np.sort(timestamps[np.isfinite(timestamps)]))
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return float(default)
    return 1e9 / float(np.median(deltas))
