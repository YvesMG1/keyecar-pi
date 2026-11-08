"""Render a recorded session as an annotated review video, or as a showcase GIF.

    python -m keyecar.training.tools.review recordings/<session>
    python -m keyecar.training.tools.review recordings/<session> --style minimal \\
        --start-frame 2050 --end-frame 2200 --stop-at-intervention --gif docs/media/onboard.gif

``--style full`` (default) shows every logged field below the frame: condition,
control and label source, training eligibility, stop and emergency-stop state,
ultrasonic distance, and the human and executed commands. Every session is
reviewed this way before it is used for training.

``--style minimal`` draws only the route condition and the executed throttle
and steering on top of the frame. It is used for the README media.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from keyecar.cli import add_keep_frames, make_parser
from keyecar.training.recordings import cell_flag, cell_float, cell_text, read_metadata, read_records

from . import drawing
from .video import VideoEncodeError, encode_gif, encode_mp4, frames_directory, recorded_fps

CANVAS_W = 640
IMAGE_H = 480
HUD_H = 220


class ReviewError(RuntimeError):
    """Raised when a recording cannot be rendered."""


@dataclass(frozen=True)
class ReviewFrame:
    row: dict[str, str]
    image_path: Path
    elapsed_s: float

    @property
    def frame_index(self) -> int:
        return int(self.row["frame_index"])

    def number(self, column: str) -> float | None:
        return cell_float(self.row, column)

    def flag(self, column: str) -> bool:
        return cell_flag(self.row, column)

    def text(self, column: str) -> str:
        return cell_text(self.row, column) or "-"


def load_review_frames(session_dir: Path) -> list[ReviewFrame]:
    try:
        rows = read_records(session_dir)
    except FileNotFoundError as exc:
        raise ReviewError(str(exc)) from exc
    if not rows:
        raise ReviewError(f"{session_dir / 'records.csv'} has no frames")
    first_ns = int(rows[0]["timestamp_monotonic_ns"])
    return [
        ReviewFrame(
            row=row,
            image_path=session_dir / row["image_path"],
            elapsed_s=max(0.0, (int(row["timestamp_monotonic_ns"]) - first_ns) / 1e9),
        )
        for row in rows
    ]


def _open_image(frame: ReviewFrame) -> Image.Image:
    if not frame.image_path.is_file():
        raise ReviewError(f"Missing image for frame {frame.frame_index}: {frame.image_path}")
    return Image.open(frame.image_path).convert("RGB")


def _ultrasonic_text(frame: ReviewFrame) -> str:
    distance = frame.number("ultrasonic_filtered_distance_cm")
    if distance is None:
        distance = frame.number("ultrasonic_raw_distance_cm")
    status = frame.text("ultrasonic_status")
    return status if distance is None else f"{status} {distance:.1f}cm"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _draw_command(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, frame: ReviewFrame, color) -> None:
    font, small = drawing.load_font(13), drawing.load_font(11)
    throttle, steering = frame.number(f"{label}_throttle"), frame.number(f"{label}_steering")
    t_text = "-" if throttle is None else f"{throttle:+.2f}"
    s_text = "-" if steering is None else f"{steering:+.2f}"
    draw.text((x, y), f"{label}  t={t_text}  s={s_text}", fill=color, font=font)
    drawing.draw_bar(draw, x=x, y=y + 28, width=250, height=16, value=throttle, color=color)
    draw.text((x + 258, y + 25), "thr", fill=drawing.MUTED, font=small)
    drawing.draw_bar(draw, x=x, y=y + 58, width=250, height=16, value=steering, color=color)
    draw.text((x + 258, y + 55), "steer", fill=drawing.MUTED, font=small)


def render_review_frame(
    *, frame: ReviewFrame, metadata: dict[str, object], output_path: Path, sequence_index: int
) -> Path:
    """Draw the camera frame with every logged field in a panel below it."""
    canvas, draw = drawing.hud_canvas(_open_image(frame), CANVAS_W, IMAGE_H, HUD_H)
    font, small = drawing.load_font(15), drawing.load_font(13)
    y = IMAGE_H

    row = frame.row
    session_id = metadata.get("session_id") or row.get("session_id", "")
    layout = metadata.get("layout_id") or row.get("layout_id") or "-"
    phase = metadata.get("phase") or row.get("phase") or "-"
    draw.text(
        (14, y + 10),
        f"frame={frame.frame_index}  t={frame.elapsed_s:06.2f}s  cond={frame.text('condition_name')}",
        fill=drawing.TEXT,
        font=font,
    )
    draw.text((14, y + 34), f"session={session_id}", fill=drawing.MUTED, font=small)
    draw.text(
        (14, y + 54),
        f"layout={layout}  phase={phase}  seq={sequence_index:06d}",
        fill=drawing.MUTED,
        font=small,
    )
    draw.text(
        (14, y + 80),
        f"source={frame.text('control_source')}  mode={frame.text('mode')}  "
        f"label={frame.text('label_source')}  eligible={_yes_no(frame.flag('training_eligible'))}",
        fill=drawing.TEXT,
        font=small,
    )
    stopped = frame.flag("stop_pressed") or frame.flag("safety_emergency_stop")
    draw.text(
        (14, y + 102),
        f"stop={_yes_no(frame.flag('stop_pressed'))}  emergency={_yes_no(frame.flag('safety_emergency_stop'))}  "
        f"ultra_stop={_yes_no(frame.flag('ultrasonic_emergency_stop'))}  ultra={_ultrasonic_text(frame)}",
        fill=drawing.RED if stopped else drawing.TEXT,
        font=small,
    )
    _draw_command(draw, 14, y + 132, "human", frame, drawing.BLUE)
    _draw_command(draw, 330, y + 132, "executed", frame, drawing.GREEN)
    return drawing.save_frame(canvas, output_path)


def render_minimal_frame(*, frame: ReviewFrame, output_path: Path, width: int = 480) -> Path:
    """Draw only the route condition and the executed command on top of the camera frame."""
    image = _open_image(frame)
    image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    scale = width / 480.0
    image = drawing.draw_condition_badge(image, frame.text("condition_name"), scale=scale)
    image = drawing.draw_command_bars(
        image,
        throttle=frame.number("executed_throttle") or 0.0,
        steering=frame.number("executed_steering") or 0.0,
        scale=scale,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(output_path)
    return output_path


def _is_takeover(frame: ReviewFrame) -> bool:
    return frame.text("intervention_state") == "manual_override" or frame.text("control_source") == "manual"


def select_frames(
    frames: list[ReviewFrame],
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    stop_at_intervention: bool = False,
    stride: int = 1,
    max_frames: int | None = None,
) -> list[ReviewFrame]:
    """Pick the frames to render. ``stop_at_intervention`` ends the clip before the first takeover."""
    selected = [
        frame
        for frame in frames
        if (start_frame is None or frame.frame_index >= start_frame)
        and (end_frame is None or frame.frame_index <= end_frame)
    ]
    if stop_at_intervention:
        takeover = next((i for i, frame in enumerate(selected) if _is_takeover(frame)), None)
        selected = selected[:takeover]
    return selected[::stride][:max_frames]


def generate_review_video(
    *,
    session_dir: Path,
    output_dir: Path = Path("runs/reviews"),
    fps: float | None = None,
    stride: int = 1,
    max_frames: int | None = None,
    keep_frames: bool = False,
    style: str = "full",
    start_frame: int | None = None,
    end_frame: int | None = None,
    stop_at_intervention: bool = False,
    gif_path: Path | None = None,
    width: int = 480,
) -> Path:
    """Render ``<output_dir>/<session_id>/review.mp4`` (or ``gif_path``) and return its path.

    ``fps=None`` plays back at the recorded rate (divided by ``stride``).
    """
    metadata = read_metadata(session_dir)
    frames = load_review_frames(session_dir)
    selected = select_frames(
        frames,
        start_frame=start_frame,
        end_frame=end_frame,
        stop_at_intervention=stop_at_intervention,
        stride=stride,
        max_frames=max_frames,
    )
    if not selected:
        raise ReviewError("No frames selected")

    session_id = str(metadata.get("session_id") or session_dir.name)
    review_dir = output_dir / session_id
    output_path = gif_path or review_dir / "review.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if fps is None:
        timestamps = [int(frame.row["timestamp_monotonic_ns"]) for frame in frames]
        fps = recorded_fps(timestamps, metadata) / stride
    fps = max(0.1, fps)
    if gif_path is not None:
        fps = float(round(fps))  # GIF frame delays are whole centiseconds

    extension = "png" if gif_path else "jpg"  # GIFs are encoded from lossless frames
    with frames_directory(review_dir / "frames" if keep_frames else None) as frames_dir:
        for i, frame in enumerate(selected):
            path = frames_dir / f"frame_{i:06d}.{extension}"
            if style == "minimal":
                render_minimal_frame(frame=frame, output_path=path, width=width)
            else:
                render_review_frame(frame=frame, metadata=metadata, output_path=path, sequence_index=i)
        if gif_path is not None:
            encode_gif(frames_dir=frames_dir, fps=fps, output_path=output_path)
        else:
            encode_mp4(frames_dir=frames_dir, fps=fps, output_path=output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser(__doc__)
    parser.add_argument("session_dir", type=Path, help="recording session folder")
    parser.add_argument(
        "--style", choices=["full", "minimal"], default="full", help="what to draw (see above)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/reviews"),
        help="review.mp4 goes to <output-dir>/<session>/",
    )
    parser.add_argument("--gif", type=Path, help="write a looping GIF here instead of review.mp4")

    frames = parser.add_argument_group("frames")
    frames.add_argument("--start-frame", type=int, help="first frame_index to render")
    frames.add_argument("--end-frame", type=int, help="last frame_index to render")
    frames.add_argument(
        "--stop-at-intervention", action="store_true", help="end the clip before the first human takeover"
    )
    frames.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    frames.add_argument("--max-frames", type=int, help="limit the number of rendered frames")
    frames.add_argument("--fps", type=float, help="playback rate (default: the recorded rate)")
    frames.add_argument("--width", type=int, default=480, help="frame width for --style minimal")
    add_keep_frames(frames)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    try:
        output_path = generate_review_video(
            session_dir=args.session_dir,
            output_dir=args.output_dir,
            fps=args.fps,
            stride=args.stride,
            max_frames=args.max_frames,
            keep_frames=args.keep_frames,
            style=args.style,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            stop_at_intervention=args.stop_at_intervention,
            gif_path=args.gif,
            width=args.width,
        )
    except (ReviewError, VideoEncodeError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
