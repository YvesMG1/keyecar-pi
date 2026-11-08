"""Build DAgger correction targets from recorded takeovers and render their review videos.

    python -m keyecar.training.tools.intervention_windows recordings/session_v2_...  [more sessions]

The first run creates ``recordings/correction_sessions.json`` with every session
``pending``. Each run then writes the correction sidecar of every non-rejected
session (``recordings/annotations/<session_id>/intervention_windows.json``) and
a review video that walks through each takeover frame by frame.

To accept a session after watching its video, edit the manifest: set
``review_status`` to ``accepted`` and copy ``source_hashes`` and
``sidecar_sha256`` into ``accepted_source_hashes`` and ``accepted_sidecar_sha256``.
There is deliberately no command for this. If a recording or sidecar changes
later, the next run puts the session back to ``pending``.

How the targets are computed is described in :mod:`keyecar.training.corrections`.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from keyecar.cli import add_keep_frames, make_parser
from keyecar.training.corrections import (
    DEFAULT_MANIFEST_PATH,
    REGIONS,
    CorrectionDataError,
    RecordingFrame,
    WindowConfig,
    initialize_manifest,
    load_correction_manifest,
    load_intervention_window_sidecar,
    load_recording_session,
    save_manifest,
    validate_manifest_artifacts,
    write_intervention_window_sidecar,
)
from keyecar.training.files import sha256_file

from . import drawing
from .video import VideoEncodeError, encode_mp4, frames_directory, recorded_fps

CANVAS_W = 800
IMAGE_H = 480
HUD_H = 260


def _fmt(value: object) -> str:
    return "-" if value is None else f"{float(value):+.3f}"


def _labelled_bar(draw: ImageDraw.ImageDraw, y: int, label: str, value: float | None, color) -> None:
    draw.text((14, y), label, fill=color, font=drawing.load_font(13))
    drawing.draw_bar(draw, x=76, y=y, width=520, height=18, value=value, color=color)
    draw.text((606, y + 2), f"{label} {_fmt(value)}", fill=color, font=drawing.load_font(11))


def render_intervention_review_frame(
    *,
    frame: RecordingFrame,
    entry: dict[str, object],
    intervention: dict[str, object],
    output_path: Path,
    sequence_index: int,
) -> Path:
    """Draw one frame with its model, human, and target steering and why it is (not) used."""
    image = Image.open(frame.image_path).convert("RGB")
    canvas, draw = drawing.hud_canvas(image, CANVAS_W, IMAGE_H, HUD_H)
    font, small, tiny = drawing.load_font(16), drawing.load_font(13), drawing.load_font(11)
    y = IMAGE_H

    included = bool(entry["included"])
    draw.text(
        (14, y + 10),
        f"{entry['intervention_id']}  region={entry['region']}  "
        f"frame={frame.frame_index}  condition={frame.condition_name}",
        fill=drawing.TEXT,
        font=font,
    )
    draw.text(
        (14, y + 34),
        f"{'INCLUDED' if included else 'EXCLUDED'}: {entry['exclusion_reason'] or 'included'}  seq={sequence_index:06d}",
        fill=drawing.GREEN if included else drawing.RED,
        font=small,
    )
    draw.text(
        (14, y + 56),
        f"anchor model={_fmt(intervention['model_anchor_steering'])}  "
        f"human={_fmt(intervention['human_anchor_steering'])}  "
        f"delta={_fmt(entry['correction_delta'])}  alpha={_fmt(entry['alpha'])}",
        fill=drawing.MUTED,
        font=small,
    )
    human = entry["exact_human_steering"]
    if human is None:
        human = entry["human_anchor_steering"]
    _labelled_bar(draw, y + 88, "model", entry["recorded_model_steering"], drawing.ORANGE)
    _labelled_bar(draw, y + 128, "human", human, drawing.BLUE)
    _labelled_bar(draw, y + 168, "target", entry["targets"]["steering"], drawing.GREEN)
    draw.text(
        (14, y + 216),
        f"exact human command: throttle={_fmt(entry['exact_human_throttle'])}  "
        f"steering={_fmt(entry['exact_human_steering'])}  median steering={_fmt(entry['human_anchor_steering'])}",
        fill=drawing.MUTED,
        font=small,
    )
    draw.text(
        (14, y + 238),
        f"intervention status={intervention['status']}  source={frame.row.get('control_source', '')}  "
        f"eligible={'yes' if frame.training_eligible else 'no'}",
        fill=drawing.MUTED,
        font=tiny,
    )
    return drawing.save_frame(canvas, output_path)


def render_title_card(intervention: dict[str, object], output_path: Path) -> Path:
    canvas = Image.new("RGB", (CANVAS_W, IMAGE_H + HUD_H), drawing.HUD_BG)
    draw = ImageDraw.Draw(canvas)
    title, font = drawing.load_font(28), drawing.load_font(17)
    status_color = drawing.GREEN if intervention["status"] == "usable" else drawing.YELLOW
    draw.text((36, 70), f"Intervention {intervention['intervention_id']}", fill=drawing.TEXT, font=title)
    draw.text(
        (36, 125),
        f"frames {intervention['start_frame_index']}-{intervention['end_frame_index']}  "
        f"status={intervention['status']}",
        fill=status_color,
        font=font,
    )
    draw.text(
        (36, 165),
        f"model anchor={_fmt(intervention['model_anchor_steering'])}  "
        f"human anchor={_fmt(intervention['human_anchor_steering'])}  "
        f"delta={_fmt(intervention['correction_delta'])}",
        fill=drawing.MUTED,
        font=font,
    )
    draw.text(
        (36, 220),
        f"human included={intervention['num_exact_human_included']}  "
        f"pseudo included={intervention['num_pseudo_included']}  "
        f"excluded={intervention['num_excluded_entries']}",
        fill=drawing.MUTED,
        font=font,
    )
    return drawing.save_frame(canvas, output_path)


def generate_intervention_review_video(
    *, session_dir: Path, sidecar_path: Path, output_path: Path, keep_frames: bool = False
) -> Path:
    """Render every takeover (a title card, then each of its entries in time order) into one MP4."""
    session = load_recording_session(session_dir)
    sidecar = load_intervention_window_sidecar(sidecar_path, session_dir=session_dir)
    frame_by_id = {frame.frame_id: frame for frame in session.frames}
    entries_by_intervention: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in sidecar["entries"]:
        entries_by_intervention[entry["intervention_id"]].append(entry)
    interventions = sorted(
        sidecar["interventions"],
        key=lambda record: (record["start_timestamp_monotonic_ns"], record["intervention_id"]),
    )
    fps = recorded_fps([frame.timestamp_monotonic_ns for frame in session.frames], session.metadata)
    title_frames = max(1, round(fps * 0.25))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with frames_directory(output_path.parent / "frames" if keep_frames else None) as frames_dir:
        index = 0
        for intervention in interventions:
            title = render_title_card(intervention, frames_dir / f"frame_{index:06d}.jpg")
            for offset in range(1, title_frames):
                shutil.copyfile(title, frames_dir / f"frame_{index + offset:06d}.jpg")
            index += title_frames
            entries = sorted(
                entries_by_intervention[intervention["intervention_id"]],
                key=lambda e: (e["timestamp_monotonic_ns"], REGIONS.index(e["region"]), e["entry_id"]),
            )
            for entry in entries:
                render_intervention_review_frame(
                    frame=frame_by_id[entry["frame_id"]],
                    entry=entry,
                    intervention=intervention,
                    output_path=frames_dir / f"frame_{index:06d}.jpg",
                    sequence_index=index,
                )
                index += 1
        if index == 0:
            raise CorrectionDataError(f"{session.session_id} has no takeovers to review")
        encode_mp4(frames_dir=frames_dir, fps=fps, output_path=output_path)
    return output_path


def generate_manifest_artifacts(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    config: WindowConfig | None = None,
    write_videos: bool = True,
    keep_frames: bool = False,
) -> dict[str, object]:
    """(Re)build sidecars and review videos and update the hashes in the manifest.

    A session whose recording or sidecar changed since it was accepted goes back
    to ``pending``.
    """
    manifest = load_correction_manifest(manifest_path)
    for entry in manifest["sessions"]:
        if entry["review_status"] == "rejected":
            continue
        session_dir = Path(entry["session_path"])
        sidecar_path = Path(entry["sidecar_path"])
        written = write_intervention_window_sidecar(
            session_dir, annotations_root=sidecar_path.parent.parent, config=config
        )
        if written != sidecar_path:
            raise CorrectionDataError(
                f"Manifest expects {sidecar_path}, but the sidecar was written to {written}"
            )
        if write_videos:
            generate_intervention_review_video(
                session_dir=session_dir,
                sidecar_path=sidecar_path,
                output_path=Path(entry["review_video_path"]),
                keep_frames=keep_frames,
            )

        source_hashes = load_recording_session(session_dir).source_hashes
        sidecar_sha256 = sha256_file(sidecar_path)
        changed = entry.get("source_hashes") != source_hashes or entry.get("sidecar_sha256") != sidecar_sha256
        entry["source_hashes"] = source_hashes
        entry["sidecar_sha256"] = sidecar_sha256
        if entry["review_status"] == "accepted" and changed:
            entry.update(
                review_status="pending",
                accepted_source_hashes=None,
                accepted_sidecar_sha256=None,
                review_notes="",
            )
    save_manifest(manifest, manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser(__doc__)
    parser.add_argument(
        "sessions", nargs="*", type=Path, help="correction session folders (needed to create the manifest)"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="review manifest")
    parser.add_argument(
        "--replace-manifest",
        action="store_true",
        help="recreate the manifest from the given sessions (all pending)",
    )
    parser.add_argument("--no-videos", action="store_true", help="only build and check the sidecars")
    add_keep_frames(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.replace_manifest or not args.manifest.exists()) and not args.sessions:
        parser.error("pass the correction session folders to create the manifest")
    try:
        initialize_manifest(args.sessions, manifest_path=args.manifest, replace=args.replace_manifest)
        manifest = generate_manifest_artifacts(
            args.manifest, write_videos=not args.no_videos, keep_frames=args.keep_frames
        )
        if not args.no_videos:
            validate_manifest_artifacts(args.manifest)
    except (CorrectionDataError, VideoEncodeError) as exc:
        raise SystemExit(str(exc)) from exc
    for entry in manifest["sessions"]:
        print(
            f"{entry['session_id']}: {entry['review_status']}  "
            f"sidecar={entry['sidecar_path']}  video={entry['review_video_path']}"
        )


if __name__ == "__main__":
    main()
