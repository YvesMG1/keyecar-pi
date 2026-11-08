"""DAgger correction targets built from recorded takeovers, and the review manifest.

For each takeover (a contiguous block of frames with the same ``intervention_id``):

* every eligible frame the human drove becomes an ``exact_human`` target;
* frames the model drove up to ``pre_window_s`` before and ``post_window_s``
  after the takeover become pseudo targets for steering only::

      target = model_steering + alpha * delta

  where ``delta`` is the median human steering during the takeover minus the
  model's last steering before it. ``alpha`` ramps from ``pre_alpha_start`` up
  to 1 before the takeover and from 1 back to 0 after it, and also weights the
  loss of these frames;
* takeovers with ``|delta| < min_abs_correction_delta`` produce no pseudo targets.

The targets of one session are written to a JSON sidecar next to the recording,
which stays untouched. A manifest (``recordings/correction_sessions.json``)
lists the correction sessions and their review status. A human watches each
session's review video and accepts it by editing the manifest. Acceptance pins
the SHA-256 of the recording and the sidecar, and training refuses any sidecar
that changed after review.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from .files import read_json, sha256_file, write_json
from .recordings import cell_flag, cell_float, cell_text, read_records

SCHEMA_VERSION = 1
SIDECAR_KIND = "keyecar_intervention_windows"
MANIFEST_KIND = "keyecar_correction_session_manifest"

DEFAULT_RECORDINGS_ROOT = Path("recordings")
DEFAULT_ANNOTATIONS_ROOT = DEFAULT_RECORDINGS_ROOT / "annotations"
DEFAULT_REVIEW_ROOT = Path("runs/intervention_windows")
DEFAULT_MANIFEST_PATH = DEFAULT_RECORDINGS_ROOT / "correction_sessions.json"

REQUIRED_COLUMNS = {
    "schema_version",
    "session_id",
    "frame_id",
    "frame_index",
    "timestamp_monotonic_ns",
    "image_path",
    "condition",
    "condition_name",
    "human_throttle",
    "human_steering",
    "model_steering",
    "training_eligible",
    "intervention_id",
    "stop_pressed",
    "safety_emergency_stop",
}
REVIEW_STATUSES = {"pending", "accepted", "rejected"}
SPLITS = {"train", "val"}
REGIONS = ("pre", "human", "post")
SOURCE_HASH_KEYS = {"records_sha256", "metadata_sha256", "images_sha256"}


class CorrectionDataError(RuntimeError):
    """Raised when correction recordings, sidecars, or the manifest are malformed."""


@dataclass(frozen=True)
class WindowConfig:
    pre_window_s: float = 0.5
    post_window_s: float = 0.25
    min_abs_correction_delta: float = 0.15
    pre_alpha_start: float = 0.25

    def validate(self) -> None:
        if not (math.isfinite(self.pre_window_s) and self.pre_window_s > 0.0):
            raise CorrectionDataError("pre_window_s must be positive")
        if not (math.isfinite(self.post_window_s) and self.post_window_s > 0.0):
            raise CorrectionDataError("post_window_s must be positive")
        if not (math.isfinite(self.min_abs_correction_delta) and self.min_abs_correction_delta >= 0.0):
            raise CorrectionDataError("min_abs_correction_delta must be non-negative")
        if not 0.0 <= self.pre_alpha_start <= 1.0:
            raise CorrectionDataError("pre_alpha_start must be in [0, 1]")

    def to_json(self) -> dict[str, object]:
        return {
            "pre_window_s": self.pre_window_s,
            "post_window_s": self.post_window_s,
            "min_abs_correction_delta": self.min_abs_correction_delta,
            "pre_alpha_start": self.pre_alpha_start,
            "target_clamp": [-1.0, 1.0],
            "pseudo_throttle_supervision": False,
            "pseudo_steering_loss_weight": "alpha",
        }


# ---------------------------------------------------------------------------
# Reading recordings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordingFrame:
    row: dict[str, str]
    session_id: str
    frame_id: str
    frame_index: int
    timestamp_monotonic_ns: int
    image_path: Path
    condition: int
    condition_name: str
    intervention_id: str
    human_throttle: float | None
    human_steering: float | None
    model_steering: float | None
    training_eligible: bool
    stop_pressed: bool
    safety_emergency_stop: bool
    ultrasonic_emergency_stop: bool

    @property
    def unsafe(self) -> bool:
        return self.stop_pressed or self.safety_emergency_stop or self.ultrasonic_emergency_stop


@dataclass(frozen=True)
class RecordingSession:
    session_dir: Path
    session_id: str
    metadata: dict[str, object]
    frames: list[RecordingFrame]
    source_hashes: dict[str, str]  # records_sha256, metadata_sha256, images_sha256


def load_recording_session(session_dir: Path | str, *, require_images: bool = True) -> RecordingSession:
    """Load one schema-v2 recording and check that it is consistent."""
    session_dir = Path(session_dir)
    metadata_path = session_dir / "metadata.json"
    records_path = session_dir / "records.csv"
    metadata = _read_json_object(metadata_path)
    if _int(metadata.get("schema_version"), metadata_path, "schema_version") < 2:
        raise CorrectionDataError(f"{metadata_path} must describe a schema-v2 recording")
    session_id = _text(metadata.get("session_id")) or session_dir.name

    try:
        rows = read_records(session_dir)
    except FileNotFoundError as exc:
        raise CorrectionDataError(str(exc)) from exc
    if not rows:
        raise CorrectionDataError(f"{records_path} has no frames")
    missing = sorted(REQUIRED_COLUMNS - set(rows[0]))
    if missing:
        raise CorrectionDataError(f"{records_path} is missing columns: {', '.join(missing)}")
    try:
        frames = [_parse_frame(row, session_dir, session_id) for row in rows]
    except ValueError as exc:
        raise CorrectionDataError(f"{records_path}: {exc}") from exc

    frame_ids = [frame.frame_id for frame in frames]
    if len(frame_ids) != len(set(frame_ids)):
        raise CorrectionDataError(f"{records_path} repeats frame_id values")
    for name in ("frame_index", "timestamp_monotonic_ns"):
        values = [getattr(frame, name) for frame in frames]
        if any(b <= a for a, b in zip(values, values[1:], strict=False)):
            raise CorrectionDataError(f"{records_path} {name} values must be strictly increasing")
    _check_intervention_blocks(frames, records_path)
    if require_images:
        missing_images = [str(frame.image_path) for frame in frames if not frame.image_path.is_file()]
        if missing_images:
            raise CorrectionDataError(
                f"{records_path} references {len(missing_images)} missing images: {', '.join(missing_images[:3])}"
            )

    return RecordingSession(
        session_dir=session_dir,
        session_id=session_id,
        metadata=metadata,
        frames=frames,
        source_hashes={
            "records_sha256": sha256_file(records_path),
            "metadata_sha256": sha256_file(metadata_path),
            "images_sha256": _hash_images(frames, session_dir),
        },
    )


def _parse_frame(row: dict[str, str], session_dir: Path, expected_session_id: str) -> RecordingFrame:
    records_path = session_dir / "records.csv"
    if _int(row.get("schema_version"), records_path, "schema_version") < 2:
        raise CorrectionDataError(f"{records_path} rows must use schema_version 2")
    session_id = _required_text(row, "session_id", records_path)
    if session_id != expected_session_id:
        raise CorrectionDataError(
            f"{records_path} session_id={session_id!r} does not match {expected_session_id!r}"
        )
    image_path = (session_dir / _required_text(row, "image_path", records_path)).resolve()
    if image_path.parent != (session_dir / "images").resolve():
        raise CorrectionDataError(f"Frame image must be inside {session_dir / 'images'}: {image_path}")
    condition = _int(row.get("condition"), records_path, "condition")
    return RecordingFrame(
        row=dict(row),
        session_id=session_id,
        frame_id=_required_text(row, "frame_id", records_path),
        frame_index=_int(row.get("frame_index"), records_path, "frame_index"),
        timestamp_monotonic_ns=_int(
            row.get("timestamp_monotonic_ns"), records_path, "timestamp_monotonic_ns"
        ),
        image_path=image_path,
        condition=condition,
        condition_name=cell_text(row, "condition_name") or str(condition),
        intervention_id=cell_text(row, "intervention_id"),
        human_throttle=cell_float(row, "human_throttle"),
        human_steering=cell_float(row, "human_steering"),
        model_steering=cell_float(row, "model_steering"),
        training_eligible=cell_flag(row, "training_eligible"),
        stop_pressed=cell_flag(row, "stop_pressed"),
        safety_emergency_stop=cell_flag(row, "safety_emergency_stop"),
        ultrasonic_emergency_stop=cell_flag(row, "ultrasonic_emergency_stop"),
    )


def _check_intervention_blocks(frames: list[RecordingFrame], records_path: Path) -> None:
    """Each intervention_id must cover one contiguous block of frames."""
    positions: dict[str, list[int]] = defaultdict(list)
    for position, frame in enumerate(frames):
        if frame.intervention_id:
            positions[frame.intervention_id].append(position)
    for intervention_id, indexes in positions.items():
        if indexes[-1] - indexes[0] + 1 != len(indexes):
            raise CorrectionDataError(
                f"{records_path} intervention_id={intervention_id!r} appears in multiple blocks"
            )


def _hash_images(frames: Iterable[RecordingFrame], session_dir: Path) -> str:
    """Hash the names and contents of all images referenced by records.csv, in order."""
    digest = hashlib.sha256()
    digest.update(b"keyecar-referenced-images-v1\0")
    for frame in frames:
        digest.update(frame.image_path.relative_to(session_dir.resolve()).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(frame.image_path)))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Building correction targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Anchors:
    """Per-takeover values shared by all of its entries."""

    model_frame_id: str | None
    model_steering: float | None
    human_steering: float | None

    @property
    def delta(self) -> float | None:
        if self.model_steering is None or self.human_steering is None:
            return None
        return self.human_steering - self.model_steering


def _entry(
    frame: RecordingFrame, intervention_id: str, region: str, anchors: _Anchors, alpha: float
) -> dict[str, object]:
    """Fields every sidecar entry has; the caller adds targets and inclusion."""
    return {
        "entry_id": f"{intervention_id}:{region}:{frame.frame_id}",
        "session_id": frame.session_id,
        "frame_id": frame.frame_id,
        "frame_index": frame.frame_index,
        "timestamp_monotonic_ns": frame.timestamp_monotonic_ns,
        "intervention_id": intervention_id,
        "region": region,
        "condition": frame.condition,
        "condition_name": frame.condition_name,
        "recorded_model_steering": _round(frame.model_steering),
        "exact_human_throttle": _round(frame.human_throttle),
        "exact_human_steering": _round(frame.human_steering),
        "model_anchor_frame_id": anchors.model_frame_id,
        "model_anchor_steering": _round(anchors.model_steering),
        "human_anchor_steering": _round(anchors.human_steering),
        "correction_delta": _round(anchors.delta),
        "alpha": _round(alpha),
    }


def _set_targets(
    entry: dict[str, object], *, throttle: float | None, steering: float | None, weight: float
) -> None:
    entry["targets"] = {"throttle": _round(throttle), "steering": _round(steering)}
    entry["supervision"] = {"throttle": throttle is not None, "steering": steering is not None}
    entry["loss_weights"] = {
        "throttle": 1.0 if throttle is not None else 0.0,
        "steering": weight if steering is not None else 0.0,
    }
    entry["included"] = steering is not None
    entry["exclusion_reason"] = ""


def _exclude(entry: dict[str, object], reason: str) -> None:
    _set_targets(entry, throttle=None, steering=None, weight=0.0)
    entry["exclusion_reason"] = reason


def _human_entry(frame: RecordingFrame, intervention_id: str, anchors: _Anchors) -> dict[str, object]:
    entry = _entry(frame, intervention_id, "human", anchors, alpha=1.0)
    if not frame.training_eligible:
        _exclude(entry, "human_not_training_eligible")
    elif frame.unsafe:
        _exclude(entry, "human_stop_or_safety_frame")
    elif frame.human_throttle is None or frame.human_steering is None:
        _exclude(entry, "missing_human_command")
    else:
        _set_targets(entry, throttle=frame.human_throttle, steering=frame.human_steering, weight=1.0)
    return entry


def _pseudo_entry(
    frame: RecordingFrame, intervention_id: str, region: str, anchors: _Anchors, alpha: float, status: str
) -> dict[str, object]:
    alpha = min(1.0, max(0.0, alpha))
    entry = _entry(frame, intervention_id, region, anchors, alpha=alpha)
    if status != "usable":
        _exclude(entry, status)
    elif frame.model_steering is None:
        _exclude(entry, "missing_model_steering")
    elif frame.unsafe:
        _exclude(entry, "pseudo_stop_or_safety_frame")
    else:
        target = min(1.0, max(-1.0, frame.model_steering + alpha * anchors.delta))
        _set_targets(entry, throttle=None, steering=target, weight=_round(alpha))
    return entry


def _intervention_entries(
    frames: list[RecordingFrame], intervention_id: str, config: WindowConfig
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Return the sidecar entries of one takeover and its summary record."""
    human_frames = [frame for frame in frames if frame.intervention_id == intervention_id]
    start_ns = human_frames[0].timestamp_monotonic_ns
    end_ns = human_frames[-1].timestamp_monotonic_ns
    pre_start_ns = start_ns - round(config.pre_window_s * 1e9)
    post_end_ns = end_ns + round(config.post_window_s * 1e9)
    pre_frames = [frame for frame in frames if pre_start_ns <= frame.timestamp_monotonic_ns < start_ns]
    post_frames = [frame for frame in frames if end_ns < frame.timestamp_monotonic_ns <= post_end_ns]

    model_anchor = next((frame for frame in reversed(pre_frames) if frame.model_steering is not None), None)
    eligible_human = [
        frame
        for frame in human_frames
        if frame.training_eligible
        and not frame.unsafe
        and frame.human_throttle is not None
        and frame.human_steering is not None
    ]
    anchors = _Anchors(
        model_frame_id=model_anchor.frame_id if model_anchor else None,
        model_steering=model_anchor.model_steering if model_anchor else None,
        human_steering=float(median(f.human_steering for f in eligible_human)) if eligible_human else None,
    )
    if anchors.human_steering is None:
        status = "no_eligible_human_anchor"
    elif anchors.model_steering is None:
        status = "missing_model_anchor"
    elif abs(anchors.delta) < config.min_abs_correction_delta:
        status = "correction_below_threshold"
    else:
        status = "usable"

    entries = []
    for frame in pre_frames:
        progress = (frame.timestamp_monotonic_ns - pre_start_ns) / max(1, start_ns - pre_start_ns)
        alpha = config.pre_alpha_start + (1.0 - config.pre_alpha_start) * progress
        entries.append(_pseudo_entry(frame, intervention_id, "pre", anchors, alpha, status))
    entries.extend(_human_entry(frame, intervention_id, anchors) for frame in human_frames)
    for frame in post_frames:
        alpha = 1.0 - (frame.timestamp_monotonic_ns - end_ns) / max(1, post_end_ns - end_ns)
        entries.append(_pseudo_entry(frame, intervention_id, "post", anchors, alpha, status))

    record = {
        "intervention_id": intervention_id,
        "status": status,
        "start_frame_id": human_frames[0].frame_id,
        "start_frame_index": human_frames[0].frame_index,
        "start_timestamp_monotonic_ns": start_ns,
        "end_frame_id": human_frames[-1].frame_id,
        "end_frame_index": human_frames[-1].frame_index,
        "end_timestamp_monotonic_ns": end_ns,
        "model_anchor_frame_id": anchors.model_frame_id,
        "model_anchor_steering": _round(anchors.model_steering),
        "human_anchor_steering": _round(anchors.human_steering),
        "correction_delta": _round(anchors.delta),
        "num_human_frames": len(human_frames),
        "num_eligible_human_frames": len(eligible_human),
        "num_pre_candidates": len(pre_frames),
        "num_post_candidates": len(post_frames),
    }
    return entries, record


def _resolve_overlaps(entries: list[dict[str, object]]) -> None:
    """Make sure each frame has at most one included target.

    Human labels win over pseudo targets. Where pseudo windows of two takeovers
    overlap, the higher alpha wins, unless they disagree on the steering
    direction, in which case the frame is dropped.
    """
    human_frame_ids = {entry["frame_id"] for entry in entries if entry["region"] == "human"}
    pseudo_by_frame: dict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        if entry["region"] != "human" and entry["included"]:
            pseudo_by_frame[str(entry["frame_id"])].append(entry)

    for frame_id, candidates in pseudo_by_frame.items():
        if frame_id in human_frame_ids:
            for entry in candidates:
                _exclude(entry, "human_label_precedence")
        elif len(candidates) > 1:
            if len({float(entry["correction_delta"]) > 0.0 for entry in candidates}) > 1:
                for entry in candidates:
                    _exclude(entry, "conflicting_correction_direction")
                continue
            ranked = sorted(
                candidates, key=lambda e: (-float(e["alpha"]), e["intervention_id"], e["entry_id"])
            )
            for entry in ranked[1:]:
                _exclude(entry, "lower_alpha_overlap")


def build_intervention_window_payload(
    session_dir: Path | str, *, config: WindowConfig | None = None, require_images: bool = True
) -> dict[str, object]:
    """Build the (deterministic) correction sidecar of one session."""
    config = config or WindowConfig()
    config.validate()
    session = load_recording_session(session_dir, require_images=require_images)
    frames = session.frames

    entries: list[dict[str, object]] = []
    interventions: list[dict[str, object]] = []
    for intervention_id in dict.fromkeys(frame.intervention_id for frame in frames if frame.intervention_id):
        intervention_entries, record = _intervention_entries(frames, intervention_id, config)
        entries.extend(intervention_entries)
        interventions.append(record)

    _resolve_overlaps(entries)
    entries.sort(
        key=lambda e: (
            e["timestamp_monotonic_ns"],
            e["intervention_id"],
            REGIONS.index(e["region"]),
            e["entry_id"],
        )
    )
    for record in interventions:
        own = [entry for entry in entries if entry["intervention_id"] == record["intervention_id"]]
        included = [entry for entry in own if entry["included"]]
        record["num_exact_human_included"] = sum(entry["region"] == "human" for entry in included)
        record["num_pseudo_included"] = sum(entry["region"] != "human" for entry in included)
        record["num_excluded_entries"] = len(own) - len(included)
        record["exclusion_counts"] = _exclusion_counts(own)

    included = [entry for entry in entries if entry["included"]]
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": SIDECAR_KIND,
        "session_id": session.session_id,
        "sidecar_join_keys": ["session_id", "frame_id"],
        "source": {
            "session_dir_name": session.session_dir.name,
            "records_path": "records.csv",
            "metadata_path": "metadata.json",
            **session.source_hashes,
            "num_frames": len(frames),
            "frame_index_min": frames[0].frame_index,
            "frame_index_max": frames[-1].frame_index,
        },
        "generation": config.to_json(),
        "summary": {
            "num_interventions": len(interventions),
            "num_usable_interventions": sum(record["status"] == "usable" for record in interventions),
            "num_exact_human_entries": sum(entry["region"] == "human" for entry in included),
            "num_pseudo_entries": sum(entry["region"] != "human" for entry in included),
            "num_included_entries": len(included),
            "num_excluded_entries": len(entries) - len(included),
            "exclusion_counts": _exclusion_counts(entries),
        },
        "interventions": interventions,
        "entries": entries,
    }
    validate_sidecar_payload(payload)
    return payload


def _exclusion_counts(entries: list[dict[str, object]]) -> dict[str, int]:
    counts = Counter(str(entry["exclusion_reason"]) for entry in entries if not entry["included"])
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Sidecars
# ---------------------------------------------------------------------------


def sidecar_path_for(session_id: str, *, annotations_root: Path | str = DEFAULT_ANNOTATIONS_ROOT) -> Path:
    return Path(annotations_root) / session_id / "intervention_windows.json"


def review_video_path_for(session_id: str, *, review_root: Path | str = DEFAULT_REVIEW_ROOT) -> Path:
    return Path(review_root) / session_id / "review.mp4"


def write_intervention_window_sidecar(
    session_dir: Path | str,
    *,
    annotations_root: Path | str = DEFAULT_ANNOTATIONS_ROOT,
    config: WindowConfig | None = None,
) -> Path:
    payload = build_intervention_window_payload(session_dir, config=config)
    path = sidecar_path_for(str(payload["session_id"]), annotations_root=annotations_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_sorted_json(path, payload)
    load_intervention_window_sidecar(path, session_dir=session_dir)  # check what was written
    return path


def load_intervention_window_sidecar(
    path: Path | str, *, session_dir: Path | str | None = None
) -> dict[str, object]:
    """Load and validate a sidecar; with ``session_dir``, also check it is not stale."""
    payload = _read_json_object(Path(path))
    validate_sidecar_payload(payload)
    if session_dir is not None:
        session = load_recording_session(session_dir, require_images=False)
        if session.session_id != payload["session_id"]:
            raise CorrectionDataError(
                f"Sidecar {path} belongs to {payload['session_id']}, not {session.session_id}"
            )
        source = payload["source"]
        for key in ("records_sha256", "metadata_sha256"):
            if source.get(key) != session.source_hashes[key]:
                raise CorrectionDataError(f"Sidecar {path} is stale: {key} changed")
        if source.get("images_sha256") != session.source_hashes["images_sha256"]:
            raise CorrectionDataError(f"Sidecar {path} is stale: images changed")
    return payload


def validate_sidecar_payload(payload: dict[str, object]) -> None:
    """Check the sidecar structure and that every frame has at most one included target."""
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != SIDECAR_KIND:
        raise CorrectionDataError(f"Not a schema-{SCHEMA_VERSION} {SIDECAR_KIND!r} sidecar")
    if not _text(payload.get("session_id")):
        raise CorrectionDataError("Sidecar is missing session_id")
    if payload.get("sidecar_join_keys") != ["session_id", "frame_id"]:
        raise CorrectionDataError("Sidecar must be keyed by session_id and frame_id")
    interventions, entries = payload.get("interventions"), payload.get("entries")
    if (
        not isinstance(interventions, list)
        or not isinstance(entries, list)
        or not isinstance(payload.get("source"), dict)
    ):
        raise CorrectionDataError("Sidecar needs a source object and intervention and entry lists")

    known_interventions = {_text(record.get("intervention_id")) for record in interventions}
    if "" in known_interventions:
        raise CorrectionDataError("Sidecar contains an intervention without an ID")
    entry_ids: set[str] = set()
    included_frames: set[str] = set()
    for entry in entries:
        entry_id = _text(entry.get("entry_id"))
        if not entry_id or entry_id in entry_ids:
            raise CorrectionDataError(f"Sidecar repeats or omits entry_id={entry_id!r}")
        entry_ids.add(entry_id)
        if not _text(entry.get("frame_id")) or _text(entry.get("intervention_id")) not in known_interventions:
            raise CorrectionDataError(f"Invalid frame or intervention in entry {entry_id}")
        if entry.get("region") not in REGIONS:
            raise CorrectionDataError(f"Invalid region in entry {entry_id}: {entry.get('region')}")
        included = entry.get("included")
        if not isinstance(included, bool) or included == bool(_text(entry.get("exclusion_reason"))):
            raise CorrectionDataError(f"Entry {entry_id} needs included=true or an exclusion_reason")
        if not included:
            continue
        expected = {"steering": True, "throttle": entry["region"] == "human"}
        if entry.get("supervision") != expected:
            raise CorrectionDataError(
                f"Entry {entry_id}: human entries supervise both outputs, pseudo only steering"
            )
        if entry["frame_id"] in included_frames:
            raise CorrectionDataError(f"Sidecar includes frame {entry['frame_id']} more than once")
        included_frames.add(entry["frame_id"])


# ---------------------------------------------------------------------------
# Review manifest
# ---------------------------------------------------------------------------


def build_manifest_payload(
    session_dirs: Iterable[Path | str],
    *,
    annotations_root: Path | str = DEFAULT_ANNOTATIONS_ROOT,
    review_root: Path | str = DEFAULT_REVIEW_ROOT,
) -> dict[str, object]:
    """Return a new manifest with every session pending review."""
    sessions = []
    for session_dir in session_dirs:
        session = load_recording_session(session_dir)
        sessions.append(
            {
                "session_id": session.session_id,
                "session_path": str(session.session_dir),
                "split": "train",
                "review_status": "pending",
                "review_notes": "",
                "source_hashes": session.source_hashes,
                "sidecar_path": str(sidecar_path_for(session.session_id, annotations_root=annotations_root)),
                "sidecar_sha256": None,
                "review_video_path": str(review_video_path_for(session.session_id, review_root=review_root)),
                "accepted_source_hashes": None,
                "accepted_sidecar_sha256": None,
            }
        )
    sessions.sort(key=lambda entry: entry["session_id"])
    return {"schema_version": SCHEMA_VERSION, "kind": MANIFEST_KIND, "sessions": sessions}


def initialize_manifest(
    session_dirs: Iterable[Path | str],
    *,
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    replace: bool = False,
) -> Path:
    """Create the manifest unless it already exists (or ``replace`` is set)."""
    manifest_path = Path(manifest_path)
    if manifest_path.exists() and not replace:
        load_correction_manifest(manifest_path)
        return manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    save_manifest(build_manifest_payload(session_dirs), manifest_path)
    return manifest_path


def save_manifest(manifest: dict[str, object], path: Path | str = DEFAULT_MANIFEST_PATH) -> None:
    _write_sorted_json(Path(path), manifest)


def load_correction_manifest(path: Path | str = DEFAULT_MANIFEST_PATH) -> dict[str, object]:
    """Load the manifest and check its structure (not yet the files it points to)."""
    path = Path(path)
    payload = _read_json_object(path)
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("kind") != MANIFEST_KIND:
        raise CorrectionDataError(f"{path} is not a schema-{SCHEMA_VERSION} {MANIFEST_KIND!r} manifest")
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        raise CorrectionDataError(f"{path}: sessions must be a list")
    seen: set[str] = set()
    for entry in sessions:
        session_id = _required_text(entry, "session_id", path)
        if session_id in seen:
            raise CorrectionDataError(f"{path} lists {session_id} twice")
        seen.add(session_id)
        for key in ("session_path", "sidecar_path", "review_video_path"):
            _required_text(entry, key, path)
        if _required_text(entry, "split", path) not in SPLITS:
            raise CorrectionDataError(f"{path}: invalid split for {session_id}")
        status = _required_text(entry, "review_status", path)
        if status not in REVIEW_STATUSES:
            raise CorrectionDataError(f"{path}: invalid review_status for {session_id}")
        if status == "rejected" and not _text(entry.get("review_notes")):
            raise CorrectionDataError(f"{path}: rejected session {session_id} needs review_notes")
        if status == "accepted" and not (
            _is_source_hashes(entry.get("accepted_source_hashes"))
            and _text(entry.get("accepted_sidecar_sha256"))
        ):
            raise CorrectionDataError(f"{path}: accepted session {session_id} needs the accepted hashes")
    return payload


def validate_manifest_artifacts(path: Path | str = DEFAULT_MANIFEST_PATH) -> dict[str, object]:
    """Check that every non-rejected session's recording, sidecar, and video match the manifest."""
    payload = load_correction_manifest(path)
    for entry in payload["sessions"]:
        if entry["review_status"] == "rejected":
            continue
        session_id = entry["session_id"]
        session_dir = Path(entry["session_path"])
        sidecar_path = Path(entry["sidecar_path"])
        session = load_recording_session(session_dir)
        if session.session_id != session_id:
            raise CorrectionDataError(f"Manifest path for {session_id} points to {session.session_id}")
        if entry.get("source_hashes") != session.source_hashes:
            raise CorrectionDataError(f"Recording of {session_id} changed since the sidecar was generated")
        if not sidecar_path.is_file():
            raise CorrectionDataError(f"Missing sidecar for {session_id}: {sidecar_path}")
        load_intervention_window_sidecar(sidecar_path, session_dir=session_dir)
        sidecar_sha256 = sha256_file(sidecar_path)
        if entry.get("sidecar_sha256") != sidecar_sha256:
            raise CorrectionDataError(f"Sidecar of {session_id} changed since the manifest was updated")
        if not Path(entry["review_video_path"]).is_file():
            raise CorrectionDataError(f"Missing review video for {session_id}: {entry['review_video_path']}")
        if entry["review_status"] == "accepted" and (
            entry.get("accepted_source_hashes") != session.source_hashes
            or entry.get("accepted_sidecar_sha256") != sidecar_sha256
        ):
            raise CorrectionDataError(f"{session_id} changed after it was accepted; review it again")
    return payload


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _read_json_object(path: Path) -> dict[str, object]:
    try:
        data = read_json(path)
    except FileNotFoundError as exc:
        raise CorrectionDataError(f"Missing JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CorrectionDataError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CorrectionDataError(f"{path} must contain a JSON object")
    return data


def _write_sorted_json(path: Path, payload: dict[str, object]) -> None:
    """Sidecar and manifest format: sorted keys and no NaN, so the bytes (and hashes) are stable."""
    write_json(path, payload, sort_keys=True, allow_nan=False)


def _is_source_hashes(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == SOURCE_HASH_KEYS
        and all(
            isinstance(v, str) and len(v) == 64 and set(v) <= set("0123456789abcdef") for v in value.values()
        )
    )


def _text(value: object) -> str:
    return "" if value is None else str(value).strip()


def _required_text(entry: dict[str, object], key: str, path: Path) -> str:
    value = _text(entry.get(key))
    if not value:
        raise CorrectionDataError(f"{path}: an entry is missing {key}")
    return value


def _int(value: object, path: Path, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CorrectionDataError(f"{path}: {key} must be an integer, got {value!r}") from exc


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 8)
