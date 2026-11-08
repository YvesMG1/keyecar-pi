"""Session recorder for teleop and autonomy runs (recording schema v2).

Each session is a folder that is never modified after recording:
``metadata.json``, ``records.csv`` with one row per frame, and ``images/``.
Human, model, and executed commands are stored in separate columns, so training
can choose its label source explicitly. Derived labels (e.g. DAgger correction
targets) live in separate sidecar files keyed by ``(session_id, frame_id)``.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from .camera import Camera
from .config import CameraSettings
from .control import Condition, DriveCommand

if TYPE_CHECKING:
    from PIL import Image

    from .inference import InferenceDiagnostics
    from .ultrasonic import UltrasonicReading


@dataclass(frozen=True)
class RecordedFrame:
    frame_id: str
    image_path: Path


class InferenceDiagnosticsWriter:
    """Appends per-inference latency to ``inference.csv``."""

    FIELDNAMES = ["inference_id", "frame_id", "timestamp_monotonic_ns", "total_ms", "policy_onnx_ms"]

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self._file = (self.output_dir / "inference.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._count = 0

    @classmethod
    def in_new_folder(cls, session_root: str | Path) -> InferenceDiagnosticsWriter:
        """Create a writer in a timestamped folder (used when no session is recorded)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return cls(Path(session_root) / f"diagnostics_{timestamp}")

    def write(self, diagnostics: InferenceDiagnostics, *, frame_id: str | None = None) -> None:
        self._count += 1
        self._writer.writerow(
            {
                "inference_id": f"inference_{self._count:06d}",
                "frame_id": frame_id or "",
                "timestamp_monotonic_ns": diagnostics.timestamp_monotonic_ns,
                "total_ms": f"{diagnostics.total_ms:.4f}",
                "policy_onnx_ms": f"{diagnostics.policy_onnx_ms:.4f}",
            }
        )
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _optional(value: object | None, fmt: str = "") -> str:
    return "" if value is None else format(value, fmt)


def _command_value(command: DriveCommand | None, field: str) -> str:
    return "" if command is None else f"{getattr(command, field):.4f}"


def _ultrasonic_columns(reading: UltrasonicReading | None) -> dict[str, object]:
    if reading is None:
        return {name: "" for name in SessionRecorder.ULTRASONIC_FIELDNAMES}
    return {
        "ultrasonic_status": reading.status.value,
        "ultrasonic_raw_distance_cm": _optional(reading.raw_distance_cm, ".4f"),
        "ultrasonic_filtered_distance_cm": _optional(reading.filtered_distance_cm, ".4f"),
        "ultrasonic_raw_valid": int(reading.raw_valid),
        "ultrasonic_filtered_valid": int(reading.filtered_valid),
        "ultrasonic_emergency_stop": int(reading.emergency_stop),
        "ultrasonic_sample_started_monotonic_ns": reading.sample_started_monotonic_ns,
        "ultrasonic_raw_timestamp_monotonic_ns": reading.raw_timestamp_monotonic_ns,
        "ultrasonic_filtered_timestamp_monotonic_ns": _optional(reading.filtered_timestamp_monotonic_ns),
        "ultrasonic_pulse_duration_s": _optional(reading.pulse_duration_s, ".8f"),
    }


class SessionRecorder:
    """Saves camera frames and the matching commands into a new session folder."""

    SCHEMA_VERSION = 2
    ULTRASONIC_FIELDNAMES = [
        "ultrasonic_status",
        "ultrasonic_raw_distance_cm",
        "ultrasonic_filtered_distance_cm",
        "ultrasonic_raw_valid",
        "ultrasonic_filtered_valid",
        "ultrasonic_emergency_stop",
        "ultrasonic_sample_started_monotonic_ns",
        "ultrasonic_raw_timestamp_monotonic_ns",
        "ultrasonic_filtered_timestamp_monotonic_ns",
        "ultrasonic_pulse_duration_s",
    ]
    FIELDNAMES = [
        "schema_version",
        "session_id",
        "frame_id",
        "frame_index",
        "timestamp_wall_ns",
        "timestamp_monotonic_ns",
        "image_path",
        "camera_sensor_timestamp_ns",
        "speed_limit",
        "stop_pressed",
        "control_source",
        "mode",
        "phase",
        "layout_id",
        "condition",
        "condition_name",
        "human_throttle",
        "human_steering",
        "model_throttle",
        "model_steering",
        "executed_throttle",
        "executed_steering",
        "label_source",
        "training_eligible",
        "intervention_state",
        "intervention_id",
        # Reserved by schema v2 for obstacle-stop labels; always 0 in this version.
        "mistake_window",
        "obstacle_stop_target",
        "safety_emergency_stop",
        *ULTRASONIC_FIELDNAMES,
    ]

    def __init__(
        self,
        *,
        session_root: str | Path,
        camera_settings: CameraSettings,
        metadata: dict[str, object] | None = None,
        camera: Camera | None = None,
    ):
        """Create the session folder. Opens its own camera unless one is passed in."""
        self.camera_settings = camera_settings
        self.frame_interval_s = 1.0 / max(camera_settings.fps, 0.1)
        self._last_frame_time = 0.0
        self.frame_index = 0

        started_at = datetime.now()
        self.session_id = f"session_v2_{started_at:%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
        self.session_dir = Path(session_root) / self.session_id
        self.images_dir = self.session_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=False)

        self._file = (self.session_dir / "records.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

        session_metadata = {"layout_id": "", "phase": "", **(metadata or {})}
        session_metadata.update(
            {
                "schema_version": self.SCHEMA_VERSION,
                "session_id": self.session_id,
                "session_started_at": started_at.isoformat(timespec="seconds"),
                "raw_recording_immutable": True,
                "sidecar_join_keys": ["session_id", "frame_id"],
                "camera": camera_settings.to_metadata(),
            }
        )
        self.layout_id = str(session_metadata["layout_id"])
        self.phase = str(session_metadata["phase"])
        (self.session_dir / "metadata.json").write_text(
            json.dumps(session_metadata, indent=2), encoding="utf-8"
        )

        self._owns_camera = camera is None
        self.camera = camera or Camera(settings=camera_settings)

    def maybe_capture(
        self,
        *,
        executed_command: DriveCommand,
        speed_limit: int,
        stop_requested: bool,
        mode: str,
        condition: Condition,
        control_source: str,
        label_source: str,
        training_eligible: bool,
        human_command: DriveCommand | None = None,
        model_command: DriveCommand | None = None,
        ultrasonic_reading: UltrasonicReading | None = None,
        intervention_state: str = "none",
        intervention_id: str = "",
        image: Image.Image | None = None,
        camera_metadata: dict[str, object] | None = None,
        force: bool = False,
    ) -> RecordedFrame | None:
        """Record one frame unless the last one was less than a frame interval ago.

        Pass ``image`` to save a frame that was already captured (the autonomy
        loop does this, so the saved frame is exactly what the policy saw);
        otherwise a new frame is captured.
        """
        now = time.monotonic()
        if not force and now - self._last_frame_time < self.frame_interval_s:
            return None
        self._last_frame_time = now
        self.frame_index += 1

        frame_id = f"{self.session_id}_frame_{self.frame_index:06d}"
        filename = f"frame_{self.frame_index:06d}.jpg"
        image_path = self.images_dir / filename
        if image is None:
            camera_metadata = self.camera.capture_file(image_path)
        else:
            image.save(image_path, format="JPEG", quality=self.camera_settings.jpeg_quality)
            camera_metadata = camera_metadata or {}

        row = {
            "schema_version": self.SCHEMA_VERSION,
            "session_id": self.session_id,
            "frame_id": frame_id,
            "frame_index": self.frame_index,
            "timestamp_wall_ns": time.time_ns(),
            "timestamp_monotonic_ns": time.monotonic_ns(),
            "image_path": f"images/{filename}",
            "camera_sensor_timestamp_ns": camera_metadata.get("SensorTimestamp", ""),
            "speed_limit": speed_limit,
            "stop_pressed": int(stop_requested),
            "control_source": control_source,
            "mode": mode.lower(),
            "phase": self.phase,
            "layout_id": self.layout_id,
            "condition": int(condition),
            "condition_name": condition.name,
            "human_throttle": _command_value(human_command, "throttle"),
            "human_steering": _command_value(human_command, "steering"),
            "model_throttle": _command_value(model_command, "throttle"),
            "model_steering": _command_value(model_command, "steering"),
            "executed_throttle": f"{executed_command.throttle:.4f}",
            "executed_steering": f"{executed_command.steering:.4f}",
            "label_source": label_source,
            "training_eligible": int(training_eligible),
            "intervention_state": intervention_state,
            "intervention_id": intervention_id,
            "mistake_window": 0,
            "obstacle_stop_target": 0,
            "safety_emergency_stop": int(bool(ultrasonic_reading and ultrasonic_reading.emergency_stop)),
            **_ultrasonic_columns(ultrasonic_reading),
        }
        self._writer.writerow(row)
        self._file.flush()
        return RecordedFrame(frame_id=frame_id, image_path=image_path)

    def close(self) -> None:
        self._file.close()
        if self._owns_camera:
            self.camera.close()
