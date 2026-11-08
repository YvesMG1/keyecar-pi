"""Picamera2 wrapper shared by teleop recording and autonomous driving."""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from .config import CameraSettings, build_camera_settings

try:
    from picamera2 import Picamera2
except ImportError:  # not running on a Raspberry Pi
    Picamera2 = None


class Camera:
    """Starts Picamera2 with a camera profile and captures PIL images."""

    def __init__(self, settings: CameraSettings | None = None):
        if Picamera2 is None:
            raise RuntimeError("Picamera2 is not available in this environment.")

        self.settings = settings or build_camera_settings()
        self._picam = Picamera2()

        main = {"size": (self.settings.width, self.settings.height)}
        controls = self.settings.controls or None
        if self.settings.capture_mode == "video":
            # Keep the buffer pool shallow. With a deep queue, captures return
            # stale frames whenever the control loop is slower than the camera,
            # and the policy reacts late. Two buffers keep the backlog at ~1 frame.
            camera_config = self._picam.create_video_configuration(
                main=main, controls=controls, buffer_count=2
            )
        elif self.settings.capture_mode == "preview":
            camera_config = self._picam.create_preview_configuration(main=main, controls=controls)
        else:
            camera_config = self._picam.create_still_configuration(main=main, controls=controls)

        self._picam.configure(camera_config)
        self._picam.start()
        if controls:
            self._picam.set_controls(dict(controls))
        time.sleep(0.5)  # let exposure settle

    def capture(self) -> Image.Image:
        """Capture the current frame."""
        return self._picam.capture_image("main")

    def capture_with_metadata(self) -> tuple[Image.Image, dict[str, object]]:
        """Capture one frame and the Picamera2 metadata of the same request."""
        request = self._picam.capture_request()
        try:
            return request.make_image("main"), request.get_metadata()
        finally:
            request.release()

    def capture_file(self, path: str | Path) -> dict[str, object]:
        """Capture one frame to a JPEG file and return its metadata."""
        image, metadata = self.capture_with_metadata()
        image.save(path, format="JPEG", quality=self.settings.jpeg_quality)
        return metadata

    def close(self) -> None:
        try:
            self._picam.stop()
        finally:
            self._picam.close()
