"""Hardware, controller, and camera configuration for the KS0223 car.

Everything that depends on the physical build lives here: GPIO pin maps, motor
polarity, ultrasonic thresholds, the pygame controller layout, and the camera
profile. The entry points (``keyecar-teleop``, ``keyecar-drive``) expose the
per-run settings as CLI flags and fall back to these defaults.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


def _frame_duration_limits_for_fps(fps: float) -> tuple[int, int]:
    """Return Picamera2 frame-duration bounds (microseconds) for a fixed frame rate."""
    frame_us = round(1_000_000 / max(fps, 0.1))
    return frame_us, frame_us


@dataclass
class MotorPins:
    """BCM pin mapping of the KS0223 motor driver board (two motors per side)."""

    left_m2_in1: int = 20
    left_m2_in2: int = 21
    left_m2_pwm: int = 0
    left_m1_in1: int = 22
    left_m1_in2: int = 23
    left_m1_pwm: int = 1
    right_m3_in1: int = 24
    right_m3_in2: int = 25
    right_m3_pwm: int = 12
    right_m4_in1: int = 26
    right_m4_in2: int = 27
    right_m4_pwm: int = 13

    @property
    def direction_pins(self) -> list[int]:
        return [
            self.left_m2_in1,
            self.left_m2_in2,
            self.left_m1_in1,
            self.left_m1_in2,
            self.right_m3_in1,
            self.right_m3_in2,
            self.right_m4_in1,
            self.right_m4_in2,
        ]

    @property
    def pwm_pins(self) -> list[int]:
        return [self.left_m2_pwm, self.left_m1_pwm, self.right_m3_pwm, self.right_m4_pwm]


@dataclass
class MotorSettings:
    """Motor polarity and speed limits.

    ``reversed_motors`` flips a motor whose forward/backward wiring is swapped.
    Speeds are PWM duty cycles in percent; L1/R1 change the limit by
    ``speed_step`` while driving.
    """

    pins: MotorPins = field(default_factory=MotorPins)
    reversed_motors: dict[str, bool] = field(
        default_factory=lambda: {"L1": True, "L2": False, "R1": False, "R2": True}
    )
    pwm_frequency_hz: int = 100
    max_duty: int = 100
    default_speed_limit: int = 30
    speed_step: int = 5


@dataclass
class UltrasonicPins:
    """BCM pins of the front HC-SR04 sensor."""

    trigger: int = 14
    echo: int = 4


@dataclass
class UltrasonicSettings:
    """HC-SR04 timing, filtering, and emergency-stop threshold."""

    pins: UltrasonicPins = field(default_factory=UltrasonicPins)
    enabled: bool = True
    trigger_pulse_s: float = 0.00001
    settle_time_s: float = 0.00006
    echo_timeout_s: float = 0.03
    min_valid_distance_cm: float = 2.0
    max_valid_distance_cm: float = 400.0
    filter_alpha: float = 0.35  # EMA weight of a new valid reading
    emergency_stop_distance_cm: float = 10.0
    bench_interval_s: float = 0.1


@dataclass
class ControllerMapping:
    """PlayStation controller axes and buttons as reported by pygame on the Pi."""

    left_stick_x_axis: int = 0
    left_stick_y_axis: int = 1
    # Triangle / Square / Circle latch the route condition. Cross only stops the
    # car; stopping is not a condition the policy learns.
    forward_button: int = 2
    left_button: int = 3
    right_button: int = 1
    stop_button: int = 0
    l1_button: int = 4
    r1_button: int = 5
    quit_button: int = 9


@dataclass
class TeleopSettings:
    """Control-loop timing and stick tuning."""

    deadzone: float = 0.12
    status_interval_s: float = 0.2
    loop_delay_s: float = 0.02
    session_root: str = "recordings"


@dataclass
class CameraSettings:
    """Resolved camera configuration; build it with :func:`build_camera_settings`."""

    width: int
    height: int
    fps: float
    jpeg_quality: int
    profile: str
    capture_mode: str = "still"
    controls: dict[str, object] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, object]:
        """Return the settings as JSON-friendly session metadata."""
        data: dict[str, object] = {
            "profile": self.profile,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "jpeg_quality": self.jpeg_quality,
            "capture_mode": self.capture_mode,
        }
        if self.controls:
            data["controls"] = copy.deepcopy(self.controls)
        return data


# The "motion" profile uses the video pipeline with a fixed frame duration, which
# gives sharper frames while the car turns than still captures do.
CAMERA_PROFILES = {
    "motion": CameraSettings(
        width=640,
        height=480,
        fps=30.0,
        jpeg_quality=85,
        profile="motion",
        capture_mode="video",
        controls={"FrameDurationLimits": _frame_duration_limits_for_fps(30.0)},
    ),
}
DEFAULT_CAMERA_PROFILE = "motion"


def build_camera_settings(
    profile: str = DEFAULT_CAMERA_PROFILE,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    jpeg_quality: int | None = None,
) -> CameraSettings:
    """Return a named camera profile with optional overrides."""
    if profile not in CAMERA_PROFILES:
        raise ValueError(f"Unknown camera profile: {profile}")

    settings = copy.deepcopy(CAMERA_PROFILES[profile])
    if width is not None:
        settings.width = width
    if height is not None:
        settings.height = height
    if fps is not None:
        settings.fps = fps
        if "FrameDurationLimits" in settings.controls:
            settings.controls["FrameDurationLimits"] = _frame_duration_limits_for_fps(fps)
    if jpeg_quality is not None:
        settings.jpeg_quality = jpeg_quality
    return settings


@dataclass
class RuntimeSettings:
    """Everything the control loops need, with the defaults above."""

    motor: MotorSettings = field(default_factory=MotorSettings)
    camera: CameraSettings = field(default_factory=build_camera_settings)
    controller: ControllerMapping = field(default_factory=ControllerMapping)
    teleop: TeleopSettings = field(default_factory=TeleopSettings)
    ultrasonic: UltrasonicSettings = field(default_factory=UltrasonicSettings)
