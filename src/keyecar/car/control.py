"""Drive commands, skid-steer mixing, and command arbitration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Condition(IntEnum):
    """Route command the operator gives the policy.

    The integer values are written to every recording and used as embedding
    indices by the model, so they must never change.
    """

    FORWARD = 0
    LEFT = 1
    RIGHT = 2


def mirror_condition(condition: Condition) -> Condition:
    """Return the condition for a horizontally flipped image."""
    if condition is Condition.LEFT:
        return Condition.RIGHT
    if condition is Condition.RIGHT:
        return Condition.LEFT
    return condition


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def apply_deadzone(value: float, deadzone: float) -> float:
    """Treat small stick deflections around the center as zero."""
    return 0.0 if abs(value) < deadzone else value


@dataclass(frozen=True)
class DriveCommand:
    """Throttle and steering, each clamped to [-1, 1]."""

    throttle: float
    steering: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "throttle", clamp(float(self.throttle), -1.0, 1.0))
        object.__setattr__(self, "steering", clamp(float(self.steering), -1.0, 1.0))

    def is_zero(self) -> bool:
        return self.throttle == 0.0 and self.steering == 0.0


STOP_COMMAND = DriveCommand(0.0, 0.0)


@dataclass(frozen=True)
class WheelCommand:
    """Left and right wheel speeds, each clamped to [-1, 1]."""

    left: float
    right: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", clamp(float(self.left), -1.0, 1.0))
        object.__setattr__(self, "right", clamp(float(self.right), -1.0, 1.0))


@dataclass(frozen=True)
class ArbitrationResult:
    """The command to execute and which input it came from."""

    command: DriveCommand
    source: str  # "stop", "safety_stop", "manual", "autonomy", or "idle"
    stop_requested: bool = False


def mix_drive(command: DriveCommand) -> WheelCommand:
    """Skid-steer mixing: steering speeds up one side and slows the other."""
    return WheelCommand(
        left=command.throttle + command.steering,
        right=command.throttle - command.steering,
    )


def _reverse_only(command: DriveCommand) -> DriveCommand | None:
    """Limit a manual command so that neither wheel turns forward.

    Inside the ultrasonic emergency zone the operator must still be able to back
    out. On a skid-steer chassis, reversing with a lot of steering would drive
    one wheel forward, so steering is limited to the magnitude of the throttle.
    """
    if command.throttle >= 0.0:
        return None
    return DriveCommand(command.throttle, clamp(command.steering, command.throttle, -command.throttle))


def arbitrate_command(
    *,
    manual: DriveCommand | None = None,
    autonomy: DriveCommand | None = None,
    stop_requested: bool = False,
    safety_stop_requested: bool = False,
) -> ArbitrationResult:
    """Pick the command to execute.

    Priority: stop button > ultrasonic emergency stop (reversing still allowed)
    > manual stick input > policy output.
    """
    if stop_requested:
        return ArbitrationResult(STOP_COMMAND, source="stop", stop_requested=True)
    if safety_stop_requested:
        recovery = _reverse_only(manual) if manual is not None else None
        if recovery is not None:
            return ArbitrationResult(recovery, source="manual")
        return ArbitrationResult(STOP_COMMAND, source="safety_stop", stop_requested=True)
    if manual is not None:
        return ArbitrationResult(manual, source="manual")
    if autonomy is not None:
        return ArbitrationResult(autonomy, source="autonomy")
    return ArbitrationResult(STOP_COMMAND, source="idle", stop_requested=True)
