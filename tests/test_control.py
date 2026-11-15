"""Command arbitration and skid-steer mixing. No hardware needed."""

from __future__ import annotations

import itertools

import pytest

from keyecar.car.control import (
    STOP_COMMAND,
    Condition,
    DriveCommand,
    WheelCommand,
    apply_deadzone,
    arbitrate_command,
    mirror_condition,
    mix_drive,
)

MANUAL = DriveCommand(0.4, -0.2)
AUTONOMY = DriveCommand(0.6, 0.3)


def test_condition_values_are_stable():
    # Written to every recording and used as embedding indices by the model.
    assert [(c.name, c.value) for c in Condition] == [("FORWARD", 0), ("LEFT", 1), ("RIGHT", 2)]


def test_mirror_condition_swaps_left_and_right():
    assert mirror_condition(Condition.LEFT) is Condition.RIGHT
    assert mirror_condition(Condition.RIGHT) is Condition.LEFT
    assert mirror_condition(Condition.FORWARD) is Condition.FORWARD


def test_deadzone_zeroes_small_deflections_only():
    assert apply_deadzone(0.05, 0.1) == 0.0
    assert apply_deadzone(-0.05, 0.1) == 0.0
    assert apply_deadzone(0.1, 0.1) == 0.1
    assert apply_deadzone(-0.5, 0.1) == -0.5


def test_commands_are_clamped_to_unit_range():
    assert DriveCommand(2.0, -3.0) == DriveCommand(1.0, -1.0)
    assert WheelCommand(-1.5, 1.5) == WheelCommand(-1.0, 1.0)


@pytest.mark.parametrize(
    ("command", "wheels"),
    [
        (DriveCommand(0.5, 0.0), WheelCommand(0.5, 0.5)),  # straight
        (DriveCommand(0.5, 0.25), WheelCommand(0.75, 0.25)),  # positive steering speeds up the left side
        (DriveCommand(0.0, 0.5), WheelCommand(0.5, -0.5)),  # turn on the spot
        (DriveCommand(-0.5, 0.0), WheelCommand(-0.5, -0.5)),  # reverse
        (DriveCommand(1.0, 1.0), WheelCommand(1.0, 0.0)),  # saturates instead of exceeding full speed
    ],
)
def test_mix_drive(command, wheels):
    assert mix_drive(command) == wheels


def test_stop_button_beats_every_other_input():
    result = arbitrate_command(
        manual=MANUAL, autonomy=AUTONOMY, stop_requested=True, safety_stop_requested=True
    )
    assert result.command == STOP_COMMAND
    assert result.source == "stop"
    assert result.stop_requested


def test_safety_stop_beats_policy_and_forward_manual_input():
    for manual in (None, MANUAL, DriveCommand(0.0, 0.5)):
        result = arbitrate_command(manual=manual, autonomy=AUTONOMY, safety_stop_requested=True)
        assert result.command == STOP_COMMAND
        assert result.source == "safety_stop"
        assert result.stop_requested


def test_operator_can_reverse_out_of_the_safety_zone():
    result = arbitrate_command(manual=DriveCommand(-0.5, 0.2), autonomy=AUTONOMY, safety_stop_requested=True)
    assert result.command == DriveCommand(-0.5, 0.2)
    assert result.source == "manual"
    assert not result.stop_requested


def test_reversing_in_the_safety_zone_never_turns_a_wheel_forward():
    values = [-1.0, -0.75, -0.5, -0.25, -0.01]
    steerings = [-1.0, -0.6, -0.2, 0.0, 0.2, 0.6, 1.0]
    for throttle, steering in itertools.product(values, steerings):
        result = arbitrate_command(manual=DriveCommand(throttle, steering), safety_stop_requested=True)
        wheels = mix_drive(result.command)
        assert result.command.throttle == throttle
        assert wheels.left <= 0.0 and wheels.right <= 0.0, (throttle, steering, wheels)


def test_manual_input_beats_the_policy():
    result = arbitrate_command(manual=MANUAL, autonomy=AUTONOMY)
    assert result.command == MANUAL
    assert result.source == "manual"
    assert not result.stop_requested


def test_policy_drives_when_the_operator_is_hands_off():
    result = arbitrate_command(autonomy=AUTONOMY)
    assert result.command == AUTONOMY
    assert result.source == "autonomy"
    assert not result.stop_requested


def test_no_input_means_stop():
    result = arbitrate_command()
    assert result.command == STOP_COMMAND
    assert result.source == "idle"
    assert result.stop_requested
