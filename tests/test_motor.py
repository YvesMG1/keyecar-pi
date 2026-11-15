"""Motor driver against a fake GPIO module: duty cycles, direction pins, and stopping."""

from __future__ import annotations

import pytest

from keyecar.car import motor as motor_module
from keyecar.car.config import MotorSettings
from keyecar.car.control import DriveCommand, WheelCommand


class FakePWM:
    def __init__(self, pin: int, frequency: int):
        self.pin = pin
        self.frequency = frequency
        self.duty = None
        self.stopped = False

    def start(self, duty: int) -> None:
        self.duty = duty

    def ChangeDutyCycle(self, duty: int) -> None:  # noqa: N802 - RPi.GPIO's name
        self.duty = duty

    def stop(self) -> None:
        self.stopped = True


class FakeGPIO:
    BCM = "BCM"
    OUT = "OUT"
    HIGH = 1
    LOW = 0

    def __init__(self):
        self.levels: dict[int, int] = {}
        self.pwms: dict[int, FakePWM] = {}
        self.cleaned_up = False

    def setwarnings(self, flag: bool) -> None:
        pass

    def setmode(self, mode: str) -> None:
        pass

    def setup(self, pin: int, direction: str) -> None:
        pass

    def output(self, pin: int, level: int) -> None:
        self.levels[pin] = level

    def PWM(self, pin: int, frequency: int) -> FakePWM:  # noqa: N802 - RPi.GPIO's name
        self.pwms[pin] = FakePWM(pin, frequency)
        return self.pwms[pin]

    def cleanup(self) -> None:
        self.cleaned_up = True


@pytest.fixture
def gpio(monkeypatch):
    fake = FakeGPIO()
    monkeypatch.setattr(motor_module, "GPIO", fake)
    return fake


def duties(motors: motor_module.Motors) -> tuple[list[int], list[int]]:
    return [m.pwm.duty for m in motors._left], [m.pwm.duty for m in motors._right]


def spins_forward(gpio: FakeGPIO, motor: motor_module._Motor) -> bool:
    """Direction of the wheel itself, i.e. after undoing a swapped wiring."""
    electrically_forward = gpio.levels[motor.in1] == gpio.HIGH and gpio.levels[motor.in2] == gpio.LOW
    return electrically_forward != motor.reversed


def test_motors_need_gpio(monkeypatch):
    monkeypatch.setattr(motor_module, "GPIO", None)
    with pytest.raises(RuntimeError):
        motor_module.Motors()


def test_all_channels_start_at_zero_duty(gpio):
    motors = motor_module.Motors()
    assert duties(motors) == ([0, 0], [0, 0])
    assert len(gpio.pwms) == 4


def test_duty_is_command_times_speed_limit(gpio):
    motors = motor_module.Motors()
    motors.set_wheels(WheelCommand(1.0, 0.5), speed_limit=40)
    assert duties(motors) == ([40, 40], [20, 20])


def test_speed_limit_is_capped_at_max_duty(gpio):
    motors = motor_module.Motors(MotorSettings(max_duty=60))
    motors.set_wheels(WheelCommand(1.0, 1.0), speed_limit=100)
    assert duties(motors) == ([60, 60], [60, 60])
    motors.set_wheels(WheelCommand(1.0, 1.0), speed_limit=-10)
    assert duties(motors) == ([0, 0], [0, 0])


def test_both_motors_of_a_side_turn_the_same_way_despite_swapped_wiring(gpio):
    motors = motor_module.Motors()
    assert {m.reversed for m in motors._left} == {True, False}  # the default wiring has both kinds

    motors.set_wheels(WheelCommand(0.5, -0.5), speed_limit=50)
    assert all(spins_forward(gpio, m) for m in motors._left)
    assert not any(spins_forward(gpio, m) for m in motors._right)

    motors.set_wheels(WheelCommand(-0.5, 0.5), speed_limit=50)
    assert not any(spins_forward(gpio, m) for m in motors._left)
    assert all(spins_forward(gpio, m) for m in motors._right)


def test_apply_drive_mixes_before_driving(gpio):
    motors = motor_module.Motors()
    wheels = motors.apply_drive(DriveCommand(0.5, 0.25), speed_limit=100)
    assert wheels == WheelCommand(0.75, 0.25)
    assert duties(motors) == ([75, 75], [25, 25])


def test_stop_zeroes_duty_and_pulls_direction_pins_low(gpio):
    motors = motor_module.Motors()
    motors.set_wheels(WheelCommand(1.0, -1.0), speed_limit=80)
    motors.stop()
    assert duties(motors) == ([0, 0], [0, 0])
    assert all(gpio.levels[pin] == gpio.LOW for pin in motors.settings.pins.direction_pins)


def test_close_stops_the_car_and_releases_gpio(gpio):
    motors = motor_module.Motors()
    motors.set_wheels(WheelCommand(1.0, 1.0), speed_limit=80)
    pwms = list(gpio.pwms.values())
    motors.close()
    assert all(pwm.duty == 0 and pwm.stopped for pwm in pwms)
    assert gpio.cleaned_up
