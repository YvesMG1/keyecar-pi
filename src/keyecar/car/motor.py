"""PWM motor driver for the KS0223 four-wheel skid-steer chassis."""

from __future__ import annotations

import contextlib
import gc
from dataclasses import dataclass

from .config import MotorSettings
from .control import DriveCommand, WheelCommand, clamp, mix_drive

try:
    import RPi.GPIO as GPIO
except ImportError:  # not running on a Raspberry Pi
    GPIO = None

if GPIO is not None:
    # PWM.__del__ can raise once GPIO.cleanup() has released the pins. close()
    # stops every channel explicitly, so the destructor has nothing left to do.
    with contextlib.suppress(Exception):
        GPIO.PWM.__del__ = lambda self: None


@dataclass
class _Motor:
    """One DC motor: two direction pins and a PWM channel."""

    in1: int
    in2: int
    pwm: object
    reversed: bool

    def drive(self, forward: bool, duty: int) -> None:
        if self.reversed:
            forward = not forward
        GPIO.output(self.in1, GPIO.HIGH if forward else GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW if forward else GPIO.HIGH)
        self.pwm.ChangeDutyCycle(duty)


class Motors:
    """Owns the GPIO pins and PWM channels of all four motors."""

    def __init__(self, settings: MotorSettings | None = None):
        if GPIO is None:
            raise RuntimeError("RPi.GPIO is not available in this environment.")

        self.settings = settings or MotorSettings()
        pins = self.settings.pins
        reversed_motors = self.settings.reversed_motors

        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        for pin in pins.direction_pins + pins.pwm_pins:
            GPIO.setup(pin, GPIO.OUT)
        for pin in pins.direction_pins:
            GPIO.output(pin, GPIO.LOW)

        def motor(in1: int, in2: int, pwm_pin: int, name: str) -> _Motor:
            pwm = GPIO.PWM(pwm_pin, self.settings.pwm_frequency_hz)
            pwm.start(0)
            return _Motor(in1, in2, pwm, reversed_motors[name])

        self._left = [
            motor(pins.left_m2_in1, pins.left_m2_in2, pins.left_m2_pwm, "L1"),
            motor(pins.left_m1_in1, pins.left_m1_in2, pins.left_m1_pwm, "L2"),
        ]
        self._right = [
            motor(pins.right_m3_in1, pins.right_m3_in2, pins.right_m3_pwm, "R1"),
            motor(pins.right_m4_in1, pins.right_m4_in2, pins.right_m4_pwm, "R2"),
        ]

    @property
    def _motors(self) -> list[_Motor]:
        return self._left + self._right

    def set_wheels(self, wheels: WheelCommand, speed_limit: int) -> WheelCommand:
        """Drive both sides; ``speed_limit`` is the duty cycle (percent) at full command."""
        limit = int(clamp(speed_limit, 0, self.settings.max_duty))
        for side, command in ((self._left, wheels.left), (self._right, wheels.right)):
            duty = int(abs(command) * limit)
            forward = command >= 0 or duty == 0
            for motor in side:
                motor.drive(forward, duty)
        return wheels

    def apply_drive(self, command: DriveCommand, speed_limit: int) -> WheelCommand:
        """Mix a drive command into wheel speeds and send them to the motors."""
        return self.set_wheels(mix_drive(command), speed_limit)

    def stop(self) -> None:
        """Set every duty cycle to zero and pull all direction pins low."""
        for motor in self._motors:
            motor.pwm.ChangeDutyCycle(0)
        for pin in self.settings.pins.direction_pins:
            GPIO.output(pin, GPIO.LOW)

    def close(self) -> None:
        """Stop the car, then release all PWM channels and GPIO pins."""
        try:
            self.stop()
        finally:
            for motor in self._motors:
                with contextlib.suppress(Exception):
                    motor.pwm.stop()
            self._left, self._right = [], []
            gc.collect()  # drop the PWM objects before cleanup releases their pins
            GPIO.cleanup()
