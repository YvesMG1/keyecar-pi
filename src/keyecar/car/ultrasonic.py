"""HC-SR04 ultrasonic distance sensor support."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum

from .config import UltrasonicSettings

try:
    import RPi.GPIO as GPIO
except ImportError:  # not running on a Raspberry Pi
    GPIO = None


class UltrasonicStatus(StrEnum):
    """High-level result of one ultrasonic measurement attempt."""

    OK = "ok"
    ECHO_START_TIMEOUT = "echo_start_timeout"
    ECHO_END_TIMEOUT = "echo_end_timeout"
    OUT_OF_RANGE = "out_of_range"


@dataclass(frozen=True)
class UltrasonicReading:
    """One raw ultrasonic sample plus filtered safety state."""

    status: UltrasonicStatus
    sample_started_monotonic_ns: int
    raw_timestamp_monotonic_ns: int
    filtered_timestamp_monotonic_ns: int | None
    raw_distance_cm: float | None
    filtered_distance_cm: float | None
    raw_valid: bool
    filtered_valid: bool
    emergency_stop: bool
    pulse_duration_s: float | None = None


class ExponentialDistanceFilter:
    """Simple EMA used to smooth valid ultrasonic samples."""

    def __init__(self, alpha: float):
        self.alpha = min(max(float(alpha), 0.0), 1.0)
        self._value: float | None = None

    @property
    def value(self) -> float | None:
        """Return the most recent filtered distance, if any."""
        return self._value

    def reset(self) -> None:
        """Forget all prior readings."""
        self._value = None

    def update(self, sample_cm: float) -> float:
        """Blend one new valid distance sample into the filter state."""
        sample = float(sample_cm)
        if self._value is None:
            self._value = sample
        else:
            self._value = self.alpha * sample + (1.0 - self.alpha) * self._value
        return self._value


class UltrasonicSensor:
    """Owns the trigger and echo pins of the front HC-SR04 sensor.

    ``gpio_module``, ``clock_ns`` and ``sleep_fn`` can be replaced to run the
    sensor logic without hardware.
    """

    SPEED_OF_SOUND_CM_PER_S = 34_300.0

    def __init__(
        self,
        settings: UltrasonicSettings | None = None,
        *,
        gpio_module=None,
        clock_ns=None,
        sleep_fn=None,
    ):
        if gpio_module is None:
            if GPIO is None:
                raise RuntimeError("RPi.GPIO is not available in this environment.")
            gpio_module = GPIO

        self.settings = settings or UltrasonicSettings()
        self.gpio = gpio_module
        self.clock_ns = clock_ns or time.monotonic_ns
        self.sleep_fn = sleep_fn or time.sleep
        self.filter = ExponentialDistanceFilter(alpha=self.settings.filter_alpha)

        pins = self.settings.pins

        self.gpio.setwarnings(False)
        self.gpio.setmode(self.gpio.BCM)
        self.gpio.setup(pins.trigger, self.gpio.OUT)
        self.gpio.setup(pins.echo, self.gpio.IN)
        self.gpio.output(pins.trigger, self.gpio.LOW)
        self.sleep_fn(self.settings.settle_time_s)

    def _wait_for_level(self, pin: int, target_level: int, timeout_s: float) -> int | None:
        """Wait for a pin to reach one level, returning the observed timestamp."""
        deadline_ns = self.clock_ns() + int(timeout_s * 1_000_000_000)
        while self.gpio.input(pin) != target_level:
            if self.clock_ns() >= deadline_ns:
                return None
        return self.clock_ns()

    def _trigger(self) -> int:
        """Send the HC-SR04 trigger pulse and return the sample start time."""
        trigger_pin = self.settings.pins.trigger
        self.gpio.output(trigger_pin, self.gpio.LOW)
        self.sleep_fn(0.000002)
        sample_started_ns = self.clock_ns()
        self.gpio.output(trigger_pin, self.gpio.HIGH)
        self.sleep_fn(self.settings.trigger_pulse_s)
        self.gpio.output(trigger_pin, self.gpio.LOW)
        return sample_started_ns

    @staticmethod
    def _invalid_reading(
        status: UltrasonicStatus, sample_started_ns: int, timestamp_ns: int
    ) -> UltrasonicReading:
        return UltrasonicReading(
            status=status,
            sample_started_monotonic_ns=sample_started_ns,
            raw_timestamp_monotonic_ns=timestamp_ns,
            filtered_timestamp_monotonic_ns=None,
            raw_distance_cm=None,
            filtered_distance_cm=None,
            raw_valid=False,
            filtered_valid=False,
            emergency_stop=False,
        )

    def read(self) -> UltrasonicReading:
        """Measure one distance sample and update the filtered estimate."""
        sample_started_ns = self._trigger()
        echo_pin = self.settings.pins.echo

        echo_start_ns = self._wait_for_level(echo_pin, self.gpio.HIGH, self.settings.echo_timeout_s)
        if echo_start_ns is None:
            return self._invalid_reading(
                UltrasonicStatus.ECHO_START_TIMEOUT, sample_started_ns, self.clock_ns()
            )

        echo_end_ns = self._wait_for_level(echo_pin, self.gpio.LOW, self.settings.echo_timeout_s)
        if echo_end_ns is None:
            return self._invalid_reading(
                UltrasonicStatus.ECHO_END_TIMEOUT, sample_started_ns, self.clock_ns()
            )

        pulse_duration_s = max((echo_end_ns - echo_start_ns) / 1_000_000_000, 0.0)
        raw_distance_cm = pulse_duration_s * self.SPEED_OF_SOUND_CM_PER_S / 2.0  # there and back
        if (
            raw_distance_cm < self.settings.min_valid_distance_cm
            or raw_distance_cm > self.settings.max_valid_distance_cm
        ):
            return self._invalid_reading(UltrasonicStatus.OUT_OF_RANGE, sample_started_ns, echo_end_ns)

        filtered_distance_cm = self.filter.update(raw_distance_cm)
        emergency_stop = filtered_distance_cm <= self.settings.emergency_stop_distance_cm
        return UltrasonicReading(
            status=UltrasonicStatus.OK,
            sample_started_monotonic_ns=sample_started_ns,
            raw_timestamp_monotonic_ns=echo_end_ns,
            filtered_timestamp_monotonic_ns=echo_end_ns,
            raw_distance_cm=raw_distance_cm,
            filtered_distance_cm=filtered_distance_cm,
            raw_valid=True,
            filtered_valid=True,
            emergency_stop=emergency_stop,
            pulse_duration_s=pulse_duration_s,
        )

    def close(self) -> None:
        """Release only the trigger/echo pins owned by this sensor."""
        pins = self.settings.pins
        try:
            self.gpio.output(pins.trigger, self.gpio.LOW)
        finally:
            self.gpio.cleanup([pins.trigger, pins.echo])


def format_reading(reading: UltrasonicReading) -> str:
    """Render one reading in a compact human-readable form."""
    raw_str = "--" if reading.raw_distance_cm is None else f"{reading.raw_distance_cm:6.2f} cm"
    filtered_str = "--" if reading.filtered_distance_cm is None else f"{reading.filtered_distance_cm:6.2f} cm"
    return (
        f"status={reading.status.value:<18} "
        f"raw={raw_str:<10} "
        f"filtered={filtered_str:<10} "
        f"emergency_stop={int(reading.emergency_stop)}"
    )
