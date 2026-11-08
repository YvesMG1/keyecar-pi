"""Continuously print HC-SR04 readings to verify wiring on the Pi.

keyecar-ultrasonic-bench --interval 0.1
"""

from __future__ import annotations

import argparse
import time

from keyecar.cli import make_parser

from .config import UltrasonicSettings
from .ultrasonic import UltrasonicSensor, format_reading


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser(__doc__, prog="keyecar-ultrasonic-bench")
    parser.add_argument(
        "--interval", type=float, default=UltrasonicSettings().bench_interval_s, help="seconds between reads"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = UltrasonicSettings()
    print(
        f"Ultrasonic bench (trigger=GPIO{settings.pins.trigger}, echo=GPIO{settings.pins.echo}, "
        f"emergency_stop<={settings.emergency_stop_distance_cm:.1f}cm). Ctrl+C to stop."
    )
    sensor = UltrasonicSensor(settings=settings)
    try:
        while True:
            print(format_reading(sensor.read()))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.close()


if __name__ == "__main__":
    main()
