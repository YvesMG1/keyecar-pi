"""Command-line options shared by ``keyecar-teleop`` and ``keyecar-drive``."""

from __future__ import annotations

import argparse

from .config import (
    MotorSettings,
    RuntimeSettings,
    TeleopSettings,
    UltrasonicSettings,
    build_camera_settings,
)


def add_camera_args(parser: argparse.ArgumentParser, *, width: int, height: int, fps: float) -> None:
    group = parser.add_argument_group("camera")
    group.add_argument("--width", type=int, default=width, help="capture width")
    group.add_argument("--height", type=int, default=height, help="capture height")
    group.add_argument("--fps", type=float, default=fps, help="capture and recording rate")
    group.add_argument(
        "--jpeg-quality",
        type=int,
        default=build_camera_settings().jpeg_quality,
        help="quality of recorded JPEGs",
    )


def add_session_args(parser: argparse.ArgumentParser, *, record_default: bool) -> None:
    group = parser.add_argument_group("recording")
    group.add_argument(
        "--record",
        action=argparse.BooleanOptionalAction,
        default=record_default,
        help="record a schema-v2 session to --session-root",
    )
    group.add_argument(
        "--session-root", default=TeleopSettings().session_root, help="parent folder for sessions"
    )
    group.add_argument("--track", default="", help="track name stored in metadata.json")
    group.add_argument("--notes", default="", help="free-form notes stored in metadata.json")


def add_safety_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("safety")
    group.add_argument(
        "--speed-limit",
        type=int,
        default=MotorSettings().default_speed_limit,
        help="initial PWM duty limit in percent; L1/R1 change it while driving",
    )
    group.add_argument(
        "--ultrasonic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use the front HC-SR04 as a last-resort emergency stop",
    )


def runtime_settings(args: argparse.Namespace) -> RuntimeSettings:
    """Build the control-loop settings from the camera, recording, and safety options."""
    return RuntimeSettings(
        motor=MotorSettings(default_speed_limit=args.speed_limit),
        camera=build_camera_settings(
            width=args.width, height=args.height, fps=args.fps, jpeg_quality=args.jpeg_quality
        ),
        teleop=TeleopSettings(session_root=args.session_root),
        ultrasonic=UltrasonicSettings(enabled=args.ultrasonic),
    )
