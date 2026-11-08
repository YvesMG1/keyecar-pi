"""Drive the car with a PlayStation controller and record human demonstrations.

    keyecar-teleop --track "track 3"
    keyecar-teleop --no-record            # just drive

The left stick drives. The face buttons set the route condition
(Triangle=FORWARD, Square=LEFT, Circle=RIGHT), which is stored with every
frame and becomes the policy's condition input.
"""

from __future__ import annotations

import argparse

from keyecar.cli import make_parser

from . import cli
from .runtime import run_playstation_teleop


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser(__doc__, prog="keyecar-teleop")
    cli.add_session_args(parser, record_default=True)
    parser.add_argument("--layout", default="", help="layout id stored in metadata.json")
    parser.add_argument("--phase", default="seed", help="collection phase stored in metadata.json")
    parser.add_argument("--lighting", default="", help="lighting note stored in metadata.json")
    # The demonstrations were recorded at 480x320, 20 fps.
    cli.add_camera_args(parser, width=480, height=320, fps=20.0)
    cli.add_safety_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_playstation_teleop(
        cli.runtime_settings(args),
        record=args.record,
        session_metadata={
            "track_name": args.track,
            "layout_id": args.layout,
            "phase": args.phase,
            "lighting": args.lighting,
            "notes": args.notes,
            "source": "teleop",
        },
    )


if __name__ == "__main__":
    main()
