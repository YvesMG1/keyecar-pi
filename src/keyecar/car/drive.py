"""Drive autonomously with an exported policy; the operator can take over any time.

    keyecar-drive --model models/rgb_history_dagger_v4 --track 7

The PlayStation controller stays connected: any left-stick deflection takes
manual control until the stick is released, and those takeovers are recorded as
interventions for the next DAgger round. The face buttons select the route
condition (FORWARD / LEFT / RIGHT); Cross stops; Options quits.
"""

from __future__ import annotations

import argparse

from keyecar.cli import make_parser

from . import cli
from .inference import resolve_bundle
from .runtime import run_autonomy_loop


def build_parser() -> argparse.ArgumentParser:
    parser = make_parser(__doc__, prog="keyecar-drive")
    parser.add_argument(
        "--model", default="models/rgb_history_dagger_v4", help="policy bundle folder or .onnx file"
    )
    parser.add_argument("--int8", action="store_true", help="load policy.int8.onnx from the bundle")
    parser.add_argument(
        "--ema",
        type=float,
        default=0.0,
        help="output smoothing factor in [0, 1); higher values delay steering",
    )
    parser.add_argument(
        "--diagnostics", action="store_true", help="log the latency of every inference and record every frame"
    )
    cli.add_session_args(parser, record_default=True)
    cli.add_camera_args(parser, width=640, height=480, fps=20.0)
    cli.add_safety_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 <= args.ema < 1.0:
        parser.error("--ema must be in [0, 1); 1 would freeze the policy output")
    model_path, metadata_path = resolve_bundle(args.model, int8=args.int8)
    run_autonomy_loop(
        cli.runtime_settings(args),
        model_path=model_path,
        metadata_path=metadata_path,
        record=args.record,
        session_metadata={
            "track_name": args.track,
            "notes": args.notes,
            "source": "autonomy",
            "policy_bundle": str(args.model),
        },
        ema_alpha=args.ema,
        diagnostics=args.diagnostics,
    )


if __name__ == "__main__":
    main()
