"""Argparse setup shared by all command-line tools."""

from __future__ import annotations

import argparse
from pathlib import Path


class HelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    """Keep the layout of the module docstring and show the options' defaults.

    Defaults that say nothing are left out: ``None`` (explained in the help
    text), ``False`` of on/off switches, and empty strings.
    """

    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.default is None or action.default is False or action.default == "":
            return action.help
        return super()._get_help_string(action)


def make_parser(doc: str | None, *, prog: str | None = None) -> argparse.ArgumentParser:
    """Return a parser described by ``doc``, usually the module docstring with usage examples."""
    return argparse.ArgumentParser(prog=prog, description=doc, formatter_class=HelpFormatter)


def add_recordings_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--recordings-root",
        type=Path,
        default=Path("recordings"),
        help="folder with the train/ and val/ session folders",
    )


def add_checkpoint(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("checkpoint", type=Path, help="training checkpoint (best.pt)")


def add_keep_frames(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--keep-frames", action="store_true", help="keep the rendered frames next to the video"
    )
