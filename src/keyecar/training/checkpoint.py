"""Load training checkpoints written by :mod:`keyecar.training.train`."""

from __future__ import annotations

from pathlib import Path, PosixPath, WindowsPath

import torch

from .model import RegressionPolicy
from .variants import PolicyVariant, variant_from_dict

# Older checkpoints stored pathlib objects in their config, and Python 3.13 moved
# those classes to ``pathlib._local``. Allow both names so torch.load's safe
# (weights_only) unpickler accepts checkpoints from either Python version.
_PATH_GLOBALS = [
    PosixPath,
    WindowsPath,
    (PosixPath, "pathlib._local.PosixPath"),
    (WindowsPath, "pathlib._local.WindowsPath"),
]


def load_checkpoint(path: Path | str) -> dict:
    """Load a checkpoint on the CPU."""
    with torch.serialization.safe_globals(_PATH_GLOBALS):
        return torch.load(path, map_location="cpu")


def variant_from_checkpoint(state: dict) -> PolicyVariant:
    return variant_from_dict(state["variant"])


def model_from_checkpoint(state: dict) -> RegressionPolicy:
    """Build the policy (without downloading ImageNet weights) and load the trained weights."""
    variant = variant_from_checkpoint(state)
    model = RegressionPolicy(
        input_channels=variant.input_channels,
        condition_history_length=variant.condition_history_length,
        pretrained=False,
    )
    model.load_state_dict(state["state_dict"])
    return model
