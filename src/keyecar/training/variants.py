"""The two policy variants. They share the network and differ only in temporal context.

* ``rgb_current``: the current frame and the current condition.
* ``rgb_history``: the last ``history_length`` frames (newest first) stacked
  along the channel axis, plus the matching condition history.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_HISTORY_LENGTH = 3


@dataclass(frozen=True)
class PolicyVariant:
    name: str
    history_length: int

    @property
    def input_channels(self) -> int:
        return 3 * self.history_length

    @property
    def condition_history_length(self) -> int:
        return self.history_length if self.history_length > 1 else 0

    def to_dict(self) -> dict[str, object]:
        """The form stored in checkpoints and ``policy_metadata.json``."""
        return {
            "name": self.name,
            "history_length": self.history_length,
            "input_channels": self.input_channels,
            "condition_history_length": self.condition_history_length,
        }


VARIANT_NAMES = ("rgb_current", "rgb_history")


def resolve_variant(name: str, *, history_length: int = DEFAULT_HISTORY_LENGTH) -> PolicyVariant:
    if name == "rgb_current":
        return PolicyVariant(name, history_length=1)
    if name == "rgb_history":
        return PolicyVariant(name, history_length=int(history_length))
    raise ValueError(f"Unknown policy variant {name!r}; expected one of {VARIANT_NAMES}")


def variant_from_dict(data: dict[str, object]) -> PolicyVariant:
    """Rebuild a variant from checkpoint or metadata contents."""
    return resolve_variant(str(data["name"]), history_length=int(data["history_length"]))
