"""Conditional driving policy: camera frame(s), route condition, previous command -> command.

A MobileNetV3-Small backbone (ImageNet weights, first stages frozen) encodes the
image. For history models the stacked frames first go through a 1x1 convolution
that maps them back to three channels. It starts as a passthrough of the newest
frame, so a history model begins as exactly the single-frame model.

The pooled image features are concatenated with a learned condition embedding,
a projection of the condition history (history models only), and a projection
of the previously executed command ("motor state"). A small MLP then predicts
throttle in [0, 1] (sigmoid) and steering in [-1, 1] (tanh).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

NUM_CONDITIONS = 3  # FORWARD, LEFT, RIGHT
BACKBONE_FEATURE_DIM = 576
MOTOR_PROJ_DIM = 16
CONDITION_EMBED_DIM = 8
CONDITION_HISTORY_PROJ_DIM = 16
TRUNK_DIM = 256


def _newest_frame_passthrough(adapter: nn.Conv2d) -> None:
    """Initialize a 1x1 convolution to copy the first three (newest) input channels."""
    with torch.no_grad():
        adapter.weight.zero_()
        adapter.bias.zero_()
        for channel in range(3):
            adapter.weight[channel, channel, 0, 0] = 1.0


class RegressionPolicy(nn.Module):
    """The driving policy (about 1M parameters).

    Args:
        input_channels: 3 for the current frame, 3 * N for an N-frame history.
        condition_history_length: if > 0, ``forward`` also takes the conditions
            of the last N frames.
        freeze_stages: number of early backbone stages that are not trained.
        pretrained: load ImageNet weights (not needed when loading a checkpoint).
    """

    def __init__(
        self,
        *,
        input_channels: int = 3,
        condition_history_length: int = 0,
        freeze_stages: int = 4,
        pretrained: bool = True,
    ):
        super().__init__()
        if input_channels <= 0 or input_channels % 3 != 0:
            raise ValueError("input_channels must be a positive multiple of 3 (stacked RGB frames)")
        if condition_history_length < 0:
            raise ValueError("condition_history_length must be non-negative")
        self.input_channels = int(input_channels)
        self.condition_history_length = int(condition_history_length)

        if self.input_channels == 3:
            self.input_adapter = nn.Identity()
        else:
            self.input_adapter = nn.Conv2d(self.input_channels, 3, kernel_size=1)
            _newest_frame_passthrough(self.input_adapter)

        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = mobilenet_v3_small(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        for parameter in self.features[:freeze_stages].parameters():
            parameter.requires_grad = False

        self.motor_proj = nn.Sequential(nn.Linear(2, MOTOR_PROJ_DIM), nn.ReLU(inplace=True))
        self.condition_embed = nn.Embedding(NUM_CONDITIONS, CONDITION_EMBED_DIM)
        trunk_in = BACKBONE_FEATURE_DIM + MOTOR_PROJ_DIM + CONDITION_EMBED_DIM
        self.condition_history_proj = None
        if self.condition_history_length > 0:
            self.condition_history_proj = nn.Sequential(
                nn.Linear(self.condition_history_length * CONDITION_EMBED_DIM, CONDITION_HISTORY_PROJ_DIM),
                nn.ReLU(inplace=True),
            )
            trunk_in += CONDITION_HISTORY_PROJ_DIM

        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, TRUNK_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(TRUNK_DIM, 2),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Return the pooled backbone features, shape (N, 576)."""
        return self.pool(self.features(self.input_adapter(image))).flatten(1)

    def head(
        self,
        features: torch.Tensor,
        motor_state: torch.Tensor,
        condition: torch.Tensor,
        condition_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict (throttle, steering) from precomputed image features."""
        parts = [features, self.motor_proj(motor_state), self.condition_embed(condition.long())]
        if self.condition_history_proj is not None:
            if condition_history is None:
                raise ValueError("this policy needs condition_history")
            history = self.condition_embed(condition_history.long()).flatten(1)
            parts.append(self.condition_history_proj(history))
        raw = self.trunk(torch.cat(parts, dim=1))
        return torch.cat([torch.sigmoid(raw[:, :1]), torch.tanh(raw[:, 1:2])], dim=1)

    def forward(
        self,
        image: torch.Tensor,
        motor_state: torch.Tensor,
        condition: torch.Tensor,
        condition_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (N, 2) commands: throttle in [0, 1], steering in [-1, 1]."""
        return self.head(self.encode_image(image), motor_state, condition, condition_history)

    def export_config(self) -> dict[str, int]:
        return {
            "input_channels": self.input_channels,
            "condition_history_length": self.condition_history_length,
        }


def batch_inputs(
    model: RegressionPolicy, batch: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The model inputs of a collated dataset batch, in ``forward`` order, on ``device``."""
    return (
        batch["image"].to(device),
        batch["motor_state"].to(device),
        batch["condition"].to(device),
        batch["condition_history"].to(device) if model.condition_history_length > 0 else None,
    )


def forward_batch(model: RegressionPolicy, batch: dict, device: torch.device) -> torch.Tensor:
    """Run the policy on a collated dataset batch."""
    return model(*batch_inputs(model, batch, device))


def default_device(*, allow_mps: bool = True) -> str:
    """CUDA if available, then Apple's MPS (if allowed), then the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if allow_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
