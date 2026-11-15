"""Conditional driving policy: camera frame(s), route condition, previous command -> command.

A MobileNetV3-Small backbone (ImageNet weights, first stages frozen) encodes the
image. For history models the stacked frames first go through a 1x1 convolution
that maps them back to three channels. It starts as a passthrough of the newest
frame, so a history model begins as exactly the single-frame model.

The pooled image features are concatenated with a learned condition embedding,
a projection of the condition history (history models only), and a projection
of the previously executed command ("motor state"). A small MLP then predicts
throttle in [0, 1] (sigmoid) and steering in [-1, 1] (tanh).

Optionally steering is predicted as a distribution over bins instead
(``steering_bins``). The policy still returns one steering value, the mean of
the bins around the most likely one, so nothing changes for the car. Unlike a
regression to the mean, this can put all its weight on full lock, and it can
keep two plausible answers apart instead of averaging them.
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
STEERING_BIN_RANGE = 1.2  # bins reach past full lock, so that +-1 has neighbours on both sides
STEERING_BIN_SIGMA = 1.0  # width of the soft training target, in bins
STEERING_DECODE_RADIUS = 2  # bins averaged on each side of the most likely one


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
        use_motor_state: feed the previous command to the head. Without it the
            policy cannot copy its last output; see "The previous-command
            problem" in docs/approach.md.
        steering_bins: if > 0, predict steering as a distribution over this
            many bins between -1.2 and 1.2 instead of regressing it.
        pretrained: load ImageNet weights (not needed when loading a checkpoint).
    """

    def __init__(
        self,
        *,
        input_channels: int = 3,
        condition_history_length: int = 0,
        freeze_stages: int = 4,
        use_motor_state: bool = True,
        steering_bins: int = 0,
        pretrained: bool = True,
    ):
        super().__init__()
        if input_channels <= 0 or input_channels % 3 != 0:
            raise ValueError("input_channels must be a positive multiple of 3 (stacked RGB frames)")
        if condition_history_length < 0:
            raise ValueError("condition_history_length must be non-negative")
        self.input_channels = int(input_channels)
        self.condition_history_length = int(condition_history_length)
        self.use_motor_state = bool(use_motor_state)
        self.steering_bins = int(steering_bins)
        if self.steering_bins != 0 and self.steering_bins < 2 * STEERING_DECODE_RADIUS + 1:
            raise ValueError(f"steering_bins must be 0 or at least {2 * STEERING_DECODE_RADIUS + 1}")
        if self.steering_bins > 0:
            centers = torch.linspace(-STEERING_BIN_RANGE, STEERING_BIN_RANGE, self.steering_bins)
            self.register_buffer("bin_centers", centers)

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

        self.condition_embed = nn.Embedding(NUM_CONDITIONS, CONDITION_EMBED_DIM)
        trunk_in = BACKBONE_FEATURE_DIM + CONDITION_EMBED_DIM
        self.motor_proj = None
        if self.use_motor_state:
            self.motor_proj = nn.Sequential(nn.Linear(2, MOTOR_PROJ_DIM), nn.ReLU(inplace=True))
            trunk_in += MOTOR_PROJ_DIM
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
            nn.Linear(TRUNK_DIM, 1 + max(self.steering_bins, 1)),
        )

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Return the pooled backbone features, shape (N, 576)."""
        return self.pool(self.features(self.input_adapter(image))).flatten(1)

    def head(
        self,
        features: torch.Tensor,
        motor_state: torch.Tensor | None,
        condition: torch.Tensor,
        condition_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict (throttle, steering) from precomputed image features.

        ``motor_state`` is ignored (and may be None) if the policy does not use it.
        """
        return self.decode(self.raw_head(features, motor_state, condition, condition_history))

    def raw_head(
        self,
        features: torch.Tensor,
        motor_state: torch.Tensor | None,
        condition: torch.Tensor,
        condition_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The undecoded outputs: the throttle logit, then the steering logit or one logit per bin."""
        parts = [features]
        if self.motor_proj is not None:
            parts.append(self.motor_proj(motor_state))
        parts.append(self.condition_embed(condition.long()))
        if self.condition_history_proj is not None:
            if condition_history is None:
                raise ValueError("this policy needs condition_history")
            history = self.condition_embed(condition_history.long()).flatten(1)
            parts.append(self.condition_history_proj(history))
        return self.trunk(torch.cat(parts, dim=1))

    def decode(self, raw: torch.Tensor) -> torch.Tensor:
        """Turn ``raw_head`` outputs into (throttle, steering)."""
        throttle = torch.sigmoid(raw[:, :1])
        if self.steering_bins == 0:
            return torch.cat([throttle, torch.tanh(raw[:, 1:2])], dim=1)
        return torch.cat([throttle, self.decode_steering_bins(raw[:, 1:])], dim=1)

    def decode_steering_bins(self, logits: torch.Tensor) -> torch.Tensor:
        """Mean of the bins around the most likely one, shape (N, 1).

        The mean over all bins would average two separate peaks, and the most
        likely bin alone would make the steering jump in steps.
        """
        probabilities = torch.softmax(logits, dim=1)
        peak = probabilities.argmax(dim=1, keepdim=True)
        index = torch.arange(self.steering_bins, device=logits.device).unsqueeze(0)
        near_peak = ((index - peak).abs() <= STEERING_DECODE_RADIUS).to(probabilities.dtype)
        weights = probabilities * near_peak
        steering = (weights * self.bin_centers).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)
        return steering.clamp(-1.0, 1.0)

    def soft_steering_targets(self, steering: torch.Tensor) -> torch.Tensor:
        """Training targets for the bins, shape (N, bins): a narrow bell around each steering value."""
        spacing = self.bin_centers[1] - self.bin_centers[0]
        distance = (self.bin_centers.unsqueeze(0) - steering.unsqueeze(1)) / (STEERING_BIN_SIGMA * spacing)
        return torch.softmax(-0.5 * distance * distance, dim=1)

    def forward_raw(
        self,
        image: torch.Tensor,
        motor_state: torch.Tensor | None,
        condition: torch.Tensor,
        condition_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``raw_head`` on images; training needs the bin logits for its loss."""
        return self.raw_head(self.encode_image(image), motor_state, condition, condition_history)

    def forward(
        self,
        image: torch.Tensor,
        motor_state: torch.Tensor | None,
        condition: torch.Tensor,
        condition_history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return (N, 2) commands: throttle in [0, 1], steering in [-1, 1]."""
        return self.head(self.encode_image(image), motor_state, condition, condition_history)

    def export_config(self) -> dict[str, int | bool]:
        return {
            "input_channels": self.input_channels,
            "condition_history_length": self.condition_history_length,
            "use_motor_state": self.use_motor_state,
            "steering_bins": self.steering_bins,
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
