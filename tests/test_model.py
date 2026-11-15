"""The policy network. Needs the training extras; skipped where torch is not installed."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from keyecar.training.model import RegressionPolicy  # noqa: E402


def policy(**options: object) -> RegressionPolicy:
    return RegressionPolicy(input_channels=9, condition_history_length=3, pretrained=False, **options).eval()


def inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    image = torch.randn(2, 9, 160, 160, generator=generator)
    return image, torch.tensor([0, 2]), torch.tensor([[0, 0, 1], [2, 2, 2]])


def test_outputs_are_throttle_and_steering_in_range() -> None:
    image, condition, history = inputs()
    with torch.no_grad():
        output = policy()(image, torch.zeros(2, 2), condition, history)
    assert output.shape == (2, 2)
    assert ((output[:, 0] >= 0) & (output[:, 0] <= 1)).all()
    assert (output[:, 1].abs() <= 1).all()


def test_previous_command_changes_the_prediction_only_if_the_policy_uses_it() -> None:
    image, condition, history = inputs()
    straight, hard_left = torch.zeros(2, 2), torch.tensor([[1.0, -1.0], [1.0, -1.0]])

    with torch.no_grad():
        uses_it = policy()
        assert not torch.allclose(
            uses_it(image, straight, condition, history), uses_it(image, hard_left, condition, history)
        )

        ignores_it = policy(use_motor_state=False)
        expected = ignores_it(image, None, condition, history)
        assert torch.equal(ignores_it(image, straight, condition, history), expected)
        assert torch.equal(ignores_it(image, hard_left, condition, history), expected)
    assert ignores_it.export_config()["use_motor_state"] is False


def test_freeze_stages_controls_how_much_of_the_backbone_trains() -> None:
    def trainable(model: RegressionPolicy) -> int:
        return sum(p.numel() for p in model.features.parameters() if p.requires_grad)

    assert trainable(policy(freeze_stages=0)) > trainable(policy(freeze_stages=4)) > 0


def test_steering_bins_decode_back_to_the_steering_they_were_trained_on() -> None:
    binned = policy(steering_bins=25)
    steering = torch.tensor([-1.0, -0.63, -0.05, 0.0, 0.41, 0.97, 1.0])
    targets = binned.soft_steering_targets(steering)
    assert targets.shape == (7, 25)
    assert torch.allclose(targets.sum(dim=1), torch.ones(7))

    decoded = binned.decode_steering_bins(targets.log()).squeeze(1)
    assert torch.allclose(decoded, steering, atol=0.02)
    assert decoded[0] == pytest.approx(-1.0, abs=1e-4)  # full lock is not pulled towards the centre
    assert decoded[-1] == pytest.approx(1.0, abs=1e-4)


def test_steering_bins_keep_two_answers_apart_instead_of_averaging_them() -> None:
    binned = policy(steering_bins=25)
    left_or_right = 0.45 * binned.soft_steering_targets(torch.tensor([-0.8])) + 0.55 * (
        binned.soft_steering_targets(torch.tensor([0.8]))
    )
    assert binned.decode_steering_bins(left_or_right.log()).item() == pytest.approx(0.8, abs=0.02)


def test_binned_policy_returns_the_same_two_outputs() -> None:
    image, condition, history = inputs()
    binned = policy(steering_bins=25, use_motor_state=False)
    with torch.no_grad():
        raw = binned.forward_raw(image, None, condition, history)
        output = binned(image, None, condition, history)
    assert raw.shape == (2, 26)
    assert output.shape == (2, 2)
    assert (output[:, 1].abs() <= 1).all()
    assert binned.export_config()["steering_bins"] == 25
