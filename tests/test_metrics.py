"""Offline metrics on synthetic command sequences: a steering step or a whole turn of the human at 20 Hz."""

from __future__ import annotations

import numpy as np
import pytest

from keyecar.car.control import Condition
from keyecar.training.metrics import (
    TurnConfig,
    TurnInConfig,
    baseline_metrics,
    build_robustness_metrics_payload,
    change_slice_metrics,
    intervention_residual_closure_metrics,
    sequence_breaks,
    skill,
    turn_following_metrics,
    turn_in_events,
    turn_in_lag_metrics,
    turn_segments,
)

FRAME_NS = 50_000_000  # 20 Hz
STRAIGHT_FRAMES, TURN_FRAMES = 10, 30
ONSET = STRAIGHT_FRAMES
STEP = 0.8


def human_steering(step: float = STEP) -> np.ndarray:
    """Straight for half a second, then a constant turn."""
    return np.concatenate([np.zeros(STRAIGHT_FRAMES), np.full(TURN_FRAMES, step)])


def commands(steering: np.ndarray) -> np.ndarray:
    return np.column_stack([np.full(len(steering), 0.5), steering])


def delayed(signal: np.ndarray, frames: int) -> np.ndarray:
    return np.concatenate([np.full(frames, signal[0]), signal[: len(signal) - frames]])


def timestamps(count: int) -> np.ndarray:
    return np.arange(count, dtype=np.int64) * FRAME_NS


def one_run(count: int) -> np.ndarray:
    breaks = np.zeros(count, dtype=bool)
    breaks[0] = True
    return breaks


def whole_turn(step: float = STEP, turn_frames: int = TURN_FRAMES) -> np.ndarray:
    """Straight for a second, a constant turn, then straight for a second."""
    straight = np.zeros(2 * STRAIGHT_FRAMES)
    return np.concatenate([straight, np.full(turn_frames, step), straight])


def turn_metrics(prediction: np.ndarray, target: np.ndarray | None = None, **options) -> dict[str, object]:
    target = whole_turn() if target is None else target
    times = timestamps(len(target))
    turns = turn_segments(target, times, one_run(len(target)))
    return turn_following_metrics(prediction, target, times, turns, **options)


def lag_metrics(prediction: np.ndarray) -> dict[str, object]:
    target = human_steering()
    events = turn_in_events(target, timestamps(len(target)), one_run(len(target)))
    return turn_in_lag_metrics(prediction, target, timestamps(len(target)), events)


# ---------------------------------------------------------------------------
# Baselines and change slices
# ---------------------------------------------------------------------------


def test_skill_is_relative_to_the_baseline() -> None:
    assert skill(0.1, 0.1) == pytest.approx(0.0)
    assert skill(0.0, 0.1) == pytest.approx(1.0)
    assert skill(0.2, 0.1) == pytest.approx(-1.0)
    assert skill(0.1, 0.0) is None
    assert skill(None, 0.1) is None


def test_copying_is_wrong_only_on_the_frame_where_the_human_moves() -> None:
    target = commands(human_steering())
    previous = commands(delayed(human_steering(), 1))

    copy = baseline_metrics(previous, target)["copy_previous"]["overall"]
    assert copy["steering_mae"] == pytest.approx(STEP / len(target))

    slices = change_slice_metrics(previous, target, previous)
    assert slices["changed"]["num_samples"] == 1
    assert slices["held"]["num_samples"] == len(target) - 1
    assert slices["changed"]["steering_mae"] == pytest.approx(STEP)
    assert slices["changed"]["steering_skill_vs_copy"] == pytest.approx(0.0)
    assert slices["changed"]["median_step_ratio"] == pytest.approx(0.0)


def test_step_ratio_says_how_much_of_the_human_step_the_model_follows() -> None:
    target = commands(human_steering())
    previous = commands(delayed(human_steering(), 1))

    perfect = change_slice_metrics(target, target, previous)["changed"]
    assert perfect["median_step_ratio"] == pytest.approx(1.0)
    assert perfect["steering_skill_vs_copy"] == pytest.approx(1.0)

    halfway = change_slice_metrics(commands(0.5 * human_steering()), target, previous)["changed"]
    assert halfway["median_step_ratio"] == pytest.approx(0.5)

    wrong_way = change_slice_metrics(commands(-human_steering()), target, previous)["changed"]
    assert wrong_way["wrong_way_rate"] == pytest.approx(1.0)


def test_sequence_breaks_flag_new_sessions_missing_frames_and_late_frames() -> None:
    sessions = ["a", "a", "a", "b", "b"]
    frames = [1, 2, 4, 1, 2]
    assert sequence_breaks(sessions, frames).tolist() == [True, False, True, True, False]

    times = [0, FRAME_NS, 2 * FRAME_NS, 3 * FRAME_NS, 10 * FRAME_NS]
    late = sequence_breaks(sessions, frames, times, max_gap_ns=4 * FRAME_NS)
    assert late.tolist() == [True, False, True, True, True]


# ---------------------------------------------------------------------------
# Turn-in lag
# ---------------------------------------------------------------------------


def test_one_steering_step_is_one_event() -> None:
    target = human_steering()
    (event,) = turn_in_events(target, timestamps(len(target)), one_run(len(target)))
    assert event.onset == ONSET
    assert (event.start_level, event.end_level) == (0.0, STEP)


def test_small_steps_and_interrupted_steps_are_not_events() -> None:
    small = human_steering(step=0.3)
    assert turn_in_events(small, timestamps(len(small)), one_run(len(small))) == []

    target = human_steering()
    interrupted = one_run(len(target))
    interrupted[ONSET + 5] = True  # frames are missing before the timeout has passed
    assert turn_in_events(target, timestamps(len(target)), interrupted) == []

    strict = TurnInConfig(min_step=0.9)
    assert turn_in_events(target, timestamps(len(target)), one_run(len(target)), strict) == []


@pytest.mark.parametrize("frames_late", [0, 1, 3])
def test_lag_is_the_delay_behind_the_human(frames_late: int) -> None:
    metrics = lag_metrics(delayed(human_steering(), frames_late))
    assert metrics["num_events"] == 1
    assert metrics["median_lag_ms"] == pytest.approx(frames_late * FRAME_NS / 1e6)
    assert metrics["missed_share"] == 0.0
    assert metrics["median_peak_ratio"] == pytest.approx(1.0)


def test_a_model_that_never_turns_misses_the_step() -> None:
    metrics = lag_metrics(np.zeros(STRAIGHT_FRAMES + TURN_FRAMES))
    assert metrics["missed_share"] == 1.0
    assert metrics["median_lag_ms"] is None
    assert metrics["median_peak_ratio"] == pytest.approx(0.0)


def test_understeer_shows_in_the_peak_ratio() -> None:
    metrics = lag_metrics(0.75 * human_steering())
    assert metrics["missed_share"] == 0.0
    assert metrics["median_peak_ratio"] == pytest.approx(0.75)


def test_a_model_already_turning_has_no_lag_to_measure() -> None:
    metrics = lag_metrics(np.full(STRAIGHT_FRAMES + TURN_FRAMES, STEP))
    assert metrics["already_past_share"] == 1.0
    assert metrics["num_lag_measured"] == 0


# ---------------------------------------------------------------------------
# Whole turns
# ---------------------------------------------------------------------------


def test_one_turn_is_found_with_the_straight_driving_around_it() -> None:
    target = whole_turn()
    (turn,) = turn_segments(target, timestamps(len(target)), one_run(len(target)))
    first = 2 * STRAIGHT_FRAMES
    assert (turn.start, turn.stop, turn.direction) == (first, first + TURN_FRAMES, 1.0)
    pad_frames = 10  # half a second
    assert (turn.window_start, turn.window_stop) == (first - pad_frames, first + TURN_FRAMES + pad_frames)


def test_cut_off_and_small_turns_are_not_turns() -> None:
    cut_off = human_steering()  # the recording ends in the turn
    assert turn_segments(cut_off, timestamps(len(cut_off)), one_run(len(cut_off))) == []

    interrupted = one_run(len(whole_turn()))
    interrupted[2 * STRAIGHT_FRAMES + 5] = True  # frames are missing inside the turn
    assert turn_segments(whole_turn(), timestamps(len(interrupted)), interrupted) == []

    for small in (whole_turn(step=0.3), whole_turn(turn_frames=3)):  # too weak, too short
        assert turn_segments(small, timestamps(len(small)), one_run(len(small))) == []


def test_the_window_stops_where_the_human_turns_the_other_way() -> None:
    target = np.concatenate([np.zeros(20), np.full(20, STEP), np.full(20, -STEP), np.zeros(20)])
    right, left = turn_segments(target, timestamps(len(target)), one_run(len(target)))
    assert (right.window_stop, left.window_start) == (40, 40)
    assert (right.direction, left.direction) == (1.0, -1.0)


@pytest.mark.parametrize("frames_late", [0, 3, 8])
def test_a_late_turn_is_still_a_followed_turn(frames_late: int) -> None:
    metrics = turn_metrics(delayed(whole_turn(), frames_late))
    assert metrics["num_turns"] == 1
    assert metrics["median_steering_ratio"] == pytest.approx(1.0)
    assert metrics["followed_share"] == 1.0


def test_a_harder_and_shorter_turn_is_still_a_followed_turn() -> None:
    target = whole_turn(step=0.5)
    prediction = np.zeros_like(target)
    middle = 2 * STRAIGHT_FRAMES + TURN_FRAMES // 4
    prediction[middle : middle + TURN_FRAMES // 2] = 1.0
    metrics = turn_metrics(prediction, target)
    assert metrics["median_steering_ratio"] == pytest.approx(1.0, abs=0.05)
    assert metrics["followed_share"] == 1.0


def test_understeer_and_oversteer_are_not_followed_turns() -> None:
    weak = turn_metrics(0.5 * whole_turn())
    assert weak["median_steering_ratio"] == pytest.approx(0.5)
    assert (weak["followed_share"], weak["understeered_share"]) == (0.0, 1.0)

    none = turn_metrics(np.zeros_like(whole_turn()))
    assert none["median_steering_ratio"] == pytest.approx(0.0)

    strong = turn_metrics(whole_turn(step=0.9), whole_turn(step=0.6))
    assert strong["median_steering_ratio"] == pytest.approx(1.5)
    assert strong["oversteered_share"] == 1.0

    lenient = turn_metrics(0.5 * whole_turn(), config=TurnConfig(followed_low=0.4))
    assert lenient["followed_share"] == 1.0


def test_turns_are_split_by_the_route_command() -> None:
    target = np.concatenate([whole_turn(), whole_turn()])
    count = len(whole_turn())
    conditions = np.array([Condition.FORWARD] * count + [Condition.RIGHT] * count)
    prediction = np.concatenate([whole_turn(), 0.5 * whole_turn()])
    by_condition = turn_metrics(prediction, target, conditions=conditions)["by_condition"]
    assert by_condition["forward"]["median_steering_ratio"] == pytest.approx(1.0)
    assert by_condition["right_turn"]["median_steering_ratio"] == pytest.approx(0.5)
    assert by_condition["left_turn"]["num_turns"] == 0


# ---------------------------------------------------------------------------
# Payload and residual closure
# ---------------------------------------------------------------------------


def test_payload_scores_every_motor_mode_against_the_label_per_session() -> None:
    steering = np.concatenate([human_steering(), human_steering()])
    count = len(human_steering())
    target = commands(steering)
    session_ids = ["a"] * count + ["b"] * count
    frame_index = np.concatenate([np.arange(count), np.arange(count)])
    times = np.concatenate([timestamps(count), timestamps(count)])
    previous = commands(np.concatenate([delayed(human_steering(), 1)] * 2))
    evaluation = {
        "predictions": {"recorded": target, "zero": commands(0.5 * steering), "recursive": previous},
        "targets": target,
        "previous": previous,
        "session_ids": session_ids,
        "timestamps_ns": times,
        "breaks": sequence_breaks(session_ids, frame_index, times, max_gap_ns=4 * FRAME_NS),
    }

    payload = build_robustness_metrics_payload("candidate", evaluation)
    modes = payload["accuracy_by_motor_mode"]["modes"]

    assert payload["baselines"]["copy_previous"]["overall"]["steering_mae"] == pytest.approx(STEP / count)
    assert modes["recorded"]["steering_skill_vs_copy"] == pytest.approx(1.0)
    assert modes["recursive"]["steering_skill_vs_copy"] == pytest.approx(0.0)
    assert modes["recursive"]["turn_in"]["median_lag_ms"] == pytest.approx(FRAME_NS / 1e6)
    assert modes["zero"]["turn_in"]["median_peak_ratio"] == pytest.approx(0.5)
    assert set(modes["recorded"]["per_session"]) == {"a", "b"}
    assert modes["recorded"]["per_session"]["b"]["turn_in_num_events"] == 1


def test_payload_scores_whole_turns_in_every_motor_mode() -> None:
    steering = np.concatenate([whole_turn(), whole_turn()])
    count = len(whole_turn())
    target = commands(steering)
    session_ids = ["a"] * count + ["b"] * count
    frame_index = np.concatenate([np.arange(count), np.arange(count)])
    times = np.concatenate([timestamps(count), timestamps(count)])
    previous = commands(np.concatenate([delayed(whole_turn(), 1)] * 2))
    evaluation = {
        "predictions": {"recorded": previous, "zero": commands(0.5 * steering), "recursive": target},
        "targets": target,
        "previous": previous,
        "session_ids": session_ids,
        "timestamps_ns": times,
        "conditions": np.full(2 * count, int(Condition.LEFT)),
        "breaks": sequence_breaks(session_ids, frame_index, times, max_gap_ns=4 * FRAME_NS),
    }

    modes = build_robustness_metrics_payload("candidate", evaluation)["accuracy_by_motor_mode"]["modes"]

    assert modes["recursive"]["turns"]["num_turns"] == 2
    # A copy of the previous command is one frame late and steers just as much.
    assert modes["recorded"]["turns"]["median_steering_ratio"] == pytest.approx(1.0)
    assert modes["zero"]["turns"]["understeered_share"] == 1.0
    assert modes["zero"]["turns"]["by_condition"]["left_turn"]["num_turns"] == 2
    assert modes["zero"]["per_session"]["b"]["turn_median_steering_ratio"] == pytest.approx(0.5)


def test_first_human_frame_closure_is_reported_separately() -> None:
    metrics = intervention_residual_closure_metrics(
        baseline=np.zeros(3),
        target=np.ones(3),
        prediction=np.array([0.5, 1.0, 0.0]),
        regions=["human", "human", "pre"],
        split="val",
        first_human=np.array([True, False, False]),
    )
    first = metrics["first_human_frame"]
    assert first["num_samples"] == 1
    assert first["mean_residual_closure"] == pytest.approx(0.5)
    assert first["share_closed_at_least_half"] == 1.0
    assert metrics["overall"]["share_moved_toward_target"] == pytest.approx(2 / 3)
    assert metrics["selection_eligible"] is True
