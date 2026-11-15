"""DAgger correction targets built from a synthetic recording with takeovers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from keyecar.training.corrections import (
    CorrectionDataError,
    WindowConfig,
    build_intervention_window_payload,
)

FRAME_NS = 50_000_000  # 20 Hz
COLUMNS = [
    "schema_version",
    "session_id",
    "frame_id",
    "frame_index",
    "timestamp_monotonic_ns",
    "image_path",
    "condition",
    "condition_name",
    "human_throttle",
    "human_steering",
    "model_steering",
    "training_eligible",
    "intervention_id",
    "stop_pressed",
    "safety_emergency_stop",
]


def model_frame(steering: float = 0.0, **extra: object) -> dict[str, object]:
    return {"model_steering": steering, **extra}


def human_frame(intervention_id: str, steering: float, **extra: object) -> dict[str, object]:
    return {
        "intervention_id": intervention_id,
        "human_throttle": 0.5,
        "human_steering": steering,
        "training_eligible": 1,
        **extra,
    }


def write_session(root: Path, frames: list[dict[str, object]]) -> Path:
    """Write a schema-v2 recording with one row per frame, 50 ms apart."""
    session_dir = root / "session"
    (session_dir / "images").mkdir(parents=True)
    (session_dir / "metadata.json").write_text(json.dumps({"schema_version": 2, "session_id": "session"}))
    with (session_dir / "records.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        for index, frame in enumerate(frames):
            image = f"images/{index:06d}.jpg"
            (session_dir / image).write_bytes(b"frame %d" % index)
            writer.writerow(
                {
                    "schema_version": 2,
                    "session_id": "session",
                    "frame_id": f"f{index:06d}",
                    "frame_index": index,
                    "timestamp_monotonic_ns": index * FRAME_NS,
                    "image_path": image,
                    "condition": 0,
                    "condition_name": "FORWARD",
                    **frame,
                }
            )
    return session_dir


def entries_by_frame(payload: dict[str, object], region: str) -> dict[int, dict[str, object]]:
    return {e["frame_index"]: e for e in payload["entries"] if e["region"] == region}


@pytest.fixture
def single_takeover(tmp_path: Path) -> dict[str, object]:
    """Model steers straight, the human steers 0.6 for frames 15-19, then the model continues."""
    frames = [model_frame(0.0) for _ in range(15)]
    frames += [human_frame("takeover-1", 0.6) for _ in range(5)]
    frames += [model_frame(0.0) for _ in range(10)]
    return build_intervention_window_payload(write_session(tmp_path, frames))


def test_human_frames_become_exact_targets(single_takeover):
    human = entries_by_frame(single_takeover, "human")
    assert sorted(human) == [15, 16, 17, 18, 19]
    for entry in human.values():
        assert entry["included"]
        assert entry["targets"] == {"throttle": 0.5, "steering": 0.6}
        assert entry["loss_weights"] == {"throttle": 1.0, "steering": 1.0}


def test_alpha_ramps_up_before_the_takeover(single_takeover):
    pre = entries_by_frame(single_takeover, "pre")
    assert sorted(pre) == list(range(5, 15))  # the 0.5 s before the takeover
    alphas = [pre[i]["alpha"] for i in sorted(pre)]
    assert alphas[0] == pytest.approx(0.25)  # pre_alpha_start
    assert alphas[-1] == pytest.approx(0.925)
    assert alphas == sorted(alphas) and len(set(alphas)) == len(alphas)


def test_alpha_ramps_down_after_the_takeover(single_takeover):
    post = entries_by_frame(single_takeover, "post")
    assert sorted(post) == [20, 21, 22, 23, 24]  # the 0.25 s after the takeover
    assert [post[i]["alpha"] for i in sorted(post)] == pytest.approx([0.8, 0.6, 0.4, 0.2, 0.0])


def test_pseudo_targets_move_the_model_steering_towards_the_human(single_takeover):
    assert single_takeover["interventions"][0]["correction_delta"] == pytest.approx(0.6)
    pseudo = {**entries_by_frame(single_takeover, "pre"), **entries_by_frame(single_takeover, "post")}
    for entry in pseudo.values():
        alpha = entry["alpha"]
        assert entry["targets"]["steering"] == pytest.approx(0.0 + alpha * 0.6)
        assert entry["targets"]["throttle"] is None  # steering only
        assert entry["supervision"] == {"throttle": False, "steering": True}
        assert entry["loss_weights"] == {"throttle": 0.0, "steering": pytest.approx(alpha)}


def test_summary_counts(single_takeover):
    summary = single_takeover["summary"]
    assert summary["num_interventions"] == 1
    assert summary["num_usable_interventions"] == 1
    assert summary["num_exact_human_entries"] == 5
    assert summary["num_pseudo_entries"] == 15
    assert summary["num_excluded_entries"] == 0


def test_payload_is_deterministic(tmp_path: Path):
    frames = [model_frame(0.1) for _ in range(12)] + [human_frame("t", -0.5) for _ in range(3)]
    session_dir = write_session(tmp_path, frames + [model_frame(0.1) for _ in range(6)])
    assert build_intervention_window_payload(session_dir) == build_intervention_window_payload(session_dir)


def test_small_corrections_produce_no_pseudo_targets(tmp_path: Path):
    frames = [model_frame(0.0) for _ in range(12)]
    frames += [human_frame("nudge", 0.1) for _ in range(4)]  # below min_abs_correction_delta = 0.15
    frames += [model_frame(0.0) for _ in range(6)]
    payload = build_intervention_window_payload(write_session(tmp_path, frames))

    assert payload["interventions"][0]["status"] == "correction_below_threshold"
    assert payload["summary"]["num_exact_human_entries"] == 4
    assert payload["summary"]["num_pseudo_entries"] == 0
    pseudo = [e for e in payload["entries"] if e["region"] != "human"]
    assert pseudo and all(e["exclusion_reason"] == "correction_below_threshold" for e in pseudo)


def test_pseudo_targets_are_clamped(tmp_path: Path):
    # The model was already at 0.9 early in the window; adding the correction would exceed 1.
    frames = [model_frame(0.9) for _ in range(12)] + [model_frame(0.0)]
    frames += [human_frame("hard-left", 0.8) for _ in range(4)]
    frames += [model_frame(0.9) for _ in range(6)]
    payload = build_intervention_window_payload(write_session(tmp_path, frames))

    targets = [e["targets"]["steering"] for e in payload["entries"] if e["region"] != "human"]
    assert max(targets) == 1.0
    assert all(-1.0 <= target <= 1.0 for target in targets)


def test_stop_and_safety_frames_are_never_targets(tmp_path: Path):
    frames = [model_frame(0.0) for _ in range(10)]
    frames += [model_frame(0.0, stop_pressed=1), model_frame(0.0)]
    frames += [human_frame("t", 0.6, safety_emergency_stop=1)]
    frames += [human_frame("t", 0.6) for _ in range(3)]
    frames += [model_frame(0.0) for _ in range(6)]
    payload = build_intervention_window_payload(write_session(tmp_path, frames))

    reasons = {e["frame_index"]: e["exclusion_reason"] for e in payload["entries"] if not e["included"]}
    assert reasons == {10: "pseudo_stop_or_safety_frame", 12: "human_stop_or_safety_frame"}


def two_takeovers(tmp_path: Path, second_steering: float) -> dict[str, object]:
    """Two takeovers 0.2 s apart, so the windows around them overlap on frames 12-15."""
    frames = [model_frame(0.0) for _ in range(10)]
    frames += [human_frame("first", 0.6) for _ in range(2)]
    frames += [model_frame(0.0) for _ in range(4)]
    frames += [human_frame("second", second_steering) for _ in range(2)]
    frames += [model_frame(0.0) for _ in range(8)]
    return build_intervention_window_payload(write_session(tmp_path, frames))


def included_per_frame(payload: dict[str, object]) -> dict[int, list[dict[str, object]]]:
    included: dict[int, list[dict[str, object]]] = {}
    for entry in payload["entries"]:
        if entry["included"]:
            included.setdefault(entry["frame_index"], []).append(entry)
    return included


def test_overlapping_windows_keep_one_target_per_frame(tmp_path: Path):
    payload = two_takeovers(tmp_path, second_steering=0.6)
    included = included_per_frame(payload)
    assert all(len(entries) == 1 for entries in included.values())

    # Human labels win over pseudo targets of the other takeover.
    assert all(included[i][0]["region"] == "human" for i in (10, 11, 16, 17))
    # Between the takeovers, the window with the higher alpha wins.
    for index in (12, 13, 14, 15):
        candidates = [e for e in payload["entries"] if e["frame_index"] == index and e["region"] != "human"]
        assert len(candidates) == 2
        assert included[index][0]["alpha"] == max(e["alpha"] for e in candidates)
        loser = next(e for e in candidates if not e["included"])
        assert loser["exclusion_reason"] == "lower_alpha_overlap"


def test_overlapping_windows_with_opposite_corrections_are_dropped(tmp_path: Path):
    payload = two_takeovers(tmp_path, second_steering=-0.6)
    included = included_per_frame(payload)
    for index in (12, 13, 14, 15):
        assert index not in included
    reasons = {e["exclusion_reason"] for e in payload["entries"] if e["frame_index"] in (12, 13, 14, 15)}
    assert reasons == {"conflicting_correction_direction"}


def test_takeover_split_into_two_blocks_is_rejected(tmp_path: Path):
    frames = [model_frame(0.0) for _ in range(5)]
    frames += [human_frame("t", 0.6), model_frame(0.0), human_frame("t", 0.6)]
    with pytest.raises(CorrectionDataError, match="multiple blocks"):
        build_intervention_window_payload(write_session(tmp_path, frames))


@pytest.mark.parametrize(
    "config",
    [
        WindowConfig(pre_window_s=0.0),
        WindowConfig(post_window_s=-1.0),
        WindowConfig(min_abs_correction_delta=-0.1),
        WindowConfig(pre_alpha_start=1.5),
    ],
)
def test_invalid_window_config_is_rejected(config: WindowConfig):
    with pytest.raises(CorrectionDataError):
        config.validate()
