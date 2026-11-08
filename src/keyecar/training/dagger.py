"""DAgger training data: reviewed correction targets mixed with the human demonstrations.

Takeovers recorded during autonomous runs are turned into correction targets by
:mod:`keyecar.training.corrections` and reviewed by a human. Only sessions
accepted in the manifest, whose recording and sidecar hashes still match, are
used. Each sidecar entry is joined back to its raw frame by
``(session_id, frame_id)``, and :class:`KeyecarSessionDataset` supplies the
frame history, conditions, motor state, and augmentation.

Every batch has a fixed mix of three sources (default 36 / 20 / 8):

* ``seed``: the human demonstrations;
* ``exact_human``: frames the human drove during a takeover;
* ``pseudo_window``: model-driven frames just before and after a takeover,
  with a steering-only target (see :mod:`keyecar.training.corrections`).

Correction frames are drawn takeover-first (a random takeover, then a random
frame of it), so a single long takeover cannot dominate the batch.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .corrections import load_intervention_window_sidecar, validate_manifest_artifacts
from .dataset import AugmentConfig, KeyecarSessionDataset
from .files import sha256_file

SEED_SOURCE = "seed"
EXACT_SOURCE = "exact_human"
PSEUDO_SOURCE = "pseudo_window"
TRAINING_SOURCES = (SEED_SOURCE, EXACT_SOURCE, PSEUDO_SOURCE)


class DaggerDataError(RuntimeError):
    """Raised when the accepted correction data cannot form training samples."""


@dataclass(frozen=True)
class AcceptedCorrectionSession:
    session_id: str
    session_path: Path
    sidecar_path: Path
    split: str
    source_hashes: dict[str, str]
    sidecar_sha256: str

    def to_metadata(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "session_path": str(self.session_path),
            "sidecar_path": str(self.sidecar_path),
            "split": self.split,
            "source_hashes": dict(sorted(self.source_hashes.items())),
            "sidecar_sha256": self.sidecar_sha256,
        }


def load_accepted_correction_sessions(
    manifest_path: Path | str, *, split: str = "train"
) -> tuple[list[AcceptedCorrectionSession], dict[str, object]]:
    """Validate the manifest and return the accepted sessions of ``split`` plus provenance."""
    manifest_path = Path(manifest_path)
    manifest = validate_manifest_artifacts(manifest_path)
    accepted = sorted(
        (
            AcceptedCorrectionSession(
                session_id=entry["session_id"],
                session_path=Path(entry["session_path"]),
                sidecar_path=Path(entry["sidecar_path"]),
                split=entry["split"],
                source_hashes=dict(entry["accepted_source_hashes"]),
                sidecar_sha256=entry["accepted_sidecar_sha256"],
            )
            for entry in manifest["sessions"]
            if entry["review_status"] == "accepted" and entry["split"] == split
        ),
        key=lambda session: session.session_id,
    )
    if not accepted:
        raise DaggerDataError(f"No accepted correction sessions for split={split!r} in {manifest_path}")
    provenance = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "review_status_required": "accepted",
        "split": split,
        "sessions": [session.to_metadata() for session in accepted],
    }
    return accepted, provenance


@dataclass(frozen=True)
class CorrectionTarget:
    """One included sidecar entry, joined to its row in the base dataset."""

    base_index: int
    intervention_key: str  # "<session_id>:<intervention_id>"
    region: str  # "pre", "human", or "post"
    throttle: float
    steering: float
    supervision: tuple[float, float]
    weights: tuple[float, float]
    # Steering the model actually commanded (or its last command before the
    # takeover); the baseline for the residual-closure metric.
    baseline_steering: float


def _correction_target(entry: dict, base_index: int) -> CorrectionTarget:
    baseline = entry.get("recorded_model_steering")
    if baseline is None:
        baseline = entry.get("model_anchor_steering")
    targets, supervision, weights = entry["targets"], entry["supervision"], entry["loss_weights"]
    return CorrectionTarget(
        base_index=base_index,
        intervention_key=f"{entry['session_id']}:{entry['intervention_id']}",
        region=entry["region"],
        throttle=0.0 if targets["throttle"] is None else float(targets["throttle"]),
        steering=float(targets["steering"]),
        supervision=(float(supervision["throttle"]), float(supervision["steering"])),
        weights=(float(weights["throttle"]), float(weights["steering"])),
        baseline_steering=float("nan") if baseline is None else float(baseline),
    )


class CorrectionWindowDataset(Dataset):
    """The exact-human or pseudo-window targets of the accepted correction sessions."""

    def __init__(
        self,
        sessions: Sequence[AcceptedCorrectionSession],
        *,
        source: str,
        image_size: int = 160,
        history_length: int = 3,
        history_max_gap_s: float = 0.2,
        augment: bool = True,
        augment_config: AugmentConfig | None = None,
        seed: int | None = None,
    ):
        if source not in (EXACT_SOURCE, PSEUDO_SOURCE):
            raise ValueError(f"source must be {EXACT_SOURCE!r} or {PSEUDO_SOURCE!r}")
        if not sessions:
            raise DaggerDataError("At least one accepted correction session is required")
        self.source = source
        self.base = KeyecarSessionDataset(
            [session.session_path for session in sessions],
            image_size=image_size,
            history_length=history_length,
            history_max_gap_s=history_max_gap_s,
            augment=augment,
            augment_config=augment_config,
            filter_targets=False,
            seed=seed,
        )
        base_index = {
            (session_id, frame_id): index
            for index, (session_id, frame_id) in enumerate(
                zip(self.base.df["session_id"], self.base.df["frame_id"], strict=True)
            )
        }

        entries = []
        self._sidecar_summaries = []
        for session in sessions:
            sidecar = load_intervention_window_sidecar(session.sidecar_path)
            self._sidecar_summaries.append(
                {
                    "session_id": session.session_id,
                    "summary": sidecar["summary"],
                    "generation": sidecar["generation"],
                }
            )
            entries += [
                entry
                for entry in sidecar["entries"]
                if entry["included"] and (entry["region"] == "human") == (source == EXACT_SOURCE)
            ]
        if not entries:
            raise DaggerDataError(f"No included {source} entries in the accepted sidecars")

        self.targets: list[CorrectionTarget] = []
        for entry in sorted(entries, key=lambda e: e["entry_id"]):
            key = (entry["session_id"], entry["frame_id"])
            if key not in base_index:
                raise DaggerDataError(f"Sidecar entry has no recorded frame: {key}")
            self.targets.append(_correction_target(entry, base_index[key]))

        groups: dict[str, list[int]] = defaultdict(list)
        for index, target in enumerate(self.targets):
            groups[target.intervention_key].append(index)
        self.intervention_groups = {key: tuple(indices) for key, indices in sorted(groups.items())}

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict[str, object]:
        target = self.targets[index]
        sample = self.base.get(target.base_index, target=(target.throttle, target.steering))
        sample["supervision_mask"] = torch.tensor(target.supervision, dtype=torch.float32)
        sample["output_weights"] = torch.tensor(target.weights, dtype=torch.float32)
        return sample

    def stats(self) -> dict[str, object]:
        return {
            "source": self.source,
            "num_samples": len(self),
            "num_interventions": len(self.intervention_groups),
            "samples_per_intervention": {
                key: len(indices) for key, indices in self.intervention_groups.items()
            },
            "sidecars": self._sidecar_summaries,
        }


class DaggerMixedDataset(Dataset):
    """The three training sources behind one integer index: seed, then exact, then pseudo."""

    def __init__(self, seed: Dataset, exact: CorrectionWindowDataset, pseudo: CorrectionWindowDataset):
        self.datasets = {SEED_SOURCE: seed, EXACT_SOURCE: exact, PSEUDO_SOURCE: pseudo}
        self.offsets: dict[str, int] = {}
        offset = 0
        for source in TRAINING_SOURCES:
            if len(self.datasets[source]) == 0:
                raise DaggerDataError(f"Training source {source!r} is empty")
            self.offsets[source] = offset
            offset += len(self.datasets[source])
        self._length = offset

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> dict[str, object]:
        if not 0 <= index < self._length:
            raise IndexError(index)
        source = TRAINING_SOURCES[bisect_right(list(self.offsets.values()), index) - 1]
        return self.datasets[source][index - self.offsets[source]]

    def global_intervention_groups(self, source: str) -> dict[str, tuple[int, ...]]:
        """The takeover groups of a correction source, as indices into this dataset."""
        offset = self.offsets[source]
        groups = self.datasets[source].intervention_groups
        return {key: tuple(offset + index for index in indices) for key, indices in groups.items()}

    @property
    def source_counts(self) -> dict[str, int]:
        return {source: len(self.datasets[source]) for source in TRAINING_SOURCES}


class DaggerBatchSampler(Sampler[list[int]]):
    """Yields batches with a fixed number of samples from each source.

    Seed samples go through a fresh permutation of the demonstrations (one pass
    per epoch by default). Correction samples pick a random takeover first and
    then a random frame of it.
    """

    def __init__(
        self,
        dataset: DaggerMixedDataset,
        *,
        batch_counts: Mapping[str, int],
        steps_per_epoch: int | None = None,
        seed: int = 42,
    ):
        if set(batch_counts) != set(TRAINING_SOURCES) or min(batch_counts.values()) <= 0:
            raise ValueError(f"batch_counts needs a positive count for each of {TRAINING_SOURCES}")
        self.dataset = dataset
        self.batch_counts = {source: int(count) for source, count in batch_counts.items()}
        if steps_per_epoch is None:
            steps_per_epoch = math.ceil(len(dataset.datasets[SEED_SOURCE]) / self.batch_counts[SEED_SOURCE])
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.epoch = 0
        self._groups = {
            source: dataset.global_intervention_groups(source) for source in (EXACT_SOURCE, PSEUDO_SOURCE)
        }

    @property
    def batch_size(self) -> int:
        return sum(self.batch_counts.values())

    def __len__(self) -> int:
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _seed_indices(self, rng: np.random.Generator, count: int) -> list[int]:
        offset = self.dataset.offsets[SEED_SOURCE]
        population = len(self.dataset.datasets[SEED_SOURCE])
        indices: list[int] = []
        while len(indices) < count:
            indices.extend((offset + rng.permutation(population)).tolist())
        return indices[:count]

    def _correction_indices(self, source: str, rng: np.random.Generator, count: int) -> list[int]:
        groups = self._groups[source]
        keys = sorted(groups)
        indices = []
        for _ in range(count):
            frames = groups[keys[int(rng.integers(0, len(keys)))]]
            indices.append(frames[int(rng.integers(0, len(frames)))])
        return indices

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        seed_count = self.batch_counts[SEED_SOURCE]
        seed_indices = self._seed_indices(rng, self.steps_per_epoch * seed_count)
        for step in range(self.steps_per_epoch):
            batch = seed_indices[step * seed_count : (step + 1) * seed_count]
            for source in (EXACT_SOURCE, PSEUDO_SOURCE):
                batch.extend(self._correction_indices(source, rng, self.batch_counts[source]))
            rng.shuffle(batch)
            yield batch


def correction_exclusion_counts(sessions: Sequence[AcceptedCorrectionSession]) -> dict[str, int]:
    """Sum the per-reason counts of excluded entries over all sidecars."""
    counts: Counter[str] = Counter()
    for session in sessions:
        counts.update(load_intervention_window_sidecar(session.sidecar_path)["summary"]["exclusion_counts"])
    return dict(sorted(counts.items()))
