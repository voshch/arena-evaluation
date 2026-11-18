"""The closed ``null_reason`` vocabulary of the stored fact tables."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Final


class NullReason(StrEnum):
    """Why a metric cell is NULL; a missing measurement is never a zero."""

    TOPIC_MISSING = "topic_missing"
    PEDSIM_UNAVAILABLE = "pedsim_unavailable"
    NO_SAMPLES = "no_samples"
    REFERENCE_MISSING = "reference_missing"
    NO_CONFLICT = "no_conflict"
    ZERO_DENOMINATOR = "zero_denominator"
    NOT_RECORDED = "not_recorded"
    NOT_APPLICABLE = "not_applicable"
    INVALID_EPISODE = "invalid_episode"
    EXCEPTION = "exception"


#: Codes written as ``code:parameter``.
NULL_REASON_PARAMETERISED: Final[frozenset[str]] = frozenset(
    {
        NullReason.TOPIC_MISSING.value,
        NullReason.INVALID_EPISODE.value,
        NullReason.EXCEPTION.value,
    }
)

#: One-line meaning of every vocabulary member, the documentation the schema tool serves.
NULL_REASON_MEANINGS: Final[Mapping[str, str]] = {
    f"{NullReason.TOPIC_MISSING.value}:<topic>": "the recording carries no <topic> stream, so the value was never measured",
    NullReason.PEDSIM_UNAVAILABLE.value: "the scenario spawned no dynamic pedestrians",
    NullReason.NO_SAMPLES.value: "the stream was recorded but holds no sample inside the window",
    NullReason.REFERENCE_MISSING.value: "no robot-free recorded reference matched this stage and seed",
    NullReason.NO_CONFLICT.value: "clearance stayed above the encounter distance, so nothing happened",
    NullReason.ZERO_DENOMINATOR.value: "the ratio's denominator is zero inside this window",
    NullReason.NOT_RECORDED.value: "the recording does not carry this parameter or value",
    NullReason.NOT_APPLICABLE.value: "the quantity has no meaning for this row's kind",
    f"{NullReason.INVALID_EPISODE.value}:<why>": "the episode is not rankable; <why> names the defect",
    f"{NullReason.EXCEPTION.value}:<Class>": "the calculator raised <Class>",
}

#: ``<why>`` values of ``invalid_episode:<why>``, the defects that make an episode unrankable.
INVALID_EPISODE_REASONS: Final[Mapping[str, str]] = {
    "pose_teleport": "the ground-truth pose jumps faster than any robot can move, so the trajectory is not physical",
    "incomplete_recording": "the recording has no terminal outcome or reports lost sections",
    "simulator_error": "the simulator ended the episode with FATAL",
    "missing_stream:<topic>": "a stream the metric needs was not recorded",
    "reference_exceeds_path": "the reference geodesic is longer than the travelled path, so path efficiency is out of range",
}
