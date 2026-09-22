from __future__ import annotations

import enum
import pathlib
import typing

import attrs

from .config import Contest, Suite
from .lockstep import LockstepSummary


@attrs.frozen
class Step:
    contestant: Contest.Contestant
    stage: Suite.Stage
    episodes: int = 1
    record_dir: pathlib.Path | None = None
    is_reference: bool = False
    reference_type: str | None = None

    @property
    def key(self) -> str:
        if self.is_reference and self.reference_type:
            if self.contestant.name.endswith(self.reference_type):
                return f"{self.contestant.name}/{self.stage.name}"
            return f"{self.contestant.name}_{self.reference_type}/{self.stage.name}"
        return f"{self.contestant.name}/{self.stage.name}"


class StepErrorKind(enum.StrEnum):
    ENV_SETUP = "env_setup"
    ROBOT_SETUP = "robot_setup"
    EPISODE_TIMEOUT = "episode_timeout"
    SIM_DEAD = "sim_dead"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


@attrs.frozen
class StepResult:
    key: str
    status: typing.Literal["ok", "failed", "skipped", "partial", "in_progress"]
    env_id: int | None
    started_at: float
    ended_at: float | None
    error_kind: StepErrorKind | None
    error_detail: str | None
    episodes_run: int = 0
    episodes_failed: int = 0
    episodes_weak: int = 0
    episodes_worst_progress: float | None = None
    episodes_total: int = 0
    sim_s: float = 0.0
    lockstep: LockstepSummary | None = None
