from __future__ import annotations

import array
import csv
import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import subprocess
import typing

import yaml
from rclpy.parameter import Parameter

from .lockstep import BEAT_PREFIXES, LockstepSummary
from .step import StepErrorKind, StepResult


def compute_config_hash(suite: object, contest: object) -> str:
    blob = json.dumps([suite, contest], sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def find_most_recent_resumable(data_root: pathlib.Path) -> str | None:
    """Return the run_id of the most recent run with at least one non-ok step,
    or None if no resumable run exists."""
    if not data_root.is_dir():
        return None
    candidates: list[str] = []
    for child in data_root.iterdir():
        if not child.is_dir():
            continue
        manifest_path = child / "manifest.yaml"
        state_path = child / ".benchmark_state.json"
        if not manifest_path.exists():
            continue
        try:
            Manifest.from_yaml(manifest_path.read_text())
        except Exception:
            continue
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text())
                steps = state.get("steps") or {}
            except Exception:
                steps = {}
        else:
            steps = {}
        if steps and all(v.get("status") == "ok" for v in steps.values()):
            continue
        candidates.append(child.name)
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0]


def capture_git_sha(workspace: pathlib.Path) -> tuple[str | None, bool]:
    try:
        sha = subprocess.run(
            ["git", "-C", str(workspace), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if sha.returncode != 0:
            return None, False
        dirty_out = subprocess.run(
            ["git", "-C", str(workspace), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return sha.stdout.strip(), bool(dirty_out.stdout.strip())
    except Exception:
        return None, False


@dataclasses.dataclass
class Manifest:
    run_id: str
    created_at: str
    arena_git_sha: str | None
    arena_git_dirty: bool
    cli_args: list[str]
    env_n: int
    headless: bool
    config_hash: str
    simulator: str | None
    scale_episodes: float
    suite_name: str
    contest_name: str
    suite: dict
    contest: list | dict
    steps: list[dict]
    launch_args: dict = dataclasses.field(default_factory=dict)
    suite_provenance: dict | None = None
    contest_provenance: dict | None = None

    def to_yaml(self) -> str:
        return yaml.dump(dataclasses.asdict(self), allow_unicode=True, sort_keys=False)

    @classmethod
    def from_yaml(cls, text: str) -> Manifest:
        return cls(**yaml.safe_load(text))


class StateFile:
    _STATE_FILENAME = ".benchmark_state.json"

    def __init__(self, path: pathlib.Path, steps: dict[str, StepResult]) -> None:
        self.path = path
        self.steps = steps

    @classmethod
    def open(cls, path: pathlib.Path) -> StateFile:
        state_path = path / cls._STATE_FILENAME
        if state_path.exists():
            data = json.loads(state_path.read_text())
            steps: dict[str, StepResult] = {}
            for key, val in data.get("steps", {}).items():
                raw_kind = val.get("error_kind")
                error_kind = StepErrorKind(raw_kind) if raw_kind is not None else None
                # Backward-compat: old state files stored a single "error" string.
                error_detail = val.get("error_detail") or val.get("error")
                steps[key] = StepResult(
                    key=key,
                    status=val["status"],
                    env_id=val.get("env_id"),
                    started_at=val["started_at"],
                    ended_at=val.get("ended_at"),
                    error_kind=error_kind,
                    error_detail=error_detail,
                    episodes_run=val.get("episodes_run", 0),
                    episodes_failed=val.get("episodes_failed", 0),
                    lockstep=LockstepSummary.from_dict(val["lockstep"]) if val.get("lockstep") else None,
                )
            return cls(path, steps)
        return cls(path, {})

    def write(self, steps: typing.Mapping[str, StepResult]) -> None:
        data = {
            "steps": {
                k: {
                    "status": v.status,
                    "env_id": v.env_id,
                    "started_at": v.started_at,
                    "ended_at": v.ended_at,
                    "error_kind": v.error_kind.value if v.error_kind is not None else None,
                    "error_detail": v.error_detail,
                    "episodes_run": v.episodes_run,
                    "episodes_failed": v.episodes_failed,
                    "lockstep": v.lockstep.to_dict() if v.lockstep is not None else None,
                }
                for k, v in steps.items()
            }
        }
        tmp = self.path / ".benchmark_state.json.tmp"
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, self.path / self._STATE_FILENAME)
        self.steps = dict(steps)


def _params_to_json(params: list) -> str:
    rows = []
    for p in params:
        try:
            value = Parameter.from_parameter_msg(p).value
            if isinstance(value, array.array):
                value = value.tolist()
            elif isinstance(value, (bytes, bytearray)):
                value = list(value)
        except Exception:
            value = str(p.value)
        rows.append({"name": p.name, "value": value})
    return json.dumps(rows)


class ProgressLog:
    _HEADER = (
        "ts_iso,run_id,step_key,contestant,stage,env_id,episode_id,parent_episode_id,"
        "is_reference,reference_type,"
        "world,seed,tm_robots,tm_obstacles,tm_modules,robots,"
        "outcome_state,outcome_info,started_at,ended_at,runtime_s,"
        "robots_params_json,obstacles_params_json,"
        "error_kind,error_detail,"
        "lockstep_stalls,lockstep_max_stall_s,lockstep_rtf,lockstep_beats,"
        "goal_dist_start,goal_dist_min,path_length"
    )

    def __init__(self, path: pathlib.Path) -> None:
        self._path = path
        is_empty = not path.exists() or path.stat().st_size == 0
        self._fh = path.open("a", newline="")
        if is_empty:
            self._fh.write(self._HEADER + "\n")
            self._fh.flush()
        self._writer = csv.writer(self._fh)

    def append(
        self,
        *,
        ts_iso: str,
        run_id: str,
        step_key: str,
        contestant: str,
        stage: str,
        env_id: int | None,
        episode_id: int,
        episode_record: object,
        started_at: float,
        ended_at: float,
        parent_episode_id: int | None = None,
        is_reference: bool = False,
        reference_type: str | None = None,
        error_kind: StepErrorKind | None = None,
        error_detail: str | None = None,
        lockstep: LockstepSummary | None = None,
    ) -> None:
        rec = episode_record
        runtime = round(ended_at - started_at, 3)
        self._writer.writerow(
            [
                ts_iso,
                run_id,
                step_key,
                contestant,
                stage,
                env_id if env_id is not None else "",
                episode_id,
                parent_episode_id if parent_episode_id is not None else "",
                "true" if is_reference else "false",
                reference_type or "",
                rec.world,
                rec.seed,
                rec.tm_robots,
                rec.tm_obstacles,
                ",".join(rec.tm_modules),
                ",".join(rec.robots),
                rec.outcome_state,
                rec.outcome_info,
                started_at,
                ended_at,
                runtime,
                _params_to_json(rec.robots_params),
                _params_to_json(rec.obstacles_params),
                error_kind.value if error_kind is not None else "",
                error_detail or "",
                lockstep.stalls if lockstep is not None else "",
                round(lockstep.max_stall_s, 3) if lockstep is not None else "",
                round(lockstep.rtf, 3) if lockstep is not None else "",
                ",".join(ch for ch in lockstep.channels if ch.startswith(BEAT_PREFIXES)) if lockstep is not None else "",
                rec.goal_dist_start,
                rec.goal_dist_min,
                rec.path_length,
            ]
        )
        self._fh.flush()

    def write_comment(self, text: str) -> None:
        self._fh.write(f"# {text}\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def dedupe_in_place(self) -> None:
        """Remove duplicate (step_key, episode_id) rows, keeping the latest by ts_iso; drops old comment lines."""
        self._fh.flush()
        path = self._path

        raw_rows: list[dict] = []
        with path.open(newline="") as fh:
            for line in fh:
                stripped = line.rstrip("\n")
                if stripped.startswith("#") or not stripped:
                    continue
                raw_rows.append(stripped)

        if not raw_rows:
            return

        header_line = raw_rows[0]
        header = next(csv.reader([header_line]))
        data_lines = raw_rows[1:]

        step_key_idx = header.index("step_key")
        episode_id_idx = header.index("episode_id")
        ts_iso_idx = header.index("ts_iso")

        best: dict[tuple[str, str], list[str]] = {}
        for line in data_lines:
            row = next(csv.reader([line]))
            sk = row[step_key_idx]
            eid = row[episode_id_idx]
            ts = row[ts_iso_idx]
            key = (sk, eid)
            existing = best.get(key)
            if existing is None or ts > existing[ts_iso_idx]:
                best[key] = row

        deduped = sorted(best.values(), key=lambda r: r[ts_iso_idx])

        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as out:
            writer = csv.writer(out)
            out.write(header_line + "\n")
            writer.writerows(deduped)
        os.replace(tmp, path)


class RunDir:
    def __init__(
        self,
        path: pathlib.Path,
        manifest: Manifest,
        state: StateFile,
        progress: ProgressLog,
    ) -> None:
        self.path = path
        self.manifest = manifest
        self.state = state
        self.progress = progress

    @classmethod
    def create(cls, data_root: pathlib.Path, run_id: str, manifest: Manifest) -> RunDir:
        path = data_root / run_id
        path.mkdir(parents=True, exist_ok=False)
        manifest_path = path / "manifest.yaml"
        manifest_path.write_text(manifest.to_yaml())
        state = StateFile.open(path)
        progress = ProgressLog(path / "progress.csv")
        return cls(path, manifest, state, progress)

    @classmethod
    def open(cls, data_root: pathlib.Path, run_id: str) -> RunDir:
        path = data_root / run_id
        manifest_path = path / "manifest.yaml"
        manifest = Manifest.from_yaml(manifest_path.read_text())
        state = StateFile.open(path)
        progress = ProgressLog(path / "progress.csv")
        return cls(path, manifest, state, progress)

    def attach_log_handler(self, logger: logging.Logger) -> None:
        handler = logging.FileHandler(self.path / "runner.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
