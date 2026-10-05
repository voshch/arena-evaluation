from __future__ import annotations

import array
import csv
import dataclasses
import fcntl
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


def _load_steps(state_path: pathlib.Path) -> dict[str, StepResult]:
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
    return steps


def _dump_steps(steps: typing.Mapping[str, StepResult]) -> str:
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
    return json.dumps(data, indent=2)


def _union_steps(path: pathlib.Path) -> dict[str, StepResult]:
    """Merge the run state with every lane state: an ok entry wins, otherwise the latest started_at."""
    merged: dict[str, StepResult] = {}
    sources = [path / StateFile._STATE_FILENAME, *sorted(path.glob(".benchmark_state.*.json"))]
    for state_path in sources:
        if not state_path.exists():
            continue
        for key, res in _load_steps(state_path).items():
            cur = merged.get(key)
            if cur is None or (res.status == "ok", res.started_at) > (cur.status == "ok", cur.started_at):
                merged[key] = res
    return merged


class StateFile:
    _STATE_FILENAME = ".benchmark_state.json"

    def __init__(self, path: pathlib.Path, steps: dict[str, StepResult], lane: str | None = None) -> None:
        self.path = path
        self.steps = steps
        self.lane = lane

    @property
    def filename(self) -> str:
        return self._STATE_FILENAME if self.lane is None else f".benchmark_state.{self.lane}.json"

    @classmethod
    def open(cls, path: pathlib.Path, lane: str | None = None) -> StateFile:
        return cls(path, _union_steps(path), lane)

    def write(self, steps: typing.Mapping[str, StepResult]) -> None:
        tmp = self.path / f"{self.filename}.tmp"
        tmp.write_text(_dump_steps(steps))
        os.replace(tmp, self.path / self.filename)
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

        raw_rows = [line.rstrip("\n") for line in _csv_lines(path)]

        if not raw_rows:
            return

        header_line = raw_rows[0]
        header = next(csv.reader([header_line]))
        best = _latest_rows(header, (next(csv.reader([line])) for line in raw_rows[1:]))
        ts_iso_idx = header.index("ts_iso")
        deduped = sorted(best.values(), key=lambda r: r[ts_iso_idx])

        tmp = path.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as out:
            writer = csv.writer(out)
            out.write(header_line + "\n")
            writer.writerows(deduped)
        os.replace(tmp, path)


def _csv_lines(path: pathlib.Path) -> list[str]:
    """Non-empty, non-comment lines of a progress csv, line endings kept."""
    with path.open(newline="") as fh:
        return [line for line in fh if line.rstrip("\n") and not line.startswith("#")]


def _latest_rows(header: list[str], rows: typing.Iterable[list[str]]) -> dict[tuple[str, str], list[str]]:
    """Rows keyed by (step_key, episode_id), keeping the latest ts_iso."""
    step_key_idx = header.index("step_key")
    episode_id_idx = header.index("episode_id")
    ts_iso_idx = header.index("ts_iso")
    best: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (row[step_key_idx], row[episode_id_idx])
        existing = best.get(key)
        if existing is None or row[ts_iso_idx] > existing[ts_iso_idx]:
            best[key] = row
    return best


def _fold_progress(path: pathlib.Path) -> None:
    sources = [p for p in (path / "progress.csv", *sorted(path.glob("progress.*.csv"))) if p.exists()]
    header: list[str] | None = None
    rows: list[list[str]] = []
    for source in sources:
        lines = _csv_lines(source)
        if lines and not lines[-1].endswith("\n"):
            lines.pop()
        records = list(csv.reader(lines))
        if not records:
            continue
        src_header = records[0]
        if header is None:
            header = src_header
        for row in records[1:]:
            if len(row) != len(src_header):
                continue
            by_name = dict(zip(src_header, row, strict=True))
            rows.append([by_name.get(col, "") for col in header])
    if header is None:
        header = next(csv.reader([ProgressLog._HEADER]))
    ts_iso_idx = header.index("ts_iso")
    folded = sorted(_latest_rows(header, rows).values(), key=lambda r: (r[ts_iso_idx], r))
    tmp = path / "progress.csv.fold.tmp"
    with tmp.open("w", newline="") as out:
        writer = csv.writer(out)
        out.write(",".join(header) + "\n")
        writer.writerows(folded)
    os.replace(tmp, path / "progress.csv")


def fold_run_dir(path: pathlib.Path) -> None:
    """Merge every lane state and progress file into .benchmark_state.json and progress.csv."""
    with (path / ".fold.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        steps = _union_steps(path)
        tmp = path / ".benchmark_state.json.fold.tmp"
        tmp.write_text(_dump_steps(dict(sorted(steps.items()))))
        os.replace(tmp, path / StateFile._STATE_FILENAME)
        _fold_progress(path)


class ManifestMismatchError(ValueError):
    """Raised when a lane joins a shared run dir created with a different config."""


class RunDir:
    def __init__(
        self,
        path: pathlib.Path,
        manifest: Manifest,
        state: StateFile,
        progress: ProgressLog,
        lane: str | None = None,
    ) -> None:
        self.path = path
        self.manifest = manifest
        self.state = state
        self.progress = progress
        self.lane = lane

    @property
    def log_path(self) -> pathlib.Path:
        return self.path / ("runner.log" if self.lane is None else f"runner.{self.lane}.log")

    @property
    def pid_path(self) -> pathlib.Path:
        return self.path / f"runner.{self.lane}.pid"

    @classmethod
    def create(cls, data_root: pathlib.Path, run_id: str, manifest: Manifest, lane: str | None = None) -> RunDir:
        path = data_root / run_id
        if lane is None:
            path.mkdir(parents=True, exist_ok=False)
            manifest_path = path / "manifest.yaml"
            manifest_path.write_text(manifest.to_yaml())
            state = StateFile.open(path)
            progress = ProgressLog(path / "progress.csv")
            return cls(path, manifest, state, progress)
        path.mkdir(parents=True, exist_ok=True)
        manifest_path = path / "manifest.yaml"
        tmp = path / f".manifest.{lane}.{os.getpid()}.tmp"
        tmp.write_text(manifest.to_yaml())
        try:
            os.link(tmp, manifest_path)
        except FileExistsError:
            existing = Manifest.from_yaml(manifest_path.read_text())
            if existing.config_hash != manifest.config_hash:
                raise ManifestMismatchError(f"run dir {path} was created with config_hash {existing.config_hash}, lane {lane!r} has {manifest.config_hash}") from None
            manifest = existing
        finally:
            tmp.unlink()
        return cls._open_lane(path, manifest, lane)

    @classmethod
    def open(cls, data_root: pathlib.Path, run_id: str, lane: str | None = None) -> RunDir:
        path = data_root / run_id
        manifest_path = path / "manifest.yaml"
        manifest = Manifest.from_yaml(manifest_path.read_text())
        if lane is not None:
            return cls._open_lane(path, manifest, lane)
        state = StateFile.open(path)
        progress = ProgressLog(path / "progress.csv")
        return cls(path, manifest, state, progress)

    @classmethod
    def _open_lane(cls, path: pathlib.Path, manifest: Manifest, lane: str) -> RunDir:
        fold_run_dir(path)
        state = StateFile.open(path, lane)
        progress = ProgressLog(path / f"progress.{lane}.csv")
        return cls(path, manifest, state, progress, lane)

    def attach_log_handler(self, logger: logging.Logger) -> None:
        handler = logging.FileHandler(self.log_path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)

    def fold(self) -> None:
        fold_run_dir(self.path)

    def claim(self, token: str, name: str) -> bool:
        """Atomically take claims/<token>/<name> for this lane, False when it already exists."""
        claim_dir = self.path / "claims" / token
        claim_dir.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(claim_dir / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as fh:
            fh.write(f"{self.lane}\n")
        return True

    def claimed(self, token: str, name: str) -> bool:
        return os.path.exists(self.path / "claims" / token / name)

    def reserve_episode_ids(self, count: int) -> int:
        """Reserve count consecutive episode ids across lanes, returning the first."""
        episodes_dir = self.path / "episodes"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        counter = episodes_dir / ".next_episode_id"
        with (episodes_dir / ".episode_id.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            first = 0
            for d in episodes_dir.glob("episode_*"):
                try:
                    first = max(first, int(d.name.split("_")[1]) + 1)
                except (IndexError, ValueError):
                    pass
            if counter.exists():
                first = max(first, int(counter.read_text().strip() or 0))
            tmp = episodes_dir / ".next_episode_id.tmp"
            tmp.write_text(f"{first + count}\n")
            os.replace(tmp, counter)
        return first
