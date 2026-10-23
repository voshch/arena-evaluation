from __future__ import annotations

from typing import Annotated, Any, Literal

import pydantic

from arena_mcp.core import LATCHED, ROS, ToolError, msgs, tg

from .base import domain

EnvId = Annotated[int | None, pydantic.Field(description="env id, the only env when omitted")]
NO_RECORDER = "the env records only with record.dir set and record.auto true, a benchmark always records"


async def _record(env_id: int | None, stop: bool, episode_id: int, outcome_state: int = 0, outcome_info: str = "") -> dict[str, Any]:
    from arena_evaluation_msgs.srv import RecordEpisode

    node = await tg(env_id)
    command = RecordEpisode.Request.COMMAND_STOP if stop else RecordEpisode.Request.COMMAND_START
    request = RecordEpisode.Request(command=command, episode_id=episode_id, outcome_state=outcome_state, outcome_info=outcome_info)
    response = await ROS.call(RecordEpisode, f"{node}/start_episode", request, hint=NO_RECORDER)
    if not response.success:
        raise ToolError(f"recorder refused to {'stop' if stop else 'start'} episode {episode_id}: {response.message}")
    return {"recorder": f"{node}/start_episode", "message": response.message}


@domain.tool(destructive=True)
async def evaluation_record_start(episode_id: Annotated[int, pydantic.Field(ge=0)], env_id: EnvId = None) -> dict[str, Any]:
    """Open a new episode recording on an env's recorder, closing the open one."""
    return await _record(env_id, False, episode_id)


@domain.tool(destructive=True)
async def evaluation_record_stop(
    episode_id: Annotated[int, pydantic.Field(ge=0)],
    outcome: Annotated[Literal["success", "failed", "skipped", "fatal"], pydantic.Field(description="terminal outcome stored in the episode metadata")],
    info: str = "",
    env_id: EnvId = None,
) -> dict[str, Any]:
    """Close an env's open episode recording with its outcome, flushing the MCAP and the episode metadata."""
    from task_generator_msgs.msg import EpisodeRecord

    states = {name: value for value, name in msgs.constants(EpisodeRecord).items()}
    return await _record(env_id, True, episode_id, outcome_state=states[outcome], outcome_info=info)


@domain.tool(read_only=True)
async def evaluation_recorded_topics(env_id: EnvId = None) -> dict[str, Any]:
    """The recordable topics an env's backend declares on <task generator>/state/recorded_topics, with QoS and whether each is recorded."""
    from task_generator_msgs.msg import RecordedTopics

    node = await tg(env_id)
    msg = await ROS.latest(RecordedTopics, f"{node}/state/recorded_topics", qos=LATCHED, hint="the task generator latches it once its backend is up")
    return {"task_generator": node, "topics": [msgs.to_dict(row) for row in msg.topics]}
