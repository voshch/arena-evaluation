from arena_evaluation.processing.metrics.base import BaseMetricCalculator
from arena_evaluation.storage.schemas import AlignedEpisodeBundle


class _Calc(BaseMetricCalculator):
    NAME = "pose_probe"
    CATEGORY = "test"
    REQUIRED_TOPICS = ["odom"]

    @classmethod
    def output_keys(cls):
        return []

    def calculate(self, episode, prior_results):
        return {}


def _episode(data, start_pos):
    return AlignedEpisodeBundle(
        episode_id=1,
        data=data,
        start_pos=start_pos,
        goal_pos=[],
    )
