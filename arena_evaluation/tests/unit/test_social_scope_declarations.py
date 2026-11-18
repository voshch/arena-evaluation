import pytest

from arena_evaluation.processing.metrics.social.proxemics import ProxemicsCalculator
from arena_evaluation.processing.metrics.social.proxemics_extended import ProxemicsExtendedCalculator
from arena_evaluation.processing.metrics.social.social_forces import SocialForcesCalculator
from arena_evaluation.storage.registry import MetricScope


@pytest.mark.parametrize("calculator", [ProxemicsCalculator, ProxemicsExtendedCalculator, SocialForcesCalculator])
def test_no_social_output_declares_the_robot_scope(calculator) -> None:
    """No calculate_for_scope arm serves ROBOT for these calculators, so a ROBOT declaration could only publish NULL."""
    assert all(MetricScope.ROBOT not in output.scopes for output in calculator.OUTPUTS)
