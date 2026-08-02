import pytest

from face_symmetry_lab.domain.models import AnalysisState, ReasonCode, Status
from face_symmetry_lab.ui.guidance import guidance_message


def test_guidance_message_is_none_when_the_gate_is_satisfied() -> None:
    state = AnalysisState(status=Status.OK, reason=None, reason_codes=())

    assert guidance_message(state) is None


@pytest.mark.parametrize("reason", list(ReasonCode))
def test_every_reason_code_has_a_japanese_guidance_message(reason: ReasonCode) -> None:
    state = AnalysisState(status=Status.HOLD, reason=reason, reason_codes=(reason,))

    message = guidance_message(state)

    assert message is not None
    assert message != ""


def test_guidance_message_is_actionable_for_common_reasons() -> None:
    too_far = AnalysisState(
        status=Status.HOLD, reason=ReasonCode.TOO_FAR, reason_codes=(ReasonCode.TOO_FAR,)
    )
    too_close = AnalysisState(
        status=Status.HOLD, reason=ReasonCode.TOO_CLOSE, reason_codes=(ReasonCode.TOO_CLOSE,)
    )

    assert guidance_message(too_far) != guidance_message(too_close)
