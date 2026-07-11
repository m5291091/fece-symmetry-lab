import pytest
from face_symmetry_lab.domain.models import AnalysisState, Status, ReasonCode

def test_analysis_state_initialization():
    """分析状態の初期化が正しく行えるかのテスト"""
    
    # OK状態（理由なし）
    state_ok = AnalysisState(status=Status.OK)
    assert state_ok.status == Status.OK
    assert state_ok.reason is None

    # HOLD状態（顔が検出されないなどの理由あり）
    state_hold = AnalysisState(status=Status.HOLD, reason=ReasonCode.NO_FACE)
    assert state_hold.status == Status.HOLD
    assert state_hold.reason == ReasonCode.NO_FACE