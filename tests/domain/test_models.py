import pytest
from face_symmetry_lab.domain.models import AnalysisState, Status, ReasonCode, SymmetryScore

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

def test_symmetry_score_initialization():
    """対称性スコアの初期化が正しく行えるかのテスト"""
    # 輪郭、目、鼻、口、全体のスコア（浮動小数点数 = float）を渡して箱を作る
    score = SymmetryScore(
        contour=85.5,
        eyes=90.0,
        nose=95.2,
        mouth=88.1,
        overall=89.7
    )
    
    # 箱の中身（メンバ変数）に正しくデータが入っているか確認
    assert score.contour == 85.5
    assert score.eyes == 90.0
    assert score.nose == 95.2
    assert score.mouth == 88.1
    assert score.overall == 89.7