import pytest
from face_symmetry_lab.domain.models import AnalysisState, Status, ReasonCode, SymmetryScore, AnalysisResult

#状態とエラー理由
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

# 顔のパーツごとの点数
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

def test_analysis_result_initialization():
    """総合的な分析結果の箱が正しく作れるかのテスト"""
    
    # 準備：状態とスコアのデータを作る
    state = AnalysisState(status=Status.OK)
    score = SymmetryScore(contour=85.5, eyes=90.0, nose=95.2, mouth=88.1, overall=89.7)
    
    # テスト対象：総合パッケージ(AnalysisResult)を作る
    result = AnalysisResult(state=state, score=score)
    
    # 確認：正しくデータが格納されているか
    assert result.state.status == Status.OK
    assert result.score.overall == 89.7
    # テスト対象2：顔が検出できなかった場合（スコアは計算できないので None になるはず）
    error_state = AnalysisState(status=Status.HOLD, reason=ReasonCode.NO_FACE)
    error_result = AnalysisResult(state=error_state, score=None)
    
    assert error_result.state.status == Status.HOLD
    assert error_result.score is None


