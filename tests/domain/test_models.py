import pytest
from pydantic import ValidationError

from face_symmetry_lab.domain.models import (
    AnalysisResult,
    AnalysisState,
    ReasonCode,
    Status,
    SymmetryDistance,
)


# 状態とエラー理由
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


# 顔のパーツごとの左右差
def test_symmetry_distance_initialization():
    """左右差（正規化RMS距離）の初期化が正しく行えるかのテスト"""
    # 輪郭、目、鼻、口、全体の距離（浮動小数点数 = float）を渡して箱を作る
    distance = SymmetryDistance(contour=0.021, eyes=0.008, nose=0.014, mouth=0.033, overall=0.019)

    # 箱の中身（メンバ変数）に正しくデータが入っているか確認
    assert distance.contour == 0.021
    assert distance.eyes == 0.008
    assert distance.nose == 0.014
    assert distance.mouth == 0.033
    assert distance.overall == 0.019


def test_a_distance_has_no_upper_bound():
    """§8.6: 距離は 0 が下限で、0〜100 のような上限を持たない。"""
    distance = SymmetryDistance(contour=250.0, eyes=1.5, nose=0.0, mouth=3.0, overall=120.0)

    assert distance.contour == 250.0
    assert distance.overall == 120.0


@pytest.mark.parametrize("field", ["contour", "eyes", "nose", "mouth", "overall"])
def test_a_negative_distance_is_rejected(field: str):
    """NFR-013: 負の距離は幾何的にあり得ないので UI・出力へ流さない。"""
    values = dict.fromkeys(("contour", "eyes", "nose", "mouth", "overall"), 0.01)
    values[field] = -0.001

    with pytest.raises(ValidationError):
        SymmetryDistance(**values)


def test_analysis_result_initialization():
    """総合的な分析結果の箱が正しく作れるかのテスト"""

    # 準備：状態と距離のデータを作る
    state = AnalysisState(status=Status.OK)
    distance = SymmetryDistance(contour=0.021, eyes=0.008, nose=0.014, mouth=0.033, overall=0.019)

    # テスト対象：総合パッケージ(AnalysisResult)を作る
    result = AnalysisResult(state=state, distance=distance)

    # 確認：正しくデータが格納されているか
    assert result.state.status == Status.OK
    assert result.distance.overall == 0.019
    # テスト対象2：顔が検出できなかった場合（距離は計算できないので None になるはず）
    error_state = AnalysisState(status=Status.HOLD, reason=ReasonCode.NO_FACE)
    error_result = AnalysisResult(state=error_state, distance=None)

    assert error_result.state.status == Status.HOLD
    assert error_result.distance is None
