"""`ui.image_section`: アップロードした写真 1 枚の分析セクション。

`ui.app` から分けた入力モードに対応する。ページ構成（同意・入力方法の選択）は
`test_app.py` が引き続き受け持つ。
"""

from datetime import UTC, datetime
from pathlib import Path

from streamlit.testing.v1 import AppTest

import face_symmetry_lab.ui.image_section as image_section
from face_symmetry_lab.domain.demographics import build_demographic_estimate


def _fake_demographic_estimate(model_version: str = "test-demographic/1.0.0"):
    return build_demographic_estimate(
        estimated_age_years=30.0,
        gender_probabilities=(0.3, 0.7),
        model_version=model_version,
        model_sha256="a" * 64,
        observed_at=datetime(2026, 8, 11, tzinfo=UTC),
    )


# --- FR-034 対称化パネル -----------------------------------------------------


def test_the_image_symmetrization_panel_reports_an_unusable_frame() -> None:
    """FR-034: 折り返せないフレームは、落ちるのではなく理由を説明する。"""
    at = AppTest.from_string(
        "import numpy as np\n"
        "import streamlit as st\n"
        "from face_symmetry_lab.application.analyzer import AnalysisOutcome\n"
        "from face_symmetry_lab.application.still_image import (\n"
        "    StillImageAnalysis,\n"
        "    StillImageLandmarks,\n"
        ")\n"
        "from face_symmetry_lab.domain.landmarks import build_landmark_frame\n"
        "from face_symmetry_lab.domain.models import AnalysisState, Status\n"
        "from face_symmetry_lab.ui.image_section import _render_image_symmetrization\n"
        "rgb = np.zeros((20, 20, 3), dtype=np.uint8)\n"
        "frame = build_landmark_frame(frame_id=1, face_count=0, width=20, height=20)\n"
        "_render_image_symmetrization(\n"
        "    rgb,\n"
        "    StillImageAnalysis(\n"
        "        landmarks=StillImageLandmarks(frame=frame, preview=rgb),\n"
        "        outcome=AnalysisOutcome(\n"
        "            frame_id=1,\n"
        "            state=AnalysisState(status=Status.NO_FACE),\n"
        "            symmetry=None,\n"
        "            overall_distance_smooth=None,\n"
        "            pose=None,\n"
        "            face_area_ratio=None,\n"
        "            metric_version='metric/test',\n"
        "            landmark_map_version='map/test',\n"
        "        ),\n"
        "    ),\n"
        ")\n"
    )
    at.run()

    assert not at.exception
    assert at.info  # 黙って空白にせず、理由を説明する


def test_the_image_symmetrization_panel_avoids_value_laden_wording() -> None:
    """§5.3/§10.7: 対称化した顔は構成物であって、規範ではない。"""
    import inspect

    source = inspect.getsource(image_section._render_image_symmetrization)

    for forbidden in ("理想", "正解", "補正後", "あるべき", "完璧", "欠点"):
        assert forbidden not in source


# --- H2: 画像アップロードの年齢・性別自動実行 --------------------------------

_DEMOGRAPHIC_TRIGGER_SCRIPT = (
    "import numpy as np\n"
    "from face_symmetry_lab.ui.image_section import (\n"
    "    _maybe_estimate_image_demographics,\n"
    "    _render_image_demographic_detail,\n"
    ")\n"
    "age_input = np.zeros((62, 62, 3), dtype=np.uint8)\n"
    "_maybe_estimate_image_demographics(\n"
    "    age_input, input_fingerprint='fp-1', recorder=None,\n"
    ")\n"
    "_render_image_demographic_detail(age_input)\n"
)


def test_single_face_image_shows_demographics_without_a_button(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """依頼 1・AC-AGE-03: 単一顔が検出できたら、ボタン操作なしに参考年齢・
    参考性別が描画される。"""
    onnx_path = tmp_path / "fake.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    monkeypatch.setattr(
        image_section, "resolve_ready_demographic", lambda: (onnx_path, "test-demographic/1.0.0")
    )

    class _FakeEstimator:
        def __init__(self, *, onnx_path: Path, model_version: str) -> None:
            del onnx_path
            self._model_version = model_version

        def estimate(self, rgb_image):
            del rgb_image
            return _fake_demographic_estimate(self._model_version)

    monkeypatch.setattr(image_section, "OpenCvDemographicEstimator", _FakeEstimator)

    at = AppTest.from_string(_DEMOGRAPHIC_TRIGGER_SCRIPT)
    at.run()

    assert not at.exception
    assert not at.button  # ボタンが存在しない
    labels = [metric.label for metric in at.metric]
    assert "参考年齢" in labels
    assert "外見由来の参考性別" in labels


def test_the_estimator_runs_once_per_image_even_after_reruns(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """依頼 1・AC-AGE-03: 同じ画像に対して estimator は 1 度しか呼ばれない
    （罠 (a): fingerprint によるキャッシュを外すと、再実行のたびに増える）。
    異なる画像（fingerprint）に対しては新たに 1 度だけ呼ばれる。"""
    onnx_path = tmp_path / "fake.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    monkeypatch.setattr(
        image_section, "resolve_ready_demographic", lambda: (onnx_path, "test-demographic/1.0.0")
    )

    calls = {"n": 0}

    class _CountingEstimator:
        def __init__(self, *, onnx_path: Path, model_version: str) -> None:
            del onnx_path
            self._model_version = model_version

        def estimate(self, rgb_image):
            del rgb_image
            calls["n"] += 1
            return _fake_demographic_estimate(self._model_version)

    monkeypatch.setattr(image_section, "OpenCvDemographicEstimator", _CountingEstimator)

    at = AppTest.from_string(
        "import numpy as np\n"
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.image_section import _maybe_estimate_image_demographics\n"
        "age_input = np.zeros((62, 62, 3), dtype=np.uint8)\n"
        "fingerprint = st.session_state.get('test_fingerprint', 'fp-1')\n"
        "_maybe_estimate_image_demographics(\n"
        "    age_input, input_fingerprint=fingerprint, recorder=None,\n"
        ")\n"
    )
    at.run()
    assert calls["n"] == 1

    # 同じ画像のまま再実行（他ウィジェットの操作を模す）しても増えない。
    at.run()
    assert not at.exception
    assert calls["n"] == 1

    # 新しい画像（別 fingerprint）は新たに 1 度だけ呼ばれる。
    at.session_state["test_fingerprint"] = "fp-2"
    at.run()
    assert not at.exception
    assert calls["n"] == 2


def test_a_failed_estimate_does_not_stop_the_symmetry_panel_from_rendering(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """依頼 1・不変条件 12・FR-024: estimator が例外を投げても、左右差側の
    パネルは描画され続け、年齢・性別区画だけに理由が示される。"""
    onnx_path = tmp_path / "fake.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    monkeypatch.setattr(
        image_section, "resolve_ready_demographic", lambda: (onnx_path, "test-demographic/1.0.0")
    )

    class _FailingEstimator:
        def __init__(self, *, onnx_path: Path, model_version: str) -> None:
            del onnx_path, model_version

        def estimate(self, rgb_image):
            del rgb_image
            raise RuntimeError("boom")

    monkeypatch.setattr(image_section, "OpenCvDemographicEstimator", _FailingEstimator)

    at = AppTest.from_string(
        "import numpy as np\n"
        "from face_symmetry_lab.ui.image_section import (\n"
        "    _maybe_estimate_image_demographics,\n"
        "    _render_image_demographic_detail,\n"
        ")\n"
        "from face_symmetry_lab.ui.still_overlay import render_image_with_overlay\n"
        "age_input = np.zeros((62, 62, 3), dtype=np.uint8)\n"
        "_maybe_estimate_image_demographics(\n"
        "    age_input, input_fingerprint='fp-1', recorder=None,\n"
        ")\n"
        # 左右差側は年齢・性別の後に描画される。ここで例外を伝播させないことが
        # 不変条件 12 の核心である。2026-08-12 以降、Image の左右差を画面へ出す
        # のは画像上の吹き出しなので、見出し結果ではなくそれで確認する。
        "render_image_with_overlay(\n"
        "    np.zeros((8, 8, 3), dtype=np.uint8),\n"
        "    caption='検出したランドマーク',\n"
        "    overall=0.02,\n"
        "    equal_weight=0.05,\n"
        "    parts={'contour': 0.01, 'eyes': 0.02, 'nose': 0.03, 'mouth': 0.015},\n"
        "    estimate=None,\n"
        "    show_demographics=False,\n"
        "    gate='#2E9E5B',\n"
        ")\n"
        "_render_image_demographic_detail(age_input)\n"
    )
    at.run()

    assert not at.exception
    overlay = " ".join(element.value for element in at.markdown)
    assert "98.00%" in overlay
    assert any("推定できませんでした" in element.value for element in at.error)


def test_no_face_detected_skips_inference_and_explains_why(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """依頼 1・AC-AGE-03: 顔が検出できない（`age_input is None`）画像では
    estimator を一切呼ばず、理由を説明する。"""
    onnx_path = tmp_path / "fake.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    monkeypatch.setattr(
        image_section, "resolve_ready_demographic", lambda: (onnx_path, "test-demographic/1.0.0")
    )

    calls = {"n": 0}

    class _CountingEstimator:
        def __init__(self, *, onnx_path: Path, model_version: str) -> None:
            del onnx_path, model_version

        def estimate(self, rgb_image):
            del rgb_image
            calls["n"] += 1
            return _fake_demographic_estimate()

    monkeypatch.setattr(image_section, "OpenCvDemographicEstimator", _CountingEstimator)

    at = AppTest.from_string(
        "from face_symmetry_lab.ui.image_section import (\n"
        "    _maybe_estimate_image_demographics,\n"
        "    _render_image_demographic_detail,\n"
        ")\n"
        "_maybe_estimate_image_demographics(\n"
        "    None, input_fingerprint='fp-1', recorder=None,\n"
        ")\n"
        "_render_image_demographic_detail(None)\n"
    )
    at.run()

    assert not at.exception
    assert calls["n"] == 0
    assert any("単一顔" in element.value for element in at.info)


def test_the_demographic_detail_panel_no_longer_mentions_a_button() -> None:
    """依頼 1: ボタンを廃止したので、案内文がボタン操作を前提にしてはならない。"""
    import inspect

    source = inspect.getsource(image_section._render_image_demographic_detail)

    assert "ボタン" not in source


def test_a_missing_demographic_model_explains_itself_without_calling_the_estimator(
    monkeypatch,
) -> None:
    """止まって報告すべきことの確認: モデル未配置でも自動実行の表示は壊れず、
    estimator を呼ばずに理由を説明する（FR-024）。"""
    monkeypatch.setattr(image_section, "resolve_ready_demographic", lambda: None)

    calls = {"n": 0}

    class _UnreachableEstimator:
        def __init__(self, *, onnx_path: Path, model_version: str) -> None:
            del onnx_path, model_version

        def estimate(self, rgb_image):
            del rgb_image
            calls["n"] += 1
            return _fake_demographic_estimate()

    monkeypatch.setattr(image_section, "OpenCvDemographicEstimator", _UnreachableEstimator)

    at = AppTest.from_string(_DEMOGRAPHIC_TRIGGER_SCRIPT)
    at.run()

    assert not at.exception
    assert calls["n"] == 0
    assert any("未配置" in element.value for element in at.info)


# --- §5.1 ページ構成との境界 -------------------------------------------------


# --- FR-038: 参考年齢の表示ゲート ---------------------------------------------


def test_gate_blocking_hides_the_demographic_detail_and_shows_the_notice(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """FR-038/AC-AGEGATE-02: ゲートが範囲外と判定したら、成功した推定でも隠す。"""
    onnx_path = tmp_path / "fake.onnx"
    onnx_path.write_bytes(b"fake onnx bytes")
    monkeypatch.setattr(
        image_section, "resolve_ready_demographic", lambda: (onnx_path, "test-demographic/1.0.0")
    )

    class _FakeEstimator:
        def __init__(self, *, onnx_path: Path, model_version: str) -> None:
            del onnx_path, model_version

        def estimate(self, rgb_image):
            del rgb_image
            return _fake_demographic_estimate()

    monkeypatch.setattr(image_section, "OpenCvDemographicEstimator", _FakeEstimator)

    at = AppTest.from_string(
        "import numpy as np\n"
        "from face_symmetry_lab.application.age_gate import AgeGateOutcome, AgeGateStatus\n"
        "from face_symmetry_lab.ui.image_section import (\n"
        "    _maybe_estimate_image_demographics,\n"
        "    _render_image_demographic_detail,\n"
        ")\n"
        "age_input = np.zeros((62, 62, 3), dtype=np.uint8)\n"
        "_maybe_estimate_image_demographics(\n"
        "    age_input, input_fingerprint='fp-1', recorder=None,\n"
        ")\n"
        "gate_outcome = AgeGateOutcome(\n"
        "    status=AgeGateStatus.OUT_OF_RANGE, decision=None, reason=None\n"
        ")\n"
        "_render_image_demographic_detail(age_input, gate_outcome)\n"
    )
    at.run()

    assert not at.exception
    assert not at.metric  # 推定は成功しているが、参考年齢・参考性別を出さない
    assert any("参考年齢" in element.value for element in at.info)


def test_gate_blocking_hides_the_bubble_estimate() -> None:
    """FR-038: 吹き出しにも数値を出さず、ゲートの固定文言だけを出す。"""
    at = AppTest.from_string(
        "import numpy as np\n"
        "import streamlit as st\n"
        "from face_symmetry_lab.application.age_gate import AgeGateOutcome, AgeGateStatus\n"
        "from face_symmetry_lab.application.analyzer import AnalysisOutcome\n"
        "from face_symmetry_lab.application.still_image import (\n"
        "    StillImageAnalysis,\n"
        "    StillImageLandmarks,\n"
        ")\n"
        "from face_symmetry_lab.domain.demographics import build_demographic_estimate\n"
        "from face_symmetry_lab.domain.landmarks import build_landmark_frame\n"
        "from face_symmetry_lab.domain.models import AnalysisState, Status\n"
        "from face_symmetry_lab.ui.image_section import (\n"
        "    RESULT_STATE_KEY,\n"
        "    _render_preview_with_overlay,\n"
        ")\n"
        "from datetime import UTC, datetime\n"
        "st.session_state[RESULT_STATE_KEY] = build_demographic_estimate(\n"
        "    estimated_age_years=30.0, gender_probabilities=(0.3, 0.7),\n"
        "    model_version='test/1.0.0', model_sha256='a' * 64,\n"
        "    observed_at=datetime(2026, 8, 11, tzinfo=UTC),\n"
        ")\n"
        "rgb = np.zeros((20, 20, 3), dtype=np.uint8)\n"
        "frame = build_landmark_frame(frame_id=1, face_count=1, width=20, height=20)\n"
        "analysis = StillImageAnalysis(\n"
        "    landmarks=StillImageLandmarks(frame=frame, preview=rgb),\n"
        "    outcome=AnalysisOutcome(\n"
        "        frame_id=1,\n"
        "        state=AnalysisState(status=Status.OK),\n"
        "        symmetry=None,\n"
        "        overall_distance_smooth=None,\n"
        "        pose=None,\n"
        "        face_area_ratio=None,\n"
        "        metric_version='metric/test',\n"
        "        landmark_map_version='map/test',\n"
        "    ),\n"
        ")\n"
        "gate_outcome = AgeGateOutcome(\n"
        "    status=AgeGateStatus.UNDETERMINED, decision=None, reason=None\n"
        ")\n"
        "_render_preview_with_overlay(analysis, gate_outcome)\n"
    )
    at.run()

    assert not at.exception
    markup = " ".join(element.value for element in at.markdown)
    assert "参考年齢 30 歳" not in markup
    assert "参考年齢・参考性別を表示できません" in markup


# --- FR-039: 参照平均の併記 ----------------------------------------------------


_BUBBLE_SETUP = (
    "import numpy as np\n"
    "import streamlit as st\n"
    "from datetime import UTC, datetime\n"
    "from face_symmetry_lab.application.age_gate import AgeGateOutcome, AgeGateStatus\n"
    "from face_symmetry_lab.application.analyzer import AnalysisOutcome\n"
    "from face_symmetry_lab.application.still_image import (\n"
    "    StillImageAnalysis,\n"
    "    StillImageLandmarks,\n"
    ")\n"
    "from face_symmetry_lab.domain.demographics import build_demographic_estimate\n"
    "from face_symmetry_lab.domain.landmarks import build_landmark_frame\n"
    "from face_symmetry_lab.domain.models import AnalysisState, Status\n"
    "from face_symmetry_lab.ui.image_section import (\n"
    "    RESULT_STATE_KEY,\n"
    "    _render_preview_with_overlay,\n"
    ")\n"
    "st.session_state[RESULT_STATE_KEY] = build_demographic_estimate(\n"
    "    estimated_age_years=30.0, gender_probabilities=(0.3, 0.7),\n"
    "    model_version='test/1.0.0', model_sha256='a' * 64,\n"
    "    observed_at=datetime(2026, 8, 11, tzinfo=UTC),\n"
    ")\n"
    "rgb = np.zeros((20, 20, 3), dtype=np.uint8)\n"
    "frame = build_landmark_frame(frame_id=1, face_count=1, width=20, height=20)\n"
    "analysis = StillImageAnalysis(\n"
    "    landmarks=StillImageLandmarks(frame=frame, preview=rgb),\n"
    "    outcome=AnalysisOutcome(\n"
    "        frame_id=1,\n"
    "        state=AnalysisState(status=Status.OK),\n"
    "        symmetry=None,\n"
    "        overall_distance_smooth=None,\n"
    "        pose=None,\n"
    "        face_area_ratio=None,\n"
    "        metric_version='metric/test',\n"
    "        landmark_map_version='map/test',\n"
    "    ),\n"
    ")\n"
)


def test_the_bubble_carries_the_baseline_for_the_shown_estimate() -> None:
    """FR-039/AC-BASELINE-01: 同梱の参照統計を通した、Image 側の端から端までの配線。

    30 歳・男性は `configs/reference/` の全人種を合わせて 283 件である。
    """
    at = AppTest.from_string(_BUBBLE_SETUP + "_render_preview_with_overlay(analysis, None)\n")
    at.run()

    assert not at.exception
    markup = " ".join(element.value for element in at.markdown)
    assert "参考年齢 30 歳" in markup
    assert "30 歳・男性" in markup
    assert "283 件" in markup
    assert "人種で絞り込まず" in markup


def test_the_bubble_drops_the_baseline_when_the_gate_hides_the_age() -> None:
    """§10.12-3: 母集団の年齢から、ゲートが隠した判断を復元させない。

    罠: 吹き出しへ渡す推定とは別に session state から母集団を引く実装は、
    参考年齢を隠したまま「30 歳・男性の平均」を出してしまう。
    """
    at = AppTest.from_string(
        _BUBBLE_SETUP + "_render_preview_with_overlay(\n"
        "    analysis,\n"
        "    AgeGateOutcome(status=AgeGateStatus.OUT_OF_RANGE, decision=None, reason=None),\n"
        ")\n"
    )
    at.run()

    assert not at.exception
    markup = " ".join(element.value for element in at.markdown)
    assert "平均 " not in markup
    assert "30 歳" not in markup
    assert "参考年齢・参考性別を表示できません" in markup


def test_the_page_module_no_longer_owns_the_image_mode() -> None:
    """H4: `ui.app` はページの骨格だけを持ち、Image モードの中身は持たない。

    罠: 便宜的に片方の関数だけ `ui.app` へ戻す変更は、ページ構成と入力モードの
    境界を曖昧にする（Camera 側と非対称になる）。
    """
    import inspect

    from face_symmetry_lab.ui import app

    source = inspect.getsource(app)
    for moved in ("file_uploader", "analyze_still_image", "OpenCvDemographicEstimator"):
        assert moved not in source
    assert "render_image_section" in source
