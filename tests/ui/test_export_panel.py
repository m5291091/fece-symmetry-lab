"""FR-017 / FR-020: Camera と Image が共有する CSV / JSON パネル（`ui.export_panel`）。

このパネルの契約は 2 つある。**記録があるときだけ download button を出す**
（FR-017）ことと、そのとき版と config fingerprint を必ず添える（FR-020）ことで
ある。

`empty_notice` は前者を緩めるものではない。記録が無いときに **download button の
代わりに** 一行の説明を出すだけの引数である。これが無かった間、Camera の
「エクスポート」タブは開始直後に見出しごと何も描かず、タブの中身が真っ白に
なっていた——利用者から見れば「壊れている」と「まだ何も無い」を区別できない。
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from face_symmetry_lab.application.quality import QualityThresholds
from face_symmetry_lab.application.session import SessionRecorder
from face_symmetry_lab.domain.metrics import PartMap, build_metric_config

_EMPTY_NOTICE = "まだ書き出せる記録がありません。"


def _metric_config(*, metric_version: str):
    return build_metric_config(
        metric_version=metric_version,
        landmark_map_version="map/test",
        landmark_count=16,
        left_eye_corners=(0, 1),
        right_eye_corners=(2, 3),
        midline_indices=(4, 5, 6),
        parts={
            "contour": PartMap(pairs=((7, 8),), min_pairs=1),
            "eyes": PartMap(pairs=((0, 3), (1, 2)), min_pairs=1),
            "nose": PartMap(pairs=((9, 10),), min_pairs=1),
            "mouth": PartMap(pairs=((11, 12),), min_pairs=1),
        },
    )


def _recorder() -> SessionRecorder:
    return SessionRecorder(
        metric_config=_metric_config(metric_version="symmetry-2d/test"),
        metric_config_3d=_metric_config(metric_version="symmetry-3d/test"),
        thresholds=QualityThresholds(),
        tau_seconds=0.4,
        landmark_model_version="landmark/1",
        landmark_model_sha256="a" * 64,
        app_version="0.1.0",
        lock_digest="b" * 64,
    )


def recorded_bundle():
    """AppTest のスクリプトから読む helper（`tests.ui.test_reference_page` と同じ作法）。"""
    recorder = _recorder()
    recorder.append_error(frame_id=4, elapsed_ms=25, processing_ms=3.4)
    return recorder.export_bundle()


def empty_bundle():
    """記録を 1 件も持たないセッション（カメラを開始した直後の状態）。"""
    return _recorder().export_bundle()


def missing_bundle():
    """recorder がまだ存在しないとき（`RealtimeLandmarkPipeline.export_bundle` の `None`）。"""
    return None


def _run(builder: str, *, notice: str | None = _EMPTY_NOTICE) -> AppTest:
    notice_arg = "None" if notice is None else repr(notice)
    script = (
        f"from tests.ui.test_export_panel import {builder}\n"
        "from face_symmetry_lab.ui.export_panel import render_export_panel\n"
        "render_export_panel(\n"
        f"    {builder}(),\n"
        "    heading='セッションデータのエクスポート',\n"
        "    warning='顔由来の派生データを含みます。',\n"
        "    key_prefix='download_test',\n"
        f"    empty_notice={notice_arg},\n"
        ")\n"
    )
    at = AppTest.from_string(script)
    at.run()
    return at


def _all_text(at: AppTest) -> str:
    parts: list[str] = []
    for block in (at.markdown, at.caption, at.info, at.warning, at.subheader):
        parts.extend(str(element.value) for element in block)
    return " ".join(parts)


def test_a_recorded_session_offers_csv_and_json_with_its_provenance() -> None:
    """FR-017 / FR-020: 版と config fingerprint を添えて 2 つだけ出す。"""
    at = _run("recorded_bundle")

    assert not at.exception
    labels = [element.label for element in at.download_button]
    assert labels == ["CSV を保存", "JSON を保存"]
    text = _all_text(at)
    assert "セッションデータのエクスポート" in text
    assert "symmetry-3d/test" in text, "FR-020: 指標の版を添える"


def test_the_notice_never_replaces_a_real_export() -> None:
    """罠: 「常に notice を出す」実装は、記録があるここで落ちる。"""
    at = _run("recorded_bundle")

    assert not at.exception
    assert _EMPTY_NOTICE not in _all_text(at)


def test_an_empty_session_explains_itself_instead_of_leaving_the_tab_blank() -> None:
    """記録がまだ無いことは事実であり、真っ白な区画で伝えてはならない。"""
    at = _run("empty_bundle")

    assert not at.exception
    assert _EMPTY_NOTICE in _all_text(at)


def test_an_empty_session_still_offers_no_download() -> None:
    """FR-017: 説明を足しても、記録が無い間は download button を出さない。"""
    at = _run("empty_bundle")

    assert not at.exception
    assert list(at.download_button) == []


def test_a_missing_recorder_is_treated_like_an_empty_session() -> None:
    """`export_bundle()` が `None` を返す段階（recorder 未接続）も同じ扱いにする。"""
    at = _run("missing_bundle")

    assert not at.exception
    assert _EMPTY_NOTICE in _all_text(at)
    assert list(at.download_button) == []


def test_without_a_notice_the_panel_stays_silent() -> None:
    """既定は従来どおり「何も描かない」。呼び出し側が文言を持つ設計を保つ。"""
    at = _run("empty_bundle", notice=None)

    assert not at.exception
    assert _all_text(at) == ""
    assert list(at.download_button) == []


# --- FR-042: 結果レポート（HTML） ---------------------------------------------


def _report_script(*, uncheck: bool = False) -> str:
    """レポート builder を渡したときのパネル。builder は引数を記録するだけ。"""
    toggle = "st.session_state['download_test_report_figures'] = False\n" if uncheck else ""
    return (
        "import streamlit as st\n"
        "from tests.ui.test_export_panel import recorded_bundle\n"
        "from face_symmetry_lab.ui.export_panel import render_export_panel\n"
        f"{toggle}"
        "def builder(include_figures):\n"
        "    st.session_state['builder_calls'] = "
        "st.session_state.get('builder_calls', []) + [include_figures]\n"
        "    return '<!doctype html><html><body>report</body></html>'\n"
        "render_export_panel(\n"
        "    recorded_bundle(),\n"
        "    heading='セッションデータのエクスポート',\n"
        "    warning='顔由来の派生データを含みます。',\n"
        "    key_prefix='download_test',\n"
        "    report_builder=builder,\n"
        "    report_mode='session',\n"
        ")\n"
    )


def _run_report(*, uncheck: bool = False) -> AppTest:
    at = AppTest.from_string(_report_script(uncheck=uncheck))
    at.run()
    return at


def test_a_report_builder_adds_a_third_button_without_removing_the_ledger() -> None:
    """FR-042: CSV / JSON は据え置き、レポートが 3 つ目として増える。"""
    at = _run_report()

    assert not at.exception
    labels = [element.label for element in at.download_button]
    assert "CSV を保存" in labels
    assert "JSON を保存" in labels
    assert len(labels) == 3


def test_the_report_button_writes_an_html_file() -> None:
    at = _run_report()

    report = next(
        element
        for element in at.download_button
        if element.label not in ("CSV を保存", "JSON を保存")
    )
    assert report.label.endswith("を保存")
    assert "HTML" in report.label or "レポート" in report.label


def test_the_figure_toggle_defaults_to_on() -> None:
    """§15.1: 既定 ON（2026-08-16 の判断、ADR-0009 (d)）。"""
    at = _run_report()

    assert len(at.checkbox) == 1
    assert at.checkbox[0].value is True
    assert at.session_state["builder_calls"] == [True]


def test_turning_the_toggle_off_reaches_the_builder() -> None:
    """罠: チェックを読まずに常に図付きで作る実装は、OFF にしても顔写真を出す。"""
    at = _run_report(uncheck=True)

    assert at.session_state["builder_calls"] == [False]


def test_the_sharing_warning_is_shown_before_the_download() -> None:
    """FR-042: 押した後では遅い。保存する人へは操作の直前で伝える。"""
    from face_symmetry_lab.ui.report import REPORT_SHARING_NOTE

    text = _all_text(_run_report())

    assert REPORT_SHARING_NOTE in text


def test_the_warning_is_dropped_when_no_figure_is_embedded() -> None:
    """図を切ったレポートに顔写真は入らない。出ない注意を出し続けない。"""
    from face_symmetry_lab.ui.report import REPORT_SHARING_NOTE

    text = _all_text(_run_report(uncheck=True))

    assert REPORT_SHARING_NOTE not in text
