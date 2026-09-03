"""FR-042 / §9.6: 結果レポート（`ui.report`）。

レポートは**画面の文脈から切り離されて共有される**。したがってここで守るのは、
画面では周りが担っていた事柄が、ファイル 1 つの中に揃っていることである。

1. **注意書きが本体に入っている**（§5.3 の留保・§10.7 の禁止用途・共有の注意）。
2. **外部リソースを参照しない**（開いた先で通信を起こさない、AC-REPORT-02）。
3. **出してはいけないものが出ていない**（ランドマーク座標・ゲートの生確率・
   順位化の語・元ファイル名や EXIF、§9.1・§9.2・§9.3・§10.7）。
4. **モードで中身が変わる**——Camera は要約、Image は 1 枚分（§9.6.1 / §9.6.2）。
"""

from __future__ import annotations

import re

import pytest

from face_symmetry_lab.domain.models import ReasonCode
from face_symmetry_lab.domain.session_summary import (
    SessionSummary,
    SummaryReading,
    SummarySection,
    TrendSeries,
)
from face_symmetry_lab.ui.report import (
    DEMOGRAPHIC_HEADING,
    REPORT_SHARING_NOTE,
    REPORT_VERSION,
    ReportFigure,
    build_camera_report,
    build_image_report,
)
from tests.ui.conftest import _outcome_for_display, _symmetry_3d_result


def _reading(median: float = 98.12) -> SummaryReading:
    return SummaryReading(median=median, minimum=96.05, maximum=99.20, iqr=0.83, sample_count=120)


def _section(segment_id: str | None) -> SummarySection:
    return SummarySection(
        segment_id=segment_id,
        total_count=150,
        ok_count=120,
        valid_rate=0.8,
        valid_span_ms=42_000,
        symmetry=_reading(),
        parts=dict.fromkeys(("contour", "eyes", "nose", "mouth"), _reading()),
        reason_counts=((ReasonCode.OFF_AXIS_YAW, 20),),
        age_gate_evaluated=100,
        age_gate_shown=62,
    )


def _summary(*, segments: int = 2) -> SessionSummary:
    return SessionSummary(
        overall=_section(None),
        segments=tuple(_section(f"seg-{index}") for index in range(1, segments + 1)),
    )


class _Manifest:
    schema_version = "4.0.0"
    app_version = "0.1.0"
    lock_digest = "d" * 64
    session_id = "11111111-1111-4111-8111-111111111111"
    metric_version = "symmetry-3d/1.0.0"

    class _Model:
        def __init__(self, name: str) -> None:
            self.name = name
            self.version = f"{name}/1.0.0"
            self.sha256 = "a" * 64

    models = (_Model("face_landmarker"), _Model("age_gender"))

    class _Revision:
        revision = 1
        fingerprint = "c" * 64

    config_revisions = (_Revision(),)


class _Bundle:
    manifest = _Manifest()
    records: tuple[object, ...] = ()


def _outcome():
    return _outcome_for_display(
        symmetry_3d=_symmetry_3d_result(
            contour=0.010, eyes=0.020, nose=0.030, mouth=0.015, overall=0.018
        ),
        smooth_3d=0.021,
    )


def _camera(**kwargs) -> str:
    settings = {"summary": _summary(), "bundle": _Bundle()}
    settings.update(kwargs)
    return build_camera_report(**settings)


def _series(**kwargs) -> TrendSeries:
    settings = {
        "symmetry": (97.5, 98.0, 98.4, 98.1),
        "parts": {name: (97.0, 98.0) for name in ("contour", "eyes", "nose", "mouth")},
    }
    settings.update(kwargs)
    return TrendSeries(**settings)


def _image(**kwargs) -> str:
    settings = {"outcome": _outcome(), "bundle": _Bundle(), "figures": ()}
    settings.update(kwargs)
    return build_image_report(**settings)


def _figure() -> ReportFigure:
    return ReportFigure(caption="対称化オーバーレイ", data_uri="data:image/jpeg;base64,AAAA")


# --- 共通の骨格 ---------------------------------------------------------------


@pytest.mark.parametrize("build", [_camera, _image])
def test_the_report_is_a_complete_html_document(build) -> None:
    html = build()

    assert html.lstrip().startswith("<!doctype html>")
    assert "</html>" in html
    assert REPORT_VERSION in html


@pytest.mark.parametrize("build", [_camera, _image])
def test_the_report_never_reaches_outside_itself(build) -> None:
    """AC-REPORT-02: 開いた先で通信が起きるレポートを作らない。"""
    html = build()

    assert "http://" not in html
    assert "https://" not in html
    assert "<script" not in html.lower()
    assert "<link" not in html.lower()


@pytest.mark.parametrize("build", [_camera, _image])
def test_the_report_has_no_notes_section_and_no_provenance_table(build) -> None:
    """2026-08-16 の指示: 注意文の節と来歴の表をレポートから外した。

    FR-020 の再現性は CSV / JSON が担う——あちらは各行が版一式を自己完結して
    持っており（§9.4）、同じ情報を 2 つの成果物へ書き写す必要はない。
    """
    html = build()

    assert "この数値を再現するための情報" not in html
    assert "1 フレームごとの記録" not in html
    assert "About" not in html
    assert "探索値です。他の人との比較" not in html
    assert "c" * 64 not in html, "config fingerprint の表も無い"


def test_the_sharing_note_appears_only_where_a_photo_can_be() -> None:
    """§10.4 / ADR-0009 (d): 残す 1 行はここだけ。Camera は図を持たないので出ない。"""
    assert "顔写真" in REPORT_SHARING_NOTE
    assert REPORT_SHARING_NOTE in _image(figures=(_figure(),))
    assert REPORT_SHARING_NOTE not in _image(figures=())
    assert REPORT_SHARING_NOTE not in _camera()


def test_the_sharing_note_follows_the_figures() -> None:
    """§9.6: 図だけを切り出して共有されても、その 1 行が付いてくる位置に置く。"""
    html = _image(figures=(_figure(),))

    assert html.index("data:image/jpeg") < html.index(REPORT_SHARING_NOTE)


def test_a_report_without_figures_has_no_empty_figure_section() -> None:
    """AC-REPORT-03: 図を切ったときに空の枠を残さない。"""
    html = _image(figures=())

    assert "<img" not in html
    assert "<h2>画像</h2>" not in html


def test_the_camera_report_never_carries_a_photo() -> None:
    """2026-08-16: リアルタイムのレポートに顔写真は載せない。

    Camera 側は不変条件 1 の限定解除を必要としなくなる——顔写真がディスクへ
    出る経路は Image だけに閉じる。
    """
    html = _camera(series=_series())

    assert "<img" not in html
    assert "data:image" not in html


def test_figures_are_embedded_with_their_captions() -> None:
    html = _image(figures=(_figure(),))

    assert 'src="data:image/jpeg;base64,AAAA"' in html
    assert "対称化オーバーレイ" in html


@pytest.mark.parametrize("build", [_camera, _image])
def test_nothing_forbidden_reaches_the_file(build) -> None:
    """§9.1・§9.2・§9.3・§10.7 の「含めない」一覧。"""
    html = build()

    for banned in (
        "偏差値",
        "パーセンタイル",
        "合否",
        "平均より上",
        "平均より下",
        "landmark_coordinates",
        "probabilities",
        "EXIF",
    ):
        assert banned not in html
    assert "順位" not in html


@pytest.mark.parametrize("build", [_camera, _image])
def test_no_value_judgement_reaches_the_file(build) -> None:
    html = build()

    for banned in ("美しい", "良い", "悪い", "優れ", "理想", "劣", "完璧", "欠点"):
        assert banned not in html


def test_the_markup_escapes_whatever_it_is_given() -> None:
    """図の説明は呼び出し側の文字列である。生のまま埋めると markup が壊れる。"""
    html = _image(figures=(ReportFigure(caption="<b>x</b>&", data_uri="data:image/jpeg;base64,A"),))

    assert "<b>x</b>&" not in html
    assert "&lt;b&gt;x&lt;/b&gt;&amp;" in html


# --- Camera（§9.6.1） ---------------------------------------------------------


def test_the_camera_report_carries_the_summary_in_two_tiers() -> None:
    html = _camera()

    assert "98.12%" in html
    assert "80.0%" in html  # 有効率
    assert "区間" in html


def test_the_camera_report_lists_the_hold_reasons() -> None:
    assert "20" in _camera()


def test_the_camera_report_draws_every_reading_as_inline_svg() -> None:
    """§9.6.1（2026-08-16）: 1 本だけでは、どの部位が動いていたのか分からない。

    §0 項目 34（2026-09-03）で「全体のスコア」を撤回したので、5 系列とも同じ
    % の目盛りになり、1 つの図へ重ねる。
    """
    html = _camera(series=_series())

    assert html.count("<svg") == 1
    # 全体の対称度 + 部位別 4 つで 5 本。
    assert html.count("<polyline") == 5
    for label in ("全体の対称度", "輪郭", "目", "鼻", "口"):
        assert label in html


def test_a_chart_is_omitted_when_its_series_is_empty() -> None:
    """罠: 空の系列で軸だけの図を出す実装は、測れなかったことを図で嘘にする。"""
    assert "<svg" not in _camera(series=None)
    assert "<svg" not in _camera(series=_series(symmetry=(), parts={}))
    # 部位別だけが取れたセッションでも、描けるものは描く。
    assert _camera(series=_series(symmetry=())).count("<svg") == 1


def test_the_camera_report_does_not_carry_every_row() -> None:
    """§9.6.1: 3,000 行は読めない。台帳は CSV / JSON が持つ。

    案内文そのものは 2026-08-16 に削除したので、ここで見るのは「行が無いこと」
    だけである。列名が 1 つも現れなければ、行の羅列も無い。
    """
    html = _camera()

    assert "frame_id" not in html
    assert "elapsed_ms" not in html


def test_the_camera_report_states_why_segments_are_split() -> None:
    html = _camera()

    assert "設定が違う区間" in html


def test_a_single_segment_session_still_reports_the_whole() -> None:
    html = _camera(summary=_summary(segments=1))

    assert "98.12%" in html


def test_a_camera_report_without_any_record_is_refused() -> None:
    """記録が無いセッションのレポートは、書き出す内容そのものが無い。"""
    from face_symmetry_lab.ui.report import ReportError

    with pytest.raises(ReportError):
        _camera(summary=SessionSummary(overall=None, segments=()))


# --- Image（§9.6.2） ----------------------------------------------------------


def test_the_image_report_carries_the_five_readings() -> None:
    html = _image()

    assert "全体の対称度" in html
    for label in ("輪郭", "目", "鼻", "口"):
        assert label in html


def test_the_image_report_carries_the_pose() -> None:
    html = _image()

    assert "yaw" in html.lower() or "向き" in html


def test_the_image_report_has_no_session_summary() -> None:
    """FR-021: 1 行しか無く、分布が存在しない。"""
    html = _image()

    assert "中央値" not in html
    assert "変動幅" not in html


def test_a_missing_reading_is_a_dash_rather_than_zero() -> None:
    """不変条件 11。"""
    html = _image(outcome=_outcome_for_display(symmetry_3d=None, smooth_3d=None))

    assert "—" in html
    assert "0.00%" not in html


def test_the_image_report_omits_the_reference_age_when_it_is_not_shown() -> None:
    """FR-038/§10.11-2: 表示していない参考年齢をレポートへ出さない。"""
    html = _image(show_demographics=False)

    assert "参考年齢" not in html


def test_the_image_report_carries_the_gate_notice_instead_of_a_number() -> None:
    html = _image(show_demographics=False, notice="この画像では参考年齢を表示できません。")

    assert "この画像では参考年齢を表示できません。" in html
    assert "参考年齢 " not in html


def test_the_symmetrised_image_is_labelled_as_processed() -> None:
    """不変条件 8 / §5.3: 対称化画像は加工画像であると明記する。"""
    figure = ReportFigure(
        caption="左右対称にそろえた画像",
        data_uri="data:image/jpeg;base64,A",
        note="幾何変形した加工画像です。無加工の写真として提示しないでください。",
    )
    html = _image(figures=(figure,))

    assert "加工画像" in html


def test_the_file_name_carries_no_person_or_clock() -> None:
    from face_symmetry_lab.ui.report import report_filename

    name = report_filename(_Bundle(), mode="image")

    assert name.endswith(".html")
    assert "image" in name
    assert re.search(r"\d{4}-\d{2}-\d{2}", name) is None


# --- FR-039（撤回、2026-09-03）: 参照平均の併記 --------------------------------


def test_neither_report_mode_shows_a_reference_statistics_average() -> None:
    """FR-039 は撤回された（§0 項目 34）。Camera・Image のどちらのレポートにも
    「参照統計の平均」を出さない。

    罠: 参考年齢・参考性別の節（§9.6.1）まで巻き添えで消す実装は、この test の
    後半の assertion で落ちる——中央値・真ん中 50%・参考性別の確率は従来どおり
    出続けなければならない。
    """
    camera = _camera(overview=_overview())
    image = _image(estimate=_estimate(), show_demographics=True)

    for html in (camera, image):
        assert "参照統計の平均" not in html

    assert "31 歳" in camera
    assert "29〜34 歳" in camera
    assert DEMOGRAPHIC_HEADING in image


# --- §0 項目 34（撤回、2026-09-03）: 「全体のスコア」--------------------------


def test_neither_report_mode_shows_the_overall_score() -> None:
    """§8.6.7 の表示スケールと H4 の求め方の開示を撤回した。

    罠: 参考年齢・参考性別の節まで巻き添えで消す実装は、この test の後半で
    落ちる。
    """
    camera = _camera(series=_series(), overview=_overview())
    image = _image(estimate=_estimate(), show_demographics=True)

    for html in (camera, image):
        for banned in ("全体のスコア", "点", "求め方"):
            assert banned not in html

    assert camera.count("<svg") == 1
    assert camera.count("<polyline") == 5
    assert DEMOGRAPHIC_HEADING in camera
    assert DEMOGRAPHIC_HEADING in image


def test_the_camera_report_carries_the_reference_age_rate() -> None:
    """FR-021 (5): 画面と同じ値をレポートにも出す（同じ domain の結果を読む）。"""
    html = _camera()

    assert "参考年齢を表示できたフレーム" in html
    assert "62%" in html


def test_the_report_never_names_why_the_reference_age_was_hidden() -> None:
    """§10.11-2: できなかった内訳を、集計でも復元させない。"""
    html = _camera()

    for banned in ("範囲外", "undetermined", "out_of_range", "未成年", "18歳"):
        assert banned not in html


def test_a_session_the_gate_never_ran_omits_the_rate_row() -> None:
    section = _section(None)
    silent = type(section)(
        segment_id=None,
        total_count=section.total_count,
        ok_count=section.ok_count,
        valid_rate=section.valid_rate,
        valid_span_ms=section.valid_span_ms,
        symmetry=section.symmetry,
        parts=section.parts,
        reason_counts=section.reason_counts,
    )
    html = _camera(summary=SessionSummary(overall=silent, segments=()))

    assert "参考年齢を表示できたフレーム" not in html


# --- §9.6 / FR-042: 参考年齢・参考性別（2026-08-26 に両モード共通へ） ---------


def _overview():
    from face_symmetry_lab.application.demographic_observations import (
        DemographicOverview,
    )
    from face_symmetry_lab.domain.demographics import ObservedAges

    return DemographicOverview(
        ages=ObservedAges(median=31, low=29, high=34, sample_count=42),
        gender_probabilities=(0.2, 0.8),
    )


def _estimate(age: float = 31.0):
    from datetime import UTC, datetime

    from face_symmetry_lab.domain.demographics import build_demographic_estimate

    return build_demographic_estimate(
        estimated_age_years=age,
        gender_probabilities=(0.2, 0.8),
        model_version="age-gender/test",
        model_sha256="b" * 64,
        observed_at=datetime(2026, 8, 26, tzinfo=UTC),
    )


def test_both_modes_carry_the_reference_age_under_the_same_heading() -> None:
    """§9.6（2026-08-26）: 見出しと並びをモードで変えない。

    罠: Camera だけ節ごと持たない旧実装は、同じセッションを画面で見た人と
    ファイルで受け取った人に別の内容を見せる。
    """
    camera = _camera(overview=_overview())
    image = _image(estimate=_estimate(), show_demographics=True)

    assert DEMOGRAPHIC_HEADING in camera
    assert DEMOGRAPHIC_HEADING in image


def test_the_camera_report_names_the_range_it_actually_showed() -> None:
    """§9.6.1: 1 点を持たないので、中央値と真ん中 50% の範囲で述べる。"""
    html = _camera(overview=_overview())

    assert "31 歳" in html
    assert "29〜34 歳" in html
    assert "男性" in html
    assert "女性" in html
    assert "80%" in html
    assert "20%" in html
    assert "42" in html


def test_a_single_year_session_is_written_without_a_range() -> None:
    from face_symmetry_lab.application.demographic_observations import (
        DemographicOverview,
    )
    from face_symmetry_lab.domain.demographics import ObservedAges

    html = _camera(
        overview=DemographicOverview(
            ages=ObservedAges(median=31, low=31, high=31, sample_count=3),
            gender_probabilities=(0.9, 0.1),
        )
    )

    assert "31 歳" in html
    assert "31〜31 歳" not in html


def test_a_camera_session_that_showed_nothing_has_no_reference_age_section() -> None:
    """§9.6.1 / §10.11-2: 表示できなかったものをレポートへ出さない。"""
    html = _camera(overview=None)

    assert DEMOGRAPHIC_HEADING not in html


def test_the_camera_report_carries_the_gate_notice_instead_of_a_number() -> None:
    """§10.11-2: 数値を出せないときに置くのは固定文言だけである。"""
    html = _camera(overview=None, notice="この画像では参考年齢・参考性別を表示できません。")

    assert "この画像では参考年齢・参考性別を表示できません。" in html


def test_the_reference_age_is_laid_out_the_same_way_in_both_modes() -> None:
    """依頼（2026-08-26）: 画像と同じ見やすさで動画にも出す。

    罠: 片方だけ表、もう片方だけ本文にする実装は「そろえた」ことにならない。
    同じ組み立て（`_demographic_tiles`）を通ることを、マークアップで確かめる。
    """
    camera = _camera(overview=_overview())
    image = _image(estimate=_estimate(), show_demographics=True)

    for html in (camera, image):
        assert html.count('class="tile"') == 2
        assert "参考性別" in html
