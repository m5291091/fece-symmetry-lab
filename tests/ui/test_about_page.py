from streamlit.testing.v1 import AppTest

_SCRIPT = "from face_symmetry_lab.ui.about_page import render_about_page\nrender_about_page()\n"


def _run() -> AppTest:
    at = AppTest.from_string(_SCRIPT)
    at.run()
    return at


def test_about_page_renders_without_raising() -> None:
    at = _run()

    assert not at.exception


def test_about_page_explains_score_methodology() -> None:
    """§4.1, FR-014: 「どう算出しているか」を知りたい利用者を案内するページ。"""
    at = _run()

    headers = [element.value for element in at.header]
    subheaders = [element.value for element in at.subheader]

    assert "左右差の算出方法" in headers
    assert any("幾何左右差" in text for text in subheaders)
    assert any("肌表面参考値" in text for text in subheaders)
    assert any("参考年齢" in text for text in subheaders)


def test_about_page_states_prohibited_uses() -> None:
    """§10.7: 禁止用途のガードレールは About ページに明記されねばならない。"""
    at = _run()

    headers = [element.value for element in at.header]
    markdown_text = " ".join(element.value for element in at.markdown)

    assert "使用しないでください" in headers
    assert "医療" in markdown_text
    assert "本人確認" in markdown_text


def test_about_page_lists_the_age_gate_version() -> None:
    """FR-038/§22.5: 「モデルとバージョン」に age-gate の版・spec_revision・sha256。"""
    at = _run()

    markdown_text = " ".join(element.value for element in at.markdown)

    assert "参考年齢の表示ゲート" in markdown_text
    assert "age-gate/1.0.0" in markdown_text
    assert "spec_revision 3" in markdown_text


# --- FR-039: 参照平均の併記 ----------------------------------------------------


def test_about_page_explains_the_reference_baseline() -> None:
    """FR-039/§10.12: 画面に出る数字の出所を About でも説明する。

    吹き出しの但し書き（`ui.readings.BASELINE_CAVEAT`）は狭い箱に収まる長さしか
    持てない。About は、そこへ書ききれない条件——どのセルを読むか、いつ併記が
    消えるか、参照統計ブラウザ（FR-037）と何が違うか——を担う。
    """
    at = _run()

    subheaders = [element.value for element in at.subheader]
    markdown_text = " ".join(element.value for element in at.markdown)

    assert any("参照平均" in text for text in subheaders)
    assert "UTKFace" in markdown_text
    assert "非商用研究限定" in markdown_text
    # 母集団の決め方（§10.12-2）と、人種を推定しないこと（不変条件 7）。
    assert "人種では絞り込みません" in markdown_text
    # 併記が消える条件（§10.12-3・§10.12-4、AC-BASELINE-03）。
    assert "参考年齢・参考性別を表示していないときは併記しません" in markdown_text
    assert "8 件" in markdown_text


def test_about_page_denies_ranking_from_the_baseline() -> None:
    """§10.12-1 / AC-BASELINE-04: 差は引き算であって順序への写像ではない。"""
    at = _run()

    markdown_text = " ".join(element.value for element in at.markdown)

    assert "偏差値" in markdown_text  # 「作らない」と否定形で述べる
    assert "基準値でも目標値でもありません" in markdown_text
    for banned in ("平均より上", "平均より下", "合格"):
        assert banned not in markdown_text


def test_about_page_forbids_reading_the_difference_as_a_rank() -> None:
    """禁止用途の一覧にも、併記の誤用を 1 行として置く（§10.7・§10.12-5）。"""
    at = _run()

    markdown_text = " ".join(element.value for element in at.markdown)

    assert "参照平均との差を、順位・優劣・目標として扱うこと" in markdown_text


def test_about_states_that_a_saved_report_may_contain_a_photo() -> None:
    """§10.4 / ADR-0009: ダウンロード後のファイルはアプリの管理外である。

    レポート本体にも同じことを書くが（受け取った人へ届く）、About にも要る
    ——保存する前に読む場所だからである。
    """
    at = _run()
    text = " ".join(str(element.value) for element in at.markdown)

    assert "顔写真を含む図がファイルへ入ります" in text
    assert "共有先や複製をアプリからは制御できません" in text
    assert "自動でファイルを作ることはなく" in text
