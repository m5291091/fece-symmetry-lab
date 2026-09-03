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
    assert any("参考年齢" in text for text in subheaders)
    assert not any("肌表面参考値" in text for text in subheaders)


def test_about_page_no_longer_explains_the_overall_score() -> None:
    """§0 項目 34（撤回、2026-09-03）: 「全体のスコア」の説明と 0〜100 点の目盛りを
    撤回した。算出方法の節そのものは残る。"""
    at = _run()

    markdown_text = " ".join(element.value for element in at.markdown)
    subheaders = [element.value for element in at.subheader]

    assert "全体のスコア" not in markdown_text
    assert "採点や評価を目的とした値ではありません" not in markdown_text
    assert any("幾何左右差" in text for text in subheaders)
    assert "対称度" in markdown_text


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


# --- FR-039（撤回、2026-09-03）: 参照平均の併記 --------------------------------


def test_about_page_no_longer_has_a_reference_baseline_section() -> None:
    """FR-039 は撤回された（§0 項目 34、§10.12 は撤回）。節そのものが無いことを固定する。"""
    at = _run()

    subheaders = [element.value for element in at.subheader]
    markdown_text = " ".join(element.value for element in at.markdown)

    assert not any("参照平均" in text for text in subheaders)
    assert "併記" not in markdown_text
    for banned in ("平均より上", "平均より下", "合格"):
        assert banned not in markdown_text


def test_about_page_forbids_linking_the_users_own_measurement_to_reference_stats() -> None:
    """§10.7（撤回前の無条件の禁止へ戻った）: 禁止用途の一覧が撤回後の文言であること。"""
    at = _run()

    markdown_text = " ".join(element.value for element in at.markdown)

    assert "利用者自身の測定結果と参照統計を結び付けること" in markdown_text
    assert "参照平均との差を、順位・優劣・目標として扱うこと" not in markdown_text


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
