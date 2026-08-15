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
