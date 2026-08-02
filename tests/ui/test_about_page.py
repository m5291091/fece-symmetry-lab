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
    """§4.1, FR-014: the page users are pointed to for "how is this scored"."""
    at = _run()

    headers = [element.value for element in at.header]
    subheaders = [element.value for element in at.subheader]

    assert "スコアの算出方法" in headers
    assert any("幾何左右差" in text for text in subheaders)
    assert any("肌表面参考値" in text for text in subheaders)
    assert any("参考年齢" in text for text in subheaders)


def test_about_page_states_prohibited_uses() -> None:
    """§10.7: prohibited-use guardrails must be documented on the About page."""
    at = _run()

    headers = [element.value for element in at.header]
    markdown_text = " ".join(element.value for element in at.markdown)

    assert "使用しないでください" in headers
    assert "医療" in markdown_text
    assert "本人確認" in markdown_text
