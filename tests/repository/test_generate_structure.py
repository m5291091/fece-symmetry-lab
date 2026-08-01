import pytest

from scripts.generate_structure import (
    BEGIN_MARKER,
    END_MARKER,
    render_tree,
    replace_generated_tree,
    select_repository_paths,
)


def test_select_repository_paths_excludes_secrets_caches_and_unsafe_paths() -> None:
    selected = select_repository_paths(
        [
            "src/app.py",
            "./tests/test_app.py",
            ".env",
            ".env.example",
            ".venv/lib/package.py",
            "../outside.txt",
            "/absolute.txt",
            "tmp/report.txt",
        ]
    )

    assert selected == [".env.example", "src/app.py", "tests/test_app.py"]


def test_render_tree_is_stable_and_names_repository() -> None:
    tree = render_tree(["tests/test_app.py", "src/package/app.py"])

    assert tree.startswith("face-symmetry-lab/")
    assert tree.index("`src/`") < tree.index("`tests/`")
    assert "`app.py`" in tree


def test_replace_generated_tree_preserves_curated_content() -> None:
    document = f"before\n{BEGIN_MARKER}\nold\n{END_MARKER}\nafter\n"

    replaced = replace_generated_tree(document, "new")

    assert replaced == f"before\n{BEGIN_MARKER}\nnew\n{END_MARKER}\nafter\n"


def test_replace_generated_tree_rejects_missing_markers() -> None:
    with pytest.raises(ValueError):
        replace_generated_tree("no markers", "new")
