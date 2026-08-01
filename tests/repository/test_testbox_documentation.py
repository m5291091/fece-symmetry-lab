from pathlib import Path


def test_testbox_lists_every_test_file() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    documentation = (repository_root / "testbox.md").read_text(encoding="utf-8")
    test_files = sorted(
        path.relative_to(repository_root).as_posix()
        for path in (repository_root / "tests").rglob("test_*.py")
    )

    missing = [path for path in test_files if f"`{path}`" not in documentation]

    assert missing == [], f"testbox.md is missing: {missing}"
