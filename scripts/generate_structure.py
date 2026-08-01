#!/usr/bin/env python3
"""Generate the compact repository tree embedded in STRUCTURE.md."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

BEGIN_MARKER = "<!-- BEGIN GENERATED TREE -->"
END_MARKER = "<!-- END GENERATED TREE -->"
MAX_DIRECTORY_DEPTH = 5
MAX_FILE_DEPTH = 3
MAX_GENERATED_LINES = 220

_EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "models",
    "tmp",
}

_TreeNode = dict[str, Any]


def select_repository_paths(paths: list[str]) -> list[str]:
    """Return sorted, safe, non-generated repository-relative paths."""
    selected: set[str] = set()
    for raw_path in paths:
        normalized = raw_path.replace("\\", "/").removeprefix("./")
        path = PurePosixPath(normalized)
        if (
            not normalized
            or path.is_absolute()
            or ".." in path.parts
            or any(part in _EXCLUDED_PARTS for part in path.parts)
        ):
            continue
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
            continue
        selected.add(path.as_posix())
    return sorted(selected)


def _new_node() -> _TreeNode:
    return {"directories": {}, "files": []}


def _build_tree(paths: list[str]) -> _TreeNode:
    root = _new_node()
    for raw_path in select_repository_paths(paths):
        parts = PurePosixPath(raw_path).parts
        if not parts:
            continue
        node = root
        for directory in parts[:-1]:
            directories: dict[str, _TreeNode] = node["directories"]
            node = directories.setdefault(directory, _new_node())
        files: list[str] = node["files"]
        files.append(parts[-1])
    return root


def _descendant_file_count(node: _TreeNode) -> int:
    directories: dict[str, _TreeNode] = node["directories"]
    files: list[str] = node["files"]
    return len(files) + sum(_descendant_file_count(child) for child in directories.values())


def _render_node(
    node: _TreeNode,
    *,
    depth: int,
    indent: str,
    output: list[str],
) -> None:
    directories: dict[str, _TreeNode] = node["directories"]
    files: list[str] = node["files"]

    for directory in sorted(directories):
        child = directories[directory]
        child_depth = depth + 1
        if child_depth >= MAX_DIRECTORY_DEPTH:
            count = _descendant_file_count(child)
            output.append(f"{indent}- `{directory}/` ({count} files)")
            continue
        output.append(f"{indent}- `{directory}/`")
        _render_node(
            child,
            depth=child_depth,
            indent=f"{indent}  ",
            output=output,
        )

    visible_files = [file_name for file_name in sorted(files) if depth + 1 <= MAX_FILE_DEPTH]
    for file_name in visible_files:
        output.append(f"{indent}- `{file_name}`")

    hidden_file_count = len(files) - len(visible_files)
    if hidden_file_count:
        output.append(f"{indent}- … ({hidden_file_count} files)")


def render_tree(paths: list[str]) -> str:
    """Render a bounded-depth Markdown tree suitable for AI context."""
    output = ["face-symmetry-lab/"]
    _render_node(_build_tree(paths), depth=0, indent="", output=output)
    if len(output) > MAX_GENERATED_LINES:
        raise RuntimeError(
            f"generated tree exceeds {MAX_GENERATED_LINES} lines "
            f"({len(output)}); increase summarization"
        )
    return "\n".join(output)


def replace_generated_tree(document: str, generated_tree: str) -> str:
    """Replace only the generated block and preserve the curated index."""
    if document.count(BEGIN_MARKER) != 1 or document.count(END_MARKER) != 1:
        raise ValueError("STRUCTURE.md must contain exactly one marker pair")
    before, remainder = document.split(BEGIN_MARKER, maxsplit=1)
    _, after = remainder.split(END_MARKER, maxsplit=1)
    return f"{before}{BEGIN_MARKER}\n{generated_tree.rstrip()}\n{END_MARKER}{after}"


def repository_paths(
    repository_root: Path,
    *,
    include_untracked: bool = False,
) -> list[str]:
    """Read the tracked, non-ignored Git file set without walking caches."""
    command = ["git", "ls-files", "--cached", "-z"]
    if include_untracked:
        command = [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ]
    result = subprocess.run(
        command,
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    paths = [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]
    return select_repository_paths(paths)


def _generated_block(paths: list[str]) -> str:
    digest = hashlib.sha256("\0".join(paths).encode()).hexdigest()
    return f"_File-list SHA-256: `{digest}`_\n\n{render_tree(paths)}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail without writing when STRUCTURE.md is stale",
    )
    parser.add_argument(
        "--include-untracked",
        action="store_true",
        help="include non-ignored untracked files",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    structure_path = repository_root / "STRUCTURE.md"
    current = structure_path.read_text(encoding="utf-8")
    desired = replace_generated_tree(
        current,
        _generated_block(
            repository_paths(
                repository_root,
                include_untracked=args.include_untracked,
            )
        ),
    )

    if current == desired:
        print("STRUCTURE.md is current")
        return 0
    if args.check:
        print(
            "STRUCTURE.md is stale; run "
            "`uv run python scripts/generate_structure.py --include-untracked`",
            file=sys.stderr,
        )
        return 1

    structure_path.write_text(desired, encoding="utf-8")
    print("Updated STRUCTURE.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
