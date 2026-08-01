#!/usr/bin/env python3
"""Run the local quality gates in the same order as CI."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def _required_executable(name: str) -> str:
    executable = shutil.which(name)
    if executable is None:
        raise SystemExit(f"required executable is missing: {name}")
    return executable


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    commands = [
        [_required_executable("ruff"), "format", "--check", "."],
        [_required_executable("ruff"), "check", "."],
        [_required_executable("mypy"), "src"],
        [sys.executable, "-m", "pytest", "-q"],
        [
            sys.executable,
            "scripts/generate_structure.py",
            "--include-untracked",
            "--check",
        ],
        ["git", "diff", "--check"],
    ]

    for command in commands:
        print(f"+ {' '.join(command)}", flush=True)
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
