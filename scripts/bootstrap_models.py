#!/usr/bin/env python3
"""Prepare the pinned model assets (FR-002, §10.6, §19).

Nothing is downloaded unless the manifest is explicitly accepted, either
interactively or with ``--accept-manifest``. Every install verifies size and
sha256 before the file is put in place.

    uv run python scripts/bootstrap_models.py            # show manifest and ask
    uv run python scripts/bootstrap_models.py --verify   # check installed files
    uv run python scripts/bootstrap_models.py --accept-manifest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from face_symmetry_lab.adapters.io.model_store import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MODELS_DIR,
    InstallStatus,
    ModelAsset,
    ModelStore,
    ModelStoreError,
    load_manifest,
)


def _describe(asset: ModelAsset, status: InstallStatus) -> str:
    return (
        f"  name    : {asset.name}\n"
        f"  version : {asset.version}\n"
        f"  url     : {asset.url}\n"
        f"  size    : {asset.size_bytes:,} bytes\n"
        f"  sha256  : {asset.sha256}\n"
        f"  license : {asset.license} ({asset.license_url})\n"
        f"  input   : {asset.input_contract}\n"
        f"  status  : {status.value}"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--model", help="restrict to one manifest entry by name")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify installed files against the manifest and exit",
    )
    parser.add_argument(
        "--accept-manifest",
        action="store_true",
        help="accept the shown manifest without the interactive prompt",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    try:
        assets = load_manifest(args.manifest)
    except ModelStoreError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if args.model:
        assets = tuple(asset for asset in assets if asset.name == args.model)
        if not assets:
            print(f"error: manifest に {args.model} がありません。", file=sys.stderr)
            return 2

    store = ModelStore(models_dir=args.models_dir)
    statuses = {asset.name: store.status(asset) for asset in assets}

    print(f"manifest: {args.manifest}")
    print(f"models  : {args.models_dir}\n")
    for asset in assets:
        print(_describe(asset, statuses[asset.name]))
        print()

    if args.verify:
        failed = [name for name, status in statuses.items() if status is not InstallStatus.READY]
        if failed:
            print(f"verification failed: {', '.join(failed)}", file=sys.stderr)
            return 1
        print("all assets verified")
        return 0

    pending = [asset for asset in assets if statuses[asset.name] is not InstallStatus.READY]
    if not pending:
        print("all assets already installed and verified")
        return 0

    if not args.accept_manifest:
        print("上記の取得元・サイズ・ライセンス・sha256 に同意する場合のみ取得します。")
        try:
            answer = input("download now? [y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in {"y", "yes"}:
            print("aborted: no model was downloaded")
            return 1

    for asset in pending:
        print(f"downloading {asset.name} ...", flush=True)
        try:
            installed = store.install(asset)
        except ModelStoreError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"installed: {installed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
