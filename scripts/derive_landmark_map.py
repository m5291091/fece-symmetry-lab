#!/usr/bin/env python3
"""Derive the left/right landmark correspondence map (maintainer tool).

The map is not hand-written. It is derived from the Face Landmarker's own mesh
topology and frozen into ``configs/metrics/symmetry-2d-v1.json``:

1. Detect landmarks on a symmetric synthetic face, in several renderings.
2. Mirror the 3D point cloud about the face's mean x axis.
3. Keep a pair only when the mirrored nearest neighbour is *mutual* and agrees
   across every rendering. Points matching themselves become midline points.
4. Group the survivors into contour / eyes / nose / mouth from the MediaPipe
   connection sets.

Anything that does not survive is left out rather than guessed, so the frozen
map contains only verifiable correspondences.

    uv run python scripts/derive_landmark_map.py --check
    uv run python scripts/derive_landmark_map.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "tests" / "model"))

from face_symmetry_lab.adapters.io.model_store import (  # noqa: E402
    DEFAULT_MODELS_DIR,
    InstallStatus,
    ModelStore,
    find_asset,
    load_manifest,
)

DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "metrics" / "symmetry-2d-v1.json"
LANDMARK_MAP_VERSION = "mediapipe-478-mirror/1.0.0"
METRIC_VERSION = "symmetry-2d/2.0.0"


def _connection_indices(connections: Any) -> set[int]:
    indices: set[int] = set()
    for connection in connections:
        indices.add(int(connection.start))
        indices.add(int(connection.end))
    return indices


def _part_indices() -> dict[str, set[int]]:
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as Connections

    return {
        "contour": _connection_indices(Connections.FACE_LANDMARKS_FACE_OVAL),
        "eyes": (
            _connection_indices(Connections.FACE_LANDMARKS_LEFT_EYE)
            | _connection_indices(Connections.FACE_LANDMARKS_RIGHT_EYE)
            | _connection_indices(Connections.FACE_LANDMARKS_LEFT_EYEBROW)
            | _connection_indices(Connections.FACE_LANDMARKS_RIGHT_EYEBROW)
        ),
        "nose": _connection_indices(Connections.FACE_LANDMARKS_NOSE),
        "mouth": _connection_indices(Connections.FACE_LANDMARKS_LIPS),
    }


def _renderings() -> list[Image.Image]:
    from conftest import _draw_synthetic_face

    base = _draw_synthetic_face()
    return [
        base,
        base.resize((320, 320)),
        base.resize((768, 768)),
        base.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
    ]


def _detect_point_cloud(detector: Any, image: Image.Image) -> NDArray[np.float64]:
    import mediapipe as mp

    frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame))
    if len(result.face_landmarks) != 1:
        raise SystemExit("synthetic reference must contain exactly one face")
    return np.array(
        [[point.x, point.y, point.z] for point in result.face_landmarks[0]],
        dtype=np.float64,
    )


def _mirror_partner(points: NDArray[np.float64]) -> NDArray[np.int64]:
    mirrored = points.copy()
    mirrored[:, 0] = 2.0 * points[:, 0].mean() - mirrored[:, 0]
    distances = np.linalg.norm(points[:, None, :] - mirrored[None, :, :], axis=2)
    return np.asarray(distances.argmin(axis=1), dtype=np.int64)


def derive_map(model_path: Path) -> dict[str, Any]:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )

    assert mp  # imported for the side effect of a clear failure when absent

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_faces=2,
    )
    detector = FaceLandmarker.create_from_options(options)
    try:
        partners = [_mirror_partner(_detect_point_cloud(detector, r)) for r in _renderings()]
    finally:
        detector.close()

    reference = partners[0]
    count = len(reference)
    agreed = [
        index
        for index in range(count)
        if all(int(partner[index]) == int(reference[index]) for partner in partners)
        and int(reference[int(reference[index])]) == index
    ]
    agreed_set = set(agreed)

    midline = sorted(index for index in agreed if int(reference[index]) == index)
    parts: dict[str, dict[str, Any]] = {}
    for name, indices in _part_indices().items():
        pairs = sorted(
            {
                (min(index, int(reference[index])), max(index, int(reference[index])))
                for index in indices
                if index in agreed_set and int(reference[index]) != index
            }
        )
        parts[name] = {
            "pairs": [list(pair) for pair in pairs],
            "min_pairs": max(3, len(pairs) // 2),
        }

    return {
        "metric_version": METRIC_VERSION,
        "landmark_map_version": LANDMARK_MAP_VERSION,
        "landmark_count": count,
        "derivation": (
            "mutual mirrored nearest neighbour over 3D landmarks, "
            "agreed across 4 renderings of the synthetic reference face"
        ),
        "eye_anchor": {
            "left_corners": [33, 133],
            "right_corners": [263, 362],
        },
        "midline_indices": midline,
        "parts": parts,
        "notes": (
            "出力は正規化RMS距離そのもの。感度係数k_pと部位重みw_pは、固定評価セットで較正されて"
            "おらず恣意的な重み付けを持ち込むため v2.0.0 で廃止した "
            "(requirement.md §8.6, AC-SCORE-04)。全体距離は全対応ペアをプールし正中線で1回だけ"
            "折りたたんだRMSであり、部位別からの合成値ではない。"
        ),
    }


def _resolve_model() -> Path:
    asset = find_asset(load_manifest(), "face_landmarker")
    store = ModelStore(models_dir=DEFAULT_MODELS_DIR)
    if store.status(asset) is not InstallStatus.READY:
        raise SystemExit(
            "検証済みモデルが必要です: uv run python scripts/bootstrap_models.py --accept-manifest"
        )
    return store.path_for(asset)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--write", action="store_true", help="overwrite the frozen config")
    parser.add_argument("--check", action="store_true", help="fail when the config is stale")
    args = parser.parse_args(argv)

    derived = derive_map(_resolve_model())
    rendered = json.dumps(derived, ensure_ascii=False, indent=2) + "\n"

    if args.check:
        if not args.config.is_file():
            print(f"missing: {args.config}", file=sys.stderr)
            return 1
        if args.config.read_text(encoding="utf-8") != rendered:
            print(f"stale: {args.config}", file=sys.stderr)
            return 1
        print(f"{args.config} is current")
        return 0

    for name, part in derived["parts"].items():
        print(f"{name}: {len(part['pairs'])} pairs (min {part['min_pairs']})")
    print(f"midline: {len(derived['midline_indices'])} points")

    if args.write:
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_text(rendered, encoding="utf-8")
        print(f"wrote {args.config}")
    else:
        print("(dry run; pass --write to freeze)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
