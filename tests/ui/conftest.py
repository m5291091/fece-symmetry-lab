"""`ui` のテストが共有する `AnalysisOutcome` の組み立て。

`test_readings.py`（値の解決）と `test_scores.py`（描画）は同じ形の outcome を
必要とする。`AppTest.from_string` で実行するスクリプトからも
`from tests.ui.conftest import ...` で読めるよう、fixture ではなく素の関数
として置いてある。
"""

from __future__ import annotations

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.domain.models import AnalysisState, Status
from face_symmetry_lab.domain.symmetry import SymmetryResult
from face_symmetry_lab.domain.symmetry_3d import AxisSplit, Symmetry3DResult

#: §8.6.5: 部位ごとの `d_3d,p`/`s_3d` から `AxisSplit`/`Symmetry3DResult` を直接
#: 組み立てるテスト用 fixture。plane/depth の内訳はここでの表示テストには関係
#: しないため distance と同じ値を入れる。
_METRIC_VERSION_3D = "symmetry-3d/test"
_LANDMARK_MAP_VERSION = "map/test"


def _symmetry_3d_result(
    *,
    contour: float | None,
    eyes: float | None,
    nose: float | None,
    mouth: float | None,
    overall: float | None,
) -> Symmetry3DResult | None:
    """§8.6.5 の出力一式を、値だけから組み立てるテスト fixture。

    `overall=None` は「重ね合わせ自体が成立しない」（§8.5.1 の全欠測）を表す。
    """
    if overall is None:
        return None

    def _split(value: float | None) -> AxisSplit | None:
        return None if value is None else AxisSplit(distance=value, plane=value, depth=0.0)

    return Symmetry3DResult(
        splits={
            "contour": _split(contour),
            "eyes": _split(eyes),
            "nose": _split(nose),
            "mouth": _split(mouth),
        },
        overall=_split(overall),
        metric_version=_METRIC_VERSION_3D,
        landmark_map_version=_LANDMARK_MAP_VERSION,
    )


def _outcome_for_display(
    *,
    symmetry_3d: Symmetry3DResult | None,
    smooth_3d: float | None,
    symmetry_2d_overall: float | None = 0.5,
    part_smooth_3d: dict[str, float | None] | None = None,
) -> AnalysisOutcome:
    """§8.6.3 の表示テスト用 `AnalysisOutcome`。

    2D の値（既定 0.5）は 3D の値（テストでは概ね 0.0〜0.1）から大きく離す。
    表示ロジックが誤って 2D を読んでいれば、この差で assertion が落ちる
    （fixture 設計上の罠、いちばん重要な部分）。``part_smooth_3d`` は §8.7 改訂
    （H1）の部位別平滑値で、既定 `None`（呼び出し側の多くは生値へのフォール
    バックだけを確かめるため、指定しなくても動く）。
    """
    symmetry_2d = None
    if symmetry_2d_overall is not None:
        symmetry_2d = SymmetryResult(
            distances={
                "contour": symmetry_2d_overall,
                "eyes": symmetry_2d_overall,
                "nose": symmetry_2d_overall,
                "mouth": symmetry_2d_overall,
            },
            overall=symmetry_2d_overall,
            metric_version="symmetry-2d/test",
            landmark_map_version=_LANDMARK_MAP_VERSION,
        )
    return AnalysisOutcome(
        frame_id=1,
        state=AnalysisState(status=Status.OK),
        symmetry=symmetry_2d,
        overall_distance_smooth=symmetry_2d_overall,
        pose=None,
        face_area_ratio=None,
        metric_version="symmetry-2d/test",
        landmark_map_version=_LANDMARK_MAP_VERSION,
        symmetry_3d=symmetry_3d,
        overall_distance_3d_smooth=smooth_3d,
        part_distances_3d_smooth=part_smooth_3d,
    )
