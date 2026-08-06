"""OpenCV による Delaunay +区分アフィン変形（§8.6.4, FR-035）。

`application.ports.PiecewiseWarper` を実装する。対称化画像機能のうち画素
ライブラリを必要とするのはここだけであり、それが adapter である理由である。
どの点が動き、どの点が固定され、そもそも変位が安全かは `domain.warp_mesh` が
すでに決めている。

`cv2` は `opencv-contrib-python` だけに由来する（不変条件 9）。モデルの読み込み
もファイルの書き出しも行わない。変形は呼び出し側が所有する配列に対する算術で
ある。

§8.6.4 の手順は次のとおり。まず *変形先* の点を三角形分割し、出力が隙間も重なり
も無く敷き詰められるようにする。次に同じ index の三つ組を変形元へ適用し、三角形
ごとに専用のアフィン写像で画素を転送する。メッシュが支えられない三角形（退化
しているもの、頂点が index へ解決できないもの）は飛ばし、推測せずに元の画素を
残す。
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.warp_mesh import WarpMesh, is_degenerate_triangle


class PiecewiseWarpError(ValueError):
    """この入力に対して変形を実行できなかった。"""


#: 頂点は、丸めた座標をキーにした表で点の index へ戻す。`cv2.Subdiv2D` は渡した
#: index ではなく座標を返すためである。この桁数で丸めると、Subdiv2D 内部の
#: float32 往復による誤差を吸収しつつ、本当に別々のランドマークを統合せずに済む。
_VERTEX_DECIMALS = 2


class OpenCvPiecewiseWarper:
    """顔画像を、その対称化ランドマーク位置へ変形する。"""

    def warp(self, rgb_image: NDArray[np.uint8], mesh: WarpMesh) -> NDArray[np.uint8]:
        """``mesh`` に従って変形元→変形先へ変形した ``rgb_image`` の複製を返す。

        返す画像は元画像の複製から始める。そのため、どの三角形にも覆われない
        領域は黒くならず、元の画素を保つ。
        """
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise PiecewiseWarpError("ワーピングには 3 チャンネルの RGB 画像が必要です。")
        height, width = rgb_image.shape[:2]
        if (width, height) != (mesh.width, mesh.height):
            raise PiecewiseWarpError("画像サイズが変形メッシュと一致しません。")

        warped = rgb_image.copy()
        for triangle in self._triangles(mesh):
            source_triangle = mesh.source[list(triangle)]
            target_triangle = mesh.target[list(triangle)]
            if is_degenerate_triangle(source_triangle) or is_degenerate_triangle(target_triangle):
                continue
            _transfer_triangle(rgb_image, warped, source_triangle, target_triangle)
        return warped

    def _triangles(self, mesh: WarpMesh) -> tuple[tuple[int, int, int], ...]:
        """変形先の点に対する Delaunay 三角形を、index の三つ組として返す。"""
        # 矩形を 1 画素分広げている。Subdiv2D は境界を半開区間として扱い、遠い側
        # の辺の上にちょうど乗る点を拒否するためである。`build_warp_mesh` が外周
        # アンカーをクランプする先が、まさにその位置である。
        subdiv = cv2.Subdiv2D((0, 0, mesh.width + 1, mesh.height + 1))
        lookup: dict[tuple[float, float], int] = {}
        for index, point in enumerate(mesh.target):
            key = _vertex_key(point)
            if key in lookup:
                # 座標が重複すると後から区別できないので、最初の index を採用
                # する。そこへ解決される三角形は、単に両方の点を一緒に動かす。
                continue
            lookup[key] = index
            try:
                subdiv.insert((float(point[0]), float(point[1])))
            except cv2.error as error:  # pragma: no cover - guarded by clamping
                raise PiecewiseWarpError(f"三角形分割に失敗しました: {error}") from error

        triangles: list[tuple[int, int, int]] = []
        for entry in subdiv.getTriangleList():
            corners = (
                _vertex_key(entry[0:2]),
                _vertex_key(entry[2:4]),
                _vertex_key(entry[4:6]),
            )
            resolved = tuple(lookup[corner] for corner in corners if corner in lookup)
            # Subdiv2D は仮想的な外側頂点に接する三角形も報告する。それらは
            # 一度も挿入していない頂点を含む。
            if len(resolved) == 3:
                triangles.append((resolved[0], resolved[1], resolved[2]))
        return tuple(triangles)


def _vertex_key(point: Sequence[float] | NDArray[np.float64]) -> tuple[float, float]:
    return (round(float(point[0]), _VERTEX_DECIMALS), round(float(point[1]), _VERTEX_DECIMALS))


def _transfer_triangle(
    source_image: NDArray[np.uint8],
    destination: NDArray[np.uint8],
    source_triangle: NDArray[np.float64],
    target_triangle: NDArray[np.float64],
) -> None:
    """三角形 1 つ分の画素をアフィン写像し、``destination`` へ直接書き込む。

    フレーム全体ではなく 2 つの外接矩形の上で処理するので、費用は画像の大きさ
    ではなく三角形の大きさに比例する。塗り潰した mask で合成するため、書き込ま
    れるのは三角形の内側だけである。
    """
    source_box = cv2.boundingRect(np.asarray([source_triangle], dtype=np.float32))
    target_box = cv2.boundingRect(np.asarray([target_triangle], dtype=np.float32))
    if source_box[2] <= 0 or source_box[3] <= 0 or target_box[2] <= 0 or target_box[3] <= 0:
        return

    source_local = np.asarray(
        [[point[0] - source_box[0], point[1] - source_box[1]] for point in source_triangle],
        dtype=np.float32,
    )
    target_local = np.asarray(
        [[point[0] - target_box[0], point[1] - target_box[1]] for point in target_triangle],
        dtype=np.float32,
    )
    patch = source_image[
        source_box[1] : source_box[1] + source_box[3],
        source_box[0] : source_box[0] + source_box[2],
    ]
    if patch.size == 0:
        return

    affine = cv2.getAffineTransform(source_local, target_local)
    warped_patch = cv2.warpAffine(
        patch,
        affine,
        (target_box[2], target_box[3]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((target_box[3], target_box[2], 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.asarray(target_local, dtype=np.int32), (1, 1, 1), cv2.LINE_AA)
    region = destination[
        target_box[1] : target_box[1] + target_box[3],
        target_box[0] : target_box[0] + target_box[2],
    ]
    if region.shape != warped_patch.shape:  # pragma: no cover - clipped at the border
        return
    np.copyto(region, warped_patch, where=mask.astype(bool))
