"""不変なランドマーク DTO と、オーバーレイが必要とする純粋な変換。

このモジュールは追跡データだけを持つ。`analysis_status`、左右差スコア、品質
ゲートを意図的に表現しない。ランドマークが存在するという事実だけでは、その
フレームが計測に適しているかどうかを何も語らないためである。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from face_symmetry_lab.domain.pose import FacePose

#: Face Landmarker は 478 点を返す。フレームの契約はこれを必須としないので、
#: 下流のコードは点数を減らした fixture でも動作できる。
FACE_LANDMARK_COUNT = 478


class LandmarkError(ValueError):
    """domain 境界におけるランドマークデータが不正である。"""


class TrackingStatus(StrEnum):
    """そのフレームが何人の顔を含んでいたか。§6.2 の解析状態とは別物である。"""

    NO_FACE = "NO_FACE"
    SINGLE_FACE = "SINGLE_FACE"
    MULTI_FACE = "MULTI_FACE"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class LandmarkFrame:
    """検出結果 1 件。計算元となったフレームに結び付いている。

    ``depths`` は §8.1.1 の推定奥行き ``z`` で、``points`` と 1 対 1 に対応する。
    **これは MediaPipe が推定した相対奥行きであり、実測値ではない。** 原点は
    頭部中心付近に置かれた相対量で、絶対距離としての較正を持たず、xy と同じ
    精度を仮定してはならない。正規化のスケールも xy とは異なり、``x`` と同じ
    係数 —— 画像の *幅* —— で正規化されている（`to_pixel_array_3d` を参照）。

    奥行きを取れなかったフレームでは ``None`` になる。0 で埋めない（不変条件
    11）。この版の指標・表示・エクスポートは ``depths`` を読まないので、欠測
    でも xy の経路はこれまでどおり動く。
    """

    frame_id: int
    face_count: int
    points: tuple[tuple[float, float], ...] = ()
    pose: FacePose | None = None
    width: int = 0
    height: int = 0
    depths: tuple[float, ...] | None = None

    @property
    def has_depth(self) -> bool:
        """§8.1.1 の推定奥行きを持っているか。"""
        return self.depths is not None

    @property
    def face_area_ratio(self) -> float | None:
        """ランドマークの外接矩形面積。フレーム全体に対する割合で返す。"""
        if not self.has_single_face or not self.points:
            return None
        xs = [x for x, _ in self.points]
        ys = [y for _, y in self.points]
        return float((max(xs) - min(xs)) * (max(ys) - min(ys)))

    @property
    def has_single_face(self) -> bool:
        return self.face_count == 1

    @property
    def status(self) -> TrackingStatus:
        if self.face_count == 0:
            return TrackingStatus.NO_FACE
        if self.face_count == 1:
            return TrackingStatus.SINGLE_FACE
        return TrackingStatus.MULTI_FACE


def build_landmark_frame(
    *,
    frame_id: int,
    face_count: int,
    points: tuple[tuple[float, float], ...] = (),
    pose: FacePose | None = None,
    width: int = 0,
    height: int = 0,
    depths: tuple[float, ...] | None = None,
) -> LandmarkFrame:
    """検出結果を、オーバーレイや UI へ届く前に検証する。

    ``depths`` は「有限値がすべて揃っているか、フレームまるごと欠測か」の
    どちらかしか受け付けない。半端に欠けた奥行きを domain の内側へ入れない
    ためである。検出器が一部の点で ``z`` を欠く場合、その判断は adapter が
    行い、ここには ``None`` が届く（§8.1.1）。
    """
    if frame_id < 0:
        raise LandmarkError("frame_id は 0 以上である必要があります。")
    if face_count < 0:
        raise LandmarkError("face_count は 0 以上である必要があります。")
    if face_count != 1 and points:
        raise LandmarkError("単一顔のフレームだけがランドマークを保持できます。")
    if width < 0 or height < 0:
        raise LandmarkError("画像サイズは 0 以上である必要があります。")
    for x, y in points:
        if not (isfinite(x) and isfinite(y)):
            raise LandmarkError("ランドマーク座標は有限値である必要があります。")
    if depths is not None:
        if face_count != 1:
            raise LandmarkError("単一顔のフレームだけが奥行きを保持できます。")
        if len(depths) != len(points):
            raise LandmarkError("奥行きはランドマークと同数である必要があります。")
        if not all(isfinite(depth) for depth in depths):
            raise LandmarkError("奥行きは有限値である必要があります。")
    if pose is not None and not all(
        isfinite(value) for value in (pose.yaw_deg, pose.pitch_deg, pose.roll_deg)
    ):
        raise LandmarkError("姿勢角は有限値である必要があります。")
    return LandmarkFrame(
        frame_id=frame_id,
        face_count=face_count,
        points=points,
        pose=pose,
        width=width,
        height=height,
        depths=depths,
    )


def to_pixel_array(frame: LandmarkFrame) -> NDArray[np.float64]:
    """画素座標のランドマーク。縦横比で歪めてはならない指標のために使う。

    奥行きを持つフレームでも `(N, 2)` を返す。§8.3〜§8.6 の指標、オーバーレイ、
    対称化はすべて面内だけで定義されており、`z` を渡されても読む場所がない。
    3 軸が要る呼び出し側は `to_pixel_array_3d` を使う。
    """
    if frame.width <= 0 or frame.height <= 0:
        raise LandmarkError("画素座標に戻すには画像サイズが必要です。")
    if not frame.has_single_face:
        raise LandmarkError("単一顔のフレームだけを画素座標へ変換できます。")
    return np.array(
        [(x * frame.width, y * frame.height) for x, y in frame.points],
        dtype=np.float64,
    )


def to_pixel_array_3d(frame: LandmarkFrame) -> NDArray[np.float64]:
    """画素相当の 3D ランドマーク `(N, 3)`。3 軸が同じ長さの単位で揃う（§8.1.1）。

    MediaPipe は `x` を画像幅、`y` を画像高さで正規化し、`z` は「`x` とおよそ
    同じスケール」と規定する。したがって `z` へ掛ける係数は高さではなく
    **画像の幅**である。この規約により、返る配列の xy 列は `to_pixel_array`
    の値と厳密に一致する。

    `z` はモデルの推定奥行きであって実測値ではない（`LandmarkFrame` を参照）。
    奥行きが欠測のフレームは 0 で埋めずに拒否する。呼び出し側は
    `LandmarkFrame.has_depth` で先に分岐すること。
    """
    if frame.depths is None:
        raise LandmarkError("奥行きが欠測のフレームは 3D 画素座標へ変換できません。")
    if frame.width <= 0 or frame.height <= 0:
        raise LandmarkError("画素座標に戻すには画像サイズが必要です。")
    if not frame.has_single_face:
        raise LandmarkError("単一顔のフレームだけを画素座標へ変換できます。")
    return np.array(
        [
            (x * frame.width, y * frame.height, depth * frame.width)
            for (x, y), depth in zip(frame.points, frame.depths, strict=True)
        ],
        dtype=np.float64,
    )


def to_pixel_points(
    frame: LandmarkFrame,
    *,
    width: int,
    height: int,
) -> tuple[tuple[int, int], ...]:
    """正規化ランドマークを画素座標へ変換し、画像内へクランプする。"""
    if width <= 0 or height <= 0:
        raise LandmarkError("画像サイズは正の値である必要があります。")
    if not frame.has_single_face:
        return ()
    max_x = width - 1
    max_y = height - 1
    return tuple(
        (
            min(max(int(x * width), 0), max_x),
            min(max(int(y * height), 0), max_y),
        )
        for x, y in frame.points
    )
