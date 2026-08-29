"""対称化ビュー（FR-034 / FR-035）の描画を、Camera と Image で 1 つにする。

**分離した理由は重複である。** 以前は `ui.camera._render_symmetrization_panel`
と `ui.app._render_image_symmetrization` が、同じオーバーレイ画像・同じ 2 つの
メトリクス・同じ 3 つの注記・同じチェックボックスの help を別々に書いていた。
この節の文言は §5.3 と §10.7 が最も強く規定する場所（対称化した写真は「補正
済みの写真」と最も取り違えられやすい出力）であり、2 か所に置くと片方だけが
古い枠付けのまま残りうる。

Camera と Image で違うのは**取り込みの導線だけ**である。Camera は動いている
映像から 1 フレームを掴む必要があり（§14.3g）、Image はすでに静止した写真を
持っている。その差は呼び出し側に残し、ここには「view が 1 つあるとき、それを
どう見せるか」だけを置く。
"""

from __future__ import annotations

from typing import Final

import streamlit as st

from face_symmetry_lab.application.symmetrization import SymmetrizationView
from face_symmetry_lab.ui.i18n import t, translate_message
from face_symmetry_lab.ui.readings import format_distance

SYMMETRIZATION_HEADING: Final[str] = "#### 左右対称にそろえた位置との比較"

#: §8.6.2 の点ごとのずれは §8.5 のペア距離の半分であり、§8.6.3 が向きを変える
#: 量とは別物である。そのため対称度ではなく「ズレ」として述べ、専用の注記を
#: 添える。
DEVIATION_UNIT_CAPTION: Final[str] = (
    "ズレは、両眼の間隔を 100% としたときの、実測点と左右対称にそろえた位置との距離です。"
    "小さいほど両者が近いことだけを意味し、良し悪しを表しません。"
)

#: FR-035 の画素変形は任意指定である（全画素を作り直すため）。ラベルと help は
#: Camera / Image で完全に同じものを使う。
WARP_CHECKBOX_LABEL: Final[str] = "左右対称にそろえた画像も生成する（幾何変形・処理が重くなります）"
WARP_CHECKBOX_HELP: Final[str] = (
    "ランドマークを結んだ三角形ごとに画素を変形し、左右対称にそろえた顔画像を作ります。"
    "加工画像であり、無加工の写真ではありません。"
)

OVERLAY_IMAGE_CAPTION: Final[str] = "実測点と、左右対称にそろえた位置（色違い）を線で結んだ表示"

#: §5.3/§10.7: FR-035 の変形画像を利用者へ提示するときの枠付け。文言を直接
#: 検証できるよう定数に切り出している。これは「補正済みの写真」と最も
#: 取り違えられやすい出力である。
SYMMETRIZED_IMAGE_CAPTION: Final[str] = (
    "左右対称にそろえた画像（ランドマークを基準に画素を変形した加工画像）"
)
SYMMETRIZED_IMAGE_NOTICE: Final[str] = (
    "対応点を左右対称の位置へ移動させ、その間の画素を三角形ごとに変形した合成画像です。"
    "実際の顔写真ではなく、より正しい・望ましい顔でもありません。"
    "加工画像であることを明示せずに提示・共有しないでください。"
)
SYMMETRIZED_IMAGE_LIMITS: Final[str] = (
    "ランドマークが覆わない髪・耳・首・背景は、画像の外周を固定したまま"
    "顔の変形に緩やかに追従するため、境界付近に不自然さが残ることがあります。"
)


def render_warp_checkbox(*, key: str) -> bool:
    """§8.6.4 の画素変形を要求するチェックボックス。既定は OFF。

    ``key`` だけを呼び出し側から受け取る。Camera と Image は同じページ内で
    別々の widget を持つ必要があるが、ラベル・help・既定値まで別々にする理由は
    無い（同じ操作が同じ言葉で説明されるべきである）。
    """
    return st.checkbox(t(WARP_CHECKBOX_LABEL), key=key, help=t(WARP_CHECKBOX_HELP))


def render_symmetrized_image(view: SymmetrizationView, *, requested: bool) -> None:
    """FR-035: 変形画像を、加工であるという枠付けのまま表示する。

    §5.3/§10.7: 対称化した写真は「補正済み」や「より良い」顔と最も取り違え
    られやすい出力なので、幾何変形であることを明示し、before/after としては
    決して見せない。

    fail-soft である。変形を要求されたが変位ガードや画像化の段階が拒否した
    場合は理由を示し、上の点オーバーレイだけで成立させる（AC-SYMM-02）。
    """
    if not requested:
        return
    if view.warped is None:
        # FR-043: `view.warp_unavailable_reason` は `WarpMeshError` 等の実行時の
        # 例外文言なので、UI 境界の `translate_message`（`ui/i18n.py`）で訳す。
        # 値を埋め込んで組み立てられた文言は原文一致せず日本語のまま残る
        # （既知の穴、`translations/_errors.py`）。
        st.info(
            translate_message(view.warp_unavailable_reason)
            if view.warp_unavailable_reason
            else t("対称化した画像は生成されていません。")
        )
        return
    st.image(view.warped, caption=t(SYMMETRIZED_IMAGE_CAPTION), width="stretch")
    st.caption(t(SYMMETRIZED_IMAGE_NOTICE))
    st.caption(t(SYMMETRIZED_IMAGE_LIMITS))


def render_symmetrization_view(view: SymmetrizationView, *, warp_requested: bool) -> None:
    """算出済みの対称化ビュー 1 つを描画する（Camera と Image で共有）。

    オーバーレイ画像 → 平均・最大のズレ → 対称化した点数 → §8.6.2 の単位注記 →
    （要求されていれば）変形画像、の順で並べる。**この並びと文言を Camera /
    Image で分岐させない。**
    """
    st.image(view.overlay, caption=t(OVERLAY_IMAGE_CAPTION), width="stretch")
    mean_column, max_column = st.columns(2)
    mean_column.metric(t("平均のズレ"), format_distance(view.mean_deviation))
    max_column.metric(t("最大のズレ"), format_distance(view.max_deviation))
    st.caption(
        t(
            "対称化した点 {count} 点。"
            "対応する左右のペアが定義されていない点は、実測のまま動かしていません。"
        ).format(count=view.symmetrized_point_count)
    )
    st.caption(t(DEVIATION_UNIT_CAPTION))
    render_symmetrized_image(view, requested=warp_requested)
