"""画面に出る 1 つの数値を「どこから読み、どう整形するか」だけを決める層。

**この module は何も描画しない。** Streamlit を import せず、`AnalysisOutcome`
から表示値を取り出す関数と、値を文字列にする関数、そして状態→色・部位→ラベル
の対応表だけを持つ。

**分離した理由。** これを読む側は 4 つある——詳細パネル（`ui.scores`）、映像上の
HUD（`ui.hud`）、静止画の吹き出し（`ui.still_overlay`）、Image の組み立て
（`ui.image_section`）。同じ測定値に対して 2 つ目の表示規則が生まれると、画面上
の別々の場所が別々の数字を出す。以前これらが `ui.scores`（描画 module）に同居
していたため、HUD が値を整形するためだけに描画 module へ依存していた。

規則は 2 つだけである。

* **欠測は `—`。** 0 でも前回値でもない（不変条件 11、NFR-013）。
* **対称度は %、全体のスコアは 点。** 目盛りが違うので、`%` を付ける関数と
  付けない関数を分けてある（§8.6.7）。
"""

from __future__ import annotations

from typing import Final

from face_symmetry_lab.application.analyzer import AnalysisOutcome
from face_symmetry_lab.domain.models import AnalysisState, GenderLabel, Status
from face_symmetry_lab.domain.score_scale import score_points

PART_LABELS: Final[dict[str, str]] = {
    "contour": "輪郭",
    "eyes": "目",
    "nose": "鼻",
    "mouth": "口",
}

GENDER_LABELS: Final[dict[GenderLabel, str]] = {
    GenderLabel.FEMALE: "女性",
    GenderLabel.MALE: "男性",
}


# --- 整形 --------------------------------------------------------------------


def format_distance(value: float | None) -> str:
    """正規化距離を百分率で表示する。欠測時は `—` を返す。"""
    return "—" if value is None else f"{value * 100.0:.2f}%"


def symmetry_percent(value: float | None) -> float | None:
    """§8.6.3 の対称度変換を、文字列ではなく数値のまま返す。

    `format_symmetry` と同じ式を共有する唯一の実装であり、`ui.hud` の折れ線
    グラフが数値を必要とする場所（テキスト表示とは別に、SVG の `points` 属性
    を計算する箇所）はここを経由する。式を 2 箇所に書くと、グラフとテキストが
    別々の表示規則へ分岐しうる。

    ずれが両眼間隔そのものを超える退化ケースのために 0% で下限を設ける。
    それ以外の場所では `d` は上限のない値のままである（§8.6.3）。
    """
    if value is None:
        return None
    return min(100.0, max(0.0, (1.0 - value) * 100.0))


def format_symmetry(value: float | None) -> str:
    """正規化距離を §8.6.3 の対称度パーセントとして表示する。

    `100% - ずれ%` は線形かつ可逆で、係数を持たない。したがって同じ測定値の
    読む向きを変えるだけであり、採点ではない。指標そのもの、エクスポートの
    列、`metric_version` はいずれもこれによって変わらない。
    """
    percent = symmetry_percent(value)
    return "—" if percent is None else f"{percent:.2f}%"


def format_score(value: float | None) -> str:
    """§8.6.7: `s_3d` を「全体のスコア」の点として表示する。

    **`%` を付けない。** 対称度（`format_symmetry`）と同じ目盛りに見えると、
    82 点と 98% が同じ量の 2 つの読み方だと誤解される。両者は目盛りが違う。

    小数 1 桁で止める。§14.4 のセッション間全幅が 6.8 点であるため、それより
    細かい桁は撮り直しで消える。欠測は `—` であり、0 点ではない（0 点は
    「ずれ 5.0% 以上」という測定結果である、不変条件 11）。
    """
    points = score_points(value)
    return "—" if points is None else f"{points:.1f} 点"


# --- 値の解決 ----------------------------------------------------------------


def resolve_overall_distance(outcome: AnalysisOutcome) -> float | None:
    """§8.6.3/§8.7: 「全体の対称度」として表示する 3D の総合距離を 1 か所で決める。

    平滑値（`overall_distance_3d_smooth`）を優先し、無ければ生値
    （`symmetry_3d.overall_distance`）へ落ちる。静止画は §8.7 のとおり平滑値を
    持たない（`application.still_image.analyze_still_image` が明示的に欠測へ
    落とす）ため、この関数は自然に生値へ落ちる——呼び出し側が入力モードを
    分岐する必要はない。3D 一式が丸ごと欠測なら `None`（`—`）を返し、`symmetry`
    （2D）へは決してフォールバックしない。
    """
    if outcome.overall_distance_3d_smooth is not None:
        return outcome.overall_distance_3d_smooth
    symmetry_3d = outcome.symmetry_3d
    return None if symmetry_3d is None else symmetry_3d.overall_distance


def resolve_equal_weight_distance(outcome: AnalysisOutcome) -> float | None:
    """§8.6.3/§8.6.5: 「全体のスコア」として表示する `s_3d` を 1 か所で決める。

    `s_3d` は §8.6.5・§8.7 のとおり EMA の対象に含めないため、常に生値である
    （`resolve_overall_distance` と異なり、平滑値へのフォールバックを持たない
    ——そもそも平滑値が存在しない）。3D 一式が欠測、または部位が 1 つでも
    欠測していれば `None`（`—`）を返し、0 や前回値へ置換しない（不変条件 11）。
    `Symmetry3DResult.equal_weight_distance` が「部位が 1 つでも欠測なら欠測に
    する。使える部位だけで平均しない」を実装しているので、ここではそれを
    そのまま読むだけでよい。
    """
    symmetry_3d = outcome.symmetry_3d
    if symmetry_3d is None:
        return None
    return symmetry_3d.equal_weight_distance


def resolve_part_distance_3d(outcome: AnalysisOutcome, name: str) -> float | None:
    """§8.6.3/§8.7 改訂(H1): 部位別 `d_3d,p` の平滑値を 1 か所で決める。

    `resolve_overall_distance` と同じ「平滑値を優先し、無ければ生値」という
    解決を部位別にも適用する。`outcome.part_distances_3d_smooth` がコンテナ
    ごと欠測(静止画・3D 一式欠測)、またはその部位のキーだけ欠測(不変条件 11、
    部位が個別に欠測したフレーム)のいずれでも自然に生値へ落ちる。詳細タブと
    HUD のどちらの呼び出し側もこの解決を個別に書かないよう、ここへ集約する
    (表示の一貫性: 片方が生値・片方が平滑値という 2 つの表示規則を作らない)。
    """
    smoothed = outcome.part_distances_3d_smooth
    if smoothed is not None:
        value = smoothed.get(name)
        if value is not None:
            return value
    symmetry_3d = outcome.symmetry_3d
    if symmetry_3d is None:
        return None
    return symmetry_3d.distance(name)


# --- 品質ゲートの色 -----------------------------------------------------------

#: §14.3e: 品質ゲートの状態ごとの色。映像の枠線と HUD の輪郭に使う。これらは
#: **状態の識別子であって評価ではない**。緑は「計測条件を満たしている」という
#: 意味であって、「良い顔である」ではない。§5.4 は色だけで意味を伝えることを
#: 禁じているため、ゲートが閉じている間は詳細結果タブの `render_guidance` が
#: 状態文言と理由を示し、ゲートが閉じると HUD・詳細結果の数値が `—` へ変わる
#: こと（値の有無）と合わせて色以外の手掛かりを保つ（項目26）。
GATE_COLORS: Final[dict[Status, str]] = {
    Status.OK: "#2E9E5B",
    Status.HOLD: "#D9A21B",
    Status.NO_FACE: "#B4433B",
    Status.MULTI_FACE: "#B4433B",
    Status.ERROR: "#6B6B6B",
}

#: 解析がまだ 1 つも無いとき、および表が取りこぼした状態のときに使う。判定を
#: 匂わせる色ではなく、`Status.ERROR` の中立的な灰色を選ぶ。
_UNKNOWN_GATE_COLOR: Final[str] = GATE_COLORS[Status.ERROR]


def gate_color(state: AnalysisState | None) -> str:
    """ゲート状態 1 つに対する §14.3e の表示色。

    ``None``（このセッションでまだ解析が無い）は中立的な灰色として読む。
    待機中の表示が「条件を満たしている」色を借りることがないようにする。
    """
    if state is None:
        return _UNKNOWN_GATE_COLOR
    return GATE_COLORS.get(state.status, _UNKNOWN_GATE_COLOR)
