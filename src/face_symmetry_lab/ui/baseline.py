"""FR-039: 併記する参照平均を、画面側から 1 か所で解決する。

`ui.hud`（Camera）と `ui.image_section`（Image）の両方がここを通る。**参照データの
読み込みを 2 か所で行わない**ためであり、`ui.reference_page` が `@st.cache_resource`
で browser を 1 つ保つのと同じ作法である。

**この module は判定条件を持たない。** 「参考年齢・参考性別を表示していないときは
併記しない」（§10.12-3）は、呼び出し側が `estimate` に `None` を渡すことで表現する
——年齢ゲートの状態を読むのは `ui.hud` / `ui.image_section` の既存の分岐であり、
それを 2 つ目の場所で書き直すと、片方だけがゲートを見落とす形の壊れ方をする。
"""

from __future__ import annotations

import streamlit as st

from face_symmetry_lab.adapters.io.reference_dataset import CsvReferenceCellRepository
from face_symmetry_lab.application.reference_baseline import (
    BaselineProvider,
    BaselineSummary,
)
from face_symmetry_lab.application.reference_browser import ReferenceBrowser
from face_symmetry_lab.domain.models import DemographicEstimate


@st.cache_resource
def default_baseline_provider() -> BaselineProvider:
    """§9.5 の凍結アセットを 1 度だけ読み、以後の再実行では読み直さない。

    読み込みが失敗しても例外は出ない——`ReferenceBrowser` が理由を保持したまま
    「利用できない」状態になり、`resolve` が `None` を返すだけである（不変条件 12）。
    """
    return BaselineProvider(ReferenceBrowser(CsvReferenceCellRepository()))


def resolve_baseline(
    estimate: DemographicEstimate | None,
    *,
    provider: BaselineProvider | None = None,
) -> BaselineSummary | None:
    """併記する母集団の要約。`estimate` が `None` なら何も返さない。

    **`estimate` へ渡してよいのは、実際に画面へ出している推定だけである。**
    年齢ゲート（FR-038）が参考年齢を隠している推定を渡すと、隠したはずの判断が
    母集団の年齢から復元できてしまう（§10.11-2、§10.12-3）。

    `provider` は test 用の差し替え口である（`ui.reference_page` の `browser` と
    同じ役割）。

    **推定が無いときは provider にも触れない。** 参照データの読み込みは初回の解決まで
    遅らせたい——年齢・性別モデルを配置していない利用者や、参考年齢を表示していない
    セッションでは、537 行の CSV と sha256 検証がそもそも要らないためである。映像上の
    HUD はカメラ開始直後（まだ推定が無い状態）から 5 Hz で描き直されるので、ここが
    最初の tick で読み込みを起こすかどうかは実際に効く。
    """
    if estimate is None:
        return None
    resolved = provider if provider is not None else default_baseline_provider()
    return resolved.resolve(estimate)
