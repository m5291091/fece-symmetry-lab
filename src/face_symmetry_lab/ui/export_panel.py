"""両方の入力モードが共有するダウンロードパネル（FR-042, §9.6）。

Camera と Image で異なるのは文言と widget のキー、そしてレポートの中身だけで
ある。提示条件（記録が 1 件以上あるときだけ出す）はどちらでも同じ契約なので、
2 箇所へ分かれて食い違わないよう、ここにまとめて置く。

**書き出せるのは HTML レポート 1 種類だけである**（2026-08-16 に FR-017 の
CSV / JSON を撤回した）。レポートの中身は入力モードごとに違うため、この module は
組み立てを知らず、呼び出し側が渡す builder を呼ぶだけである。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import streamlit as st

from face_symmetry_lab.domain.export import SessionExport
from face_symmetry_lab.ui.i18n import t
from face_symmetry_lab.ui.report import report_filename

#: `include_figures` を受け取り、レポートの HTML を返す。図の符号化は呼び出し側の
#: 責任である（この module は cv2 も画像も知らない）。
ReportBuilder = Callable[[bool], str]

#: §15.1 / ADR-0009 (d): 図の埋め込みは**既定 ON**。人間の明示判断である
#: （2026-08-16）。安全側の既定を提案したが採られなかったので、代わりに操作の
#: 直前とレポート本体の両方へ注意を置くことで補っている。
FIGURE_TOGGLE_LABEL: str = "顔写真を含む図をレポートへ入れる"

_FIGURE_TOGGLE_HELP: Final[str] = (
    "ランドマークや対称化の図をレポートへ埋め込みます。数値と来歴は変わりません。"
)
_REPORT_DOWNLOAD_LABEL: Final[str] = "レポート（HTML）を保存"

#: `ui.report.REPORT_SHARING_NOTE` と**同じ文面**。あちらは別 module の
#: module-level 定数なので、この module から `t()` へそのまま渡せない
#: （`t()` の引数は同じモジュール内の literal でなければならない、
#: `tests/ui/test_i18n.py`）。値を複製し、`test_the_sharing_warning_is_shown_before_the_download`
#: が両者の内容が一致することを担保する。
_SHARING_NOTE: Final[str] = (
    "このファイルには顔写真が含まれます。保存後の共有先はアプリからは制御できません。"
)


def render_export_panel(
    bundle: SessionExport | None,
    *,
    heading: str,
    warning: str,
    key_prefix: str,
    show_record_count: bool = False,
    empty_notice: str | None = None,
    report_builder: ReportBuilder | None = None,
    report_mode: str = "session",
    report_has_figures: bool = True,
) -> None:
    """エクスポート節を描画する。データが無いときは download button を出さない。

    FR-042 は結果が 1 件以上あるときにのみエクスポートを提示する（FR-017 から
    引き継いだ条件）。そのため空の（あるいは存在しない）bundle では見出しも
    ボタンも描かない。利用者が操作できない節をページに出さないためである。

    ``empty_notice`` はその提示条件を緩めない。**ボタンの代わりに一行の説明を
    置くだけである。** これが無かった間、Camera の「エクスポート」タブは開始
    直後に何ひとつ描かれず、中身が真っ白になっていた——利用者から見れば
    「壊れている」と「まだ何も無い」を区別できない。文言は呼び出し側が持つ
    （Camera は記録が貯まる条件を、Image は写真を選ぶことを案内する）。
    """
    if bundle is None or not bundle.records:
        if empty_notice:
            st.info(empty_notice)
        return
    if report_builder is None:
        return

    st.divider()
    st.subheader(heading)
    if warning:
        st.caption(warning)
    # FR-020: 指標の版と config fingerprint が、この数値が生成された条件を特定する。
    revision = bundle.manifest.config_revisions[-1]
    count = f"{len(bundle.records)} 件 · " if show_record_count else ""
    st.caption(
        f"{count}metric {bundle.manifest.metric_version} · config {revision.fingerprint[:12]}…"
    )
    _render_report_download(
        bundle,
        builder=report_builder,
        key_prefix=key_prefix,
        mode=report_mode,
        has_figures=report_has_figures,
    )


def _render_report_download(
    bundle: SessionExport,
    *,
    builder: ReportBuilder,
    key_prefix: str,
    mode: str,
    has_figures: bool,
) -> None:
    """FR-042: 人が読む HTML レポート。**書き出せる唯一の成果物である。**

    ``has_figures`` が偽のモード（Camera）では、切り替えも注意も出さない。
    そのレポートは顔写真を持たないので、選ぶものも警告する対象も無い
    （2026-08-16）。

    図を持ちうるモード（Image）では、**注意は押す前に出す**。押した後に本体で
    伝えるだけでは遅い（ADR-0009 (d)）。図を切っているときは出さない——起こら
    ないことへの注意を出し続けると、注意そのものが読み流される。
    """
    include_figures = False
    if has_figures:
        include_figures = st.checkbox(
            t(FIGURE_TOGGLE_LABEL),
            value=True,
            key=f"{key_prefix}_report_figures",
            help=t(_FIGURE_TOGGLE_HELP),
        )
        if include_figures:
            st.caption(t(_SHARING_NOTE))
    st.download_button(
        t(_REPORT_DOWNLOAD_LABEL),
        data=builder(include_figures).encode("utf-8"),
        file_name=report_filename(bundle, mode=mode),
        mime="text/html",
        key=f"{key_prefix}_report",
    )
