"""両方の入力モードが共有するダウンロードパネル（FR-017 / FR-042, §9）。

Camera と Image で異なるのは文言と widget のキー、そしてレポートの中身だけで
ある。再現性の注記とダウンロードボタンはどちらでも同じ契約なので、2 箇所へ
分かれて食い違わないよう、ここにまとめて置く。

FR-042 で 3 つ目のボタン（人が読む HTML レポート）が加わった。**CSV / JSON は
1 列も変えていない**——台帳と読み物は目的が違うので併存させる（ADR-0009 (b)）。
レポートの中身は入力モードごとに違うため、この module は組み立てを知らず、
呼び出し側が渡す builder を呼ぶだけである。
"""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

from face_symmetry_lab.adapters.export.session_exporter import (
    csv_bytes,
    export_filename,
    json_bytes,
)
from face_symmetry_lab.domain.export import SessionExport
from face_symmetry_lab.ui.report import REPORT_SHARING_NOTE, report_filename

#: `include_figures` を受け取り、レポートの HTML を返す。図の符号化は呼び出し側の
#: 責任である（この module は cv2 も画像も知らない）。
ReportBuilder = Callable[[bool], str]

#: §15.1 / ADR-0009 (d): 図の埋め込みは**既定 ON**。人間の明示判断である
#: （2026-08-16）。安全側の既定を提案したが採られなかったので、代わりに操作の
#: 直前とレポート本体の両方へ注意を置くことで補っている。
FIGURE_TOGGLE_LABEL: str = "顔写真を含む図をレポートへ入れる"


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

    FR-017 は結果が 1 件以上あるときにのみエクスポートを提示する。そのため空の
    （あるいは存在しない）bundle では見出しもボタンも描かない。利用者が操作
    できない節をページに出さないためである。

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

    st.divider()
    st.subheader(heading)
    st.warning(warning)
    # FR-020: 指標の版と config fingerprint が、これらの数値が生成された条件を
    # 正確に特定する。
    revision = bundle.manifest.config_revisions[-1]
    count = f"{len(bundle.records)} 件 · " if show_record_count else ""
    st.caption(
        f"{count}metric {bundle.manifest.metric_version} · config {revision.fingerprint[:12]}…"
    )

    if report_builder is not None:
        _render_report_download(
            bundle,
            builder=report_builder,
            key_prefix=key_prefix,
            mode=report_mode,
            has_figures=report_has_figures,
        )

    csv_column, json_column = st.columns(2)
    csv_column.download_button(
        "CSV を保存",
        data=csv_bytes(bundle),
        file_name=export_filename(bundle, "csv"),
        mime="text/csv",
        key=f"{key_prefix}_csv",
    )
    json_column.download_button(
        "JSON を保存",
        data=json_bytes(bundle),
        file_name=export_filename(bundle, "json"),
        mime="application/json",
        key=f"{key_prefix}_json",
    )


def _render_report_download(
    bundle: SessionExport,
    *,
    builder: ReportBuilder,
    key_prefix: str,
    mode: str,
    has_figures: bool,
) -> None:
    """FR-042: 人が読む HTML レポート。

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
            FIGURE_TOGGLE_LABEL,
            value=True,
            key=f"{key_prefix}_report_figures",
            help="ランドマークや対称化の図をレポートへ埋め込みます。数値と来歴は変わりません。",
        )
        if include_figures:
            st.caption(REPORT_SHARING_NOTE)
    st.download_button(
        "レポート（HTML）を保存",
        data=builder(include_figures).encode("utf-8"),
        file_name=report_filename(bundle, mode=mode),
        mime="text/html",
        key=f"{key_prefix}_report",
    )
