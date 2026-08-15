"""両方の入力モードが共有する CSV / JSON ダウンロードパネル（FR-017, §9）。

Camera と Image で異なるのは文言と widget のキーだけである。再現性の注記と
2 つのダウンロードボタンはどちらでも同じ契約なので、2 箇所へ分かれて食い違わ
ないよう、ここにまとめて置く。
"""

from __future__ import annotations

import streamlit as st

from face_symmetry_lab.adapters.export.session_exporter import (
    csv_bytes,
    export_filename,
    json_bytes,
)
from face_symmetry_lab.domain.export import SessionExport


def render_export_panel(
    bundle: SessionExport | None,
    *,
    heading: str,
    warning: str,
    key_prefix: str,
    show_record_count: bool = False,
    empty_notice: str | None = None,
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
