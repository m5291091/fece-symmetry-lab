"""Shared CSV / JSON download panel for both input modes (FR-017, §9).

Camera and Image differ only in wording and widget keys; the reproducibility
caption and the two download buttons are the same contract either way, so
they live here rather than drifting apart in two places.
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
) -> None:
    """Render the export section, or nothing at all when there is no data.

    FR-017 only offers an export once at least one result exists, so an
    empty (or absent) bundle renders nothing — including no heading, so the
    page never shows a section the user cannot act on.
    """
    if bundle is None or not bundle.records:
        return

    st.divider()
    st.subheader(heading)
    st.warning(warning)
    # FR-020: the metric version and config fingerprint identify the exact
    # conditions these numbers were produced under.
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
