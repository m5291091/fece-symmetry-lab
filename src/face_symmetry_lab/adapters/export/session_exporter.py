"""共通エクスポートスキーマの CSV・JSON 直列化（§9.3, §9.4）。"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Literal

from face_symmetry_lab.domain.export import ExportRecord, SessionExport


class SessionExportError(ValueError):
    """そのセッションにエクスポートできる数値記録が無い。"""


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_round_floats(item) for item in value]
    return value


def _require_records(bundle: SessionExport) -> None:
    if not bundle.records:
        raise SessionExportError("エクスポートできる結果がありません。")


def json_bytes(bundle: SessionExport) -> bytes:
    """manifest と records を、規格外の数値表記を出さずに直列化する。"""
    _require_records(bundle)
    payload = _round_floats(bundle.model_dump(mode="json"))
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ).encode("utf-8")


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list | tuple):
        return ";".join(str(item) for item in value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return _round_floats(value)


def csv_bytes(bundle: SessionExport) -> bytes:
    """同じ records を CSV として直列化する。配列はセミコロンで区切る。"""
    _require_records(bundle)
    stream = io.StringIO(newline="")
    fieldnames = list(ExportRecord.model_fields)
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for record in bundle.records:
        dumped = record.model_dump(mode="json")
        writer.writerow({name: _csv_value(dumped[name]) for name in fieldnames})
    return stream.getvalue().encode("utf-8")


def export_filename(bundle: SessionExport, format_name: Literal["csv", "json"]) -> str:
    short_id = str(bundle.manifest.session_id)[:8]
    return f"face-symmetry-session_{short_id}.{format_name}"
