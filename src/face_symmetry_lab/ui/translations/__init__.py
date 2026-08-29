"""英語の訳表（FR-043）。鍵は日本語の原文そのもの（ADR-0010 (a)）。

**同じ原文に違う訳が付くことを許さない。** 訳が食い違ったまま両方生き残ると、
どちらが効くかが import 順で決まり、画面によって訳が変わる。

表をモジュールごとに分けるのは、複数人が同時に書けるようにするためである
（`ui/i18n.py` の module docstring、および `docs/adr/0010-ui-display-language.md`
を参照）。
"""

from __future__ import annotations

from typing import Final

from . import _app, _camera, _errors, _hud, _image, _pages, _report, _scores, _shared

_TABLES: Final = (
    _app.EN,
    _camera.EN,
    _errors.EN,
    _hud.EN,
    _image.EN,
    _pages.EN,
    _report.EN,
    _scores.EN,
    _shared.EN,
)


def _merge() -> dict[str, str]:
    """表を 1 つへ畳む。**同じ鍵に違う訳が付いていたらその場で落とす。**

    同じ日本語が 2 つ以上の `ui/*.py` に現れることは実際にある（「詳細結果」
    「平均」など）。それぞれの表が同じ訳を持つのは害が無いので許す。害になるのは
    訳が食い違ったまま両方生き残ることで、そのときどちらが効くかは import 順で
    決まり、画面によって訳が変わる。したがって**値が違うときだけ**拒む。
    """
    merged: dict[str, str] = {}
    for table in _TABLES:
        conflicts = sorted(key for key, value in table.items() if merged.get(key, value) != value)
        if conflicts:
            raise RuntimeError(f"同じ原文に違う訳が付いています: {conflicts}")
        merged.update(table)
    return merged


EN: Final[dict[str, str]] = _merge()

#: `_errors.py` の鍵。UI 境界で `t(str(error))` として実行時に引かれるため、
#: `t("...")` という静的な呼び出しからは参照されない。`tests/ui/test_i18n.py`
#: の孤児判定はこれを除外の対象として使う。
RUNTIME_KEYS: Final[frozenset[str]] = frozenset(_errors.EN.keys())
