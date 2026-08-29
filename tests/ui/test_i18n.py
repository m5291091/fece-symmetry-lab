"""`ui.i18n`: UI 表示言語の切替（FR-043、AC-I18N-01〜03、ADR-0010）。

**訳表そのものを AST 走査で検査する。** 実行結果の描画テストではなく、
`translations.EN` という「訳語の在庫」に対して規約を確かめる（ADR-0010 (d)）。

守る規約は 2 つ。
1. `t(...)` へ渡す原文は「文字列リテラル」「同じモジュールの module-level で
   文字列リテラルへ束縛された名前」「同じモジュールの module-level の
   文字列辞書への添字（`t(PART_LABELS[part])`）」のいずれかでなければ
   ならない。名前と辞書は `from face_symmetry_lab.ui.X import NAME` で
   持ち込まれたものでもよい（import 元まで辿って解決する）。これは
   ADR-0010 (a) の前提——原文を鍵にするには、鍵が実行のたびに変わらない
   ことが要る——を機械的に守らせるための規約である。辞書への添字は
   **値のすべて**が訳表に在ることを要求する。
2. 解決できたすべての原文が `EN` に載っていなければならない（訳し忘れの検出。
   fail-open（ADR-0010 (b)）が抜けを隠す経路にならないようにする）。

3a の時点では `translations/_app.py` 以外の表は空である。そのため多くの
assertion は「まだ何も書かれていないので空集合どうしが一致する」形で緑になる
——3b 以降で `t()` の呼び出しと訳語を増やすたびに、ここが初めて意味のある
検査として効き始める。
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from string import Formatter
from typing import Final

from streamlit.testing.v1 import AppTest

from face_symmetry_lab.ui.i18n import t
from face_symmetry_lab.ui.readings import BASELINE_CAVEAT
from face_symmetry_lab.ui.scores import READING_LIMITS, SCORE_SCALE_CAPTION
from face_symmetry_lab.ui.smoothing_panel import SMOOTHING_SCOPE_NOTE
from face_symmetry_lab.ui.translations import EN, RUNTIME_KEYS
from face_symmetry_lab.ui.translations import _errors as _errors_table

UI_ROOT: Final[Path] = Path(__file__).parents[2] / "src" / "face_symmetry_lab" / "ui"
SRC_ROOT: Final[Path] = UI_ROOT.parent


# --- AST 走査: `t(...)` 呼び出しの収集 ---------------------------------------


class _Call:
    __slots__ = ("lineno", "path", "resolved")

    def __init__(self, path: Path, lineno: int, resolved: str | None) -> None:
        self.path = path
        self.lineno = lineno
        self.resolved = resolved


def _ui_source_files() -> list[Path]:
    """`i18n.py` と `translations/` を除く `ui/**/*.py`。"""
    return sorted(
        path
        for path in UI_ROOT.rglob("*.py")
        if path.name != "i18n.py" and "translations" not in path.relative_to(UI_ROOT).parts
    )


def _module_level_string_bindings(tree: ast.Module) -> dict[str, str]:
    """module-level で文字列リテラルへ束縛された名前 -> その文字列。

    暗黙の隣接結合（`"a" "b"`）は ast がすでに 1 つの `Constant` へまとめる
    ため、追加の処理なしに拾える。f-string（`JoinedStr`）は対象外。
    """
    bindings: dict[str, str] = {}
    for node in tree.body:
        targets: list[ast.expr]
        value: ast.expr | None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = value.value
    return bindings


def _module_level_string_dict_bindings(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    """module-level の「文字列 -> 文字列」辞書 -> その値の並び。

    `PART_LABELS[name]` のような添字は、実行時にどの値が来るかを AST では
    決められない。**そこで「どれか 1 つ」ではなく「ぜんぶ」を要求する**
    ——辞書の値のすべてが訳表に在ることを確かめれば、どの枝を通っても訳が
    見つかることが保証できる。値に文字列以外が混ざる辞書は対象にしない。
    """
    bindings: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        targets: list[ast.expr]
        value: ast.expr | None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not isinstance(value, ast.Dict):
            continue
        items = [
            item.value
            for item in value.values
            if isinstance(item, ast.Constant) and isinstance(item.value, str)
        ]
        if len(items) != len(value.values) or not items:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bindings[target.id] = tuple(items)
    return bindings


def _ui_imports(tree: ast.Module) -> dict[str, str]:
    """`from face_symmetry_lab.ui.X import NAME` -> {ローカル名: "X"}。

    **文言定数はモジュールをまたいで共有される。** `ui.symmetrization_panel` の
    見出しを `ui.image_section` が使い、`ui.age_gate_notice` の固定文言を
    `ui.camera` が使う——どちらも「同じ文言を 2 か所へ書かない」ための共有であり、
    その形を i18n のために崩す理由は無い。したがって import 元まで辿って解決する。
    """
    imports: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        prefix = "face_symmetry_lab.ui."
        if not node.module.startswith(prefix):
            continue
        module = node.module[len(prefix) :]
        for alias in node.names:
            imports[alias.asname or alias.name] = f"{module}:{alias.name}"
    return imports


def _module_bindings(module: str) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    path = UI_ROOT / f"{module.replace('.', '/')}.py"
    if not path.is_file():
        return {}, {}
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return _module_level_string_bindings(tree), _module_level_string_dict_bindings(tree)


def _scan_translation_calls(path: Path) -> list[_Call]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bindings = _module_level_string_bindings(tree)
    dict_bindings = _module_level_string_dict_bindings(tree)
    for local, origin in _ui_imports(tree).items():
        module, _, original = origin.partition(":")
        imported_strings, imported_dicts = _module_bindings(module)
        if original in imported_strings:
            bindings.setdefault(local, imported_strings[original])
        if original in imported_dicts:
            dict_bindings.setdefault(local, imported_dicts[original])
    calls: list[_Call] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "t"
        ):
            continue
        if len(node.args) != 1 or node.keywords:
            calls.append(_Call(path, node.lineno, None))
            continue
        arg = node.args[0]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            calls.append(_Call(path, node.lineno, arg.value))
        elif isinstance(arg, ast.Name) and arg.id in bindings:
            calls.append(_Call(path, node.lineno, bindings[arg.id]))
        elif (
            isinstance(arg, ast.Subscript)
            and isinstance(arg.value, ast.Name)
            and arg.value.id in dict_bindings
        ):
            # `t(PART_LABELS[part])`: どの枝を通っても訳が在ることを要求する。
            for text in dict_bindings[arg.value.id]:
                calls.append(_Call(path, node.lineno, text))
        else:
            calls.append(_Call(path, node.lineno, None))
    return calls


def _all_translation_calls() -> list[_Call]:
    calls: list[_Call] = []
    for path in _ui_source_files():
        calls.extend(_scan_translation_calls(path))
    return calls


# --- 0: 訳を import 時に固めない --------------------------------------------


def _module_level_translation_calls() -> list[tuple[str, int]]:
    """関数の外（module-level）で呼ばれている `t()` / `translate_message()`。"""
    found: list[tuple[str, int]] = []
    for path in _ui_source_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        def walk(node: ast.AST, inside_function: bool, path: Path = path) -> None:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                inside_function = True
            if (
                not inside_function
                and isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"t", "translate_message"}
            ):
                found.append((str(path.relative_to(UI_ROOT)), node.lineno))
            for child in ast.iter_child_nodes(node):
                walk(child, inside_function)

        walk(tree, False)
    return sorted(found)


def test_no_translation_is_frozen_at_import_time() -> None:
    """FR-043: `t()` は描画のたびに呼ぶ。module-level で呼ばない。

    module-level の `t()` は import した時点の言語で結果が固定される。
    `ui/app.py` は entry script として毎回再実行されるので production では
    偶然動くが、module として import されたときだけ古い言語のまま残る——
    **同じ file が読まれ方によって 2 通りに振る舞う。** 実際に
    `_INPUT_MODES` の 2 語がこれで英語モードでも日本語のまま出ていた
    （2026-08-29 に発見）。

    罠: 定数へ `t()` を書くと、`AppTest` でも本番でも「たまたま通る」経路が
    あるため、描画結果を見るテストだけでは捕まらない。
    """
    assert _module_level_translation_calls() == [], (
        "module-level の t() は import 時に訳が固定されます。"
        "描画のたびに呼ばれる関数の中へ移してください。"
    )


# --- 1: 訳し忘れの検出 -------------------------------------------------------


def _requires_a_translation(text: str) -> bool:
    """原文がすでに英語（ASCII のみ）なら、訳表に載せる意味が無い。

    `t("Analyze")` のように、原文自体がすでに英語の `st.Page` タイトルへも
    `t()` を通すことがある（3b 以降で他の呼び出しと同じ書き方に揃えるため）。
    こうした原文は fail-open のままで訳と同じ結果になるため、訳表へ載せる
    義務を課さない。
    """
    return not text.isascii()


def test_every_translated_string_has_an_english_entry() -> None:
    resolved = {call.resolved for call in _all_translation_calls() if call.resolved is not None}
    missing = sorted(text for text in resolved if _requires_a_translation(text) and text not in EN)
    assert missing == [], f"訳表に無い原文があります: {missing}"


# --- 2: 解決できない `t(...)` の検出 -----------------------------------------


def test_translation_calls_use_a_resolvable_literal() -> None:
    """罠: `t(f"平均 {x}%")` は実行時には動くが、鍵が実行のたび変わるので永久に訳せない。"""
    unresolved = [
        (str(call.path.relative_to(UI_ROOT)), call.lineno)
        for call in _all_translation_calls()
        if call.resolved is None
    ]
    assert unresolved == [], f"リテラルへ解決できない t(...) 呼び出しがあります: {unresolved}"


# --- 3: 孤児判定 --------------------------------------------------------------


def test_no_orphan_english_entries() -> None:
    """罠: 文言を消したのに訳表だけ残る状態を塞ぐ。

    `_errors.py`（`RUNTIME_KEYS`）は `t(str(error))` のように実行時の値を包む
    ため、静的な `t("...")` からは参照されない。孤児判定の対象から除く。
    """
    referenced = {call.resolved for call in _all_translation_calls() if call.resolved is not None}
    orphans = sorted(EN.keys() - referenced - RUNTIME_KEYS)
    assert orphans == [], f"どの t() からも参照されない訳語があります: {orphans}"


# --- 4: プレースホルダの整合 ---------------------------------------------------


def _placeholder_names(text: str) -> set[str]:
    return {field for _, field, _, _ in Formatter().parse(text) if field}


def test_placeholders_match_between_the_source_and_the_translation() -> None:
    """罠: 訳文でプレースホルダ名を変えると `.format()` が実行時に KeyError になる。"""
    mismatches = {
        original: (_placeholder_names(original), _placeholder_names(translated))
        for original, translated in EN.items()
        if _placeholder_names(original) != _placeholder_names(translated)
    }
    assert mismatches == {}


# --- 5: 禁止語の走査（AC-I18N-02） --------------------------------------------

#: §5.3・AC-AGEGATE-04・AC-BASELINE-04 と同じ走査を訳表そのものへ課す
#: （ADR-0010 (d)）。訳語で禁止を回避できる経路を作らない。
_FORBIDDEN_ENGLISH_WORDING: Final[tuple[str, ...]] = (
    "perfect",
    "ideal",
    "flaw",
    "defect",
    "distortion",
    "abnormal",
    "corrected",
    "enhanced",
    "improved",
    "beautified",
    "minor",
    "underage",
    "under 18",
    "not an adult",
    "elderly",
    "senior",
    "percentile",
    "rank",
    "ranking",
    "quantile",
    "above average",
    "below average",
    # §5.3 の「正常・異常」。`\b` 境界なので `normalisation`（撮影補正の
    # 説明で正当に使う語）には当たらない。
    "normal",
    # AC-BASELINE-04 の「合格」。品質ゲートは "meets the measurement
    # conditions" と述べるので、この語が要る場面は無い。
    "pass",
    "passes",
    "passed",
    "score better",
)

#: AC-AGEGATE-04 の「〇〇歳に見えます」。単語の列挙では捕まらないので
#: 正規表現で見る。属性の断定そのものを塞ぐ（§10.11-2）。
_FORBIDDEN_ENGLISH_PATTERNS: Final[tuple[str, ...]] = (
    r"\byou\s+(?:are|look|appear)\b[^.]*\b\d+\b",
    r"\blooks?\s+\d+\s+years?\s+old\b",
)


def test_the_english_table_carries_no_forbidden_wording() -> None:
    violations = [
        (original, word)
        for original, translated in EN.items()
        for word in _FORBIDDEN_ENGLISH_WORDING
        if re.search(rf"\b{re.escape(word)}\b", translated, flags=re.IGNORECASE)
    ]
    violations += [
        (original, pattern)
        for original, translated in EN.items()
        for pattern in _FORBIDDEN_ENGLISH_PATTERNS
        if re.search(pattern, translated, flags=re.IGNORECASE)
    ]
    assert violations == [], f"訳表に禁止語が入っています: {violations}"


# --- 6/7: fail-open と実際の切り替え ------------------------------------------


def test_the_default_language_returns_the_source_text_unchanged() -> None:
    assert t("参照統計") == "参照統計"


def test_switching_to_english_actually_changes_the_text() -> None:
    """罠: `t()` が常に引数を返すだけの実装は、6 番だけでは通ってしまう。"""
    at = AppTest.from_string(
        "import streamlit as st\n"
        "from face_symmetry_lab.ui.i18n import t\n"
        "st.session_state['translated'] = t('参照統計')\n"
    )
    at.session_state["ui_language"] = "en"
    at.run()

    assert not at.exception
    assert at.session_state["translated"] == "Reference statistics"


# --- 8: 英訳側も同じ長さ上限を守ること -----------------------------------------


def test_the_english_translations_obey_the_same_length_limits_as_the_japanese() -> None:
    """既存の日本語 test が縛っている桁数を、英訳側にも同じだけ課す。

    3a の時点では該当する表がまだ空なので、`EN.get(original, original)` は
    原文（日本語）へ fail-open し、この assertion は既存の日本語制約の
    再確認にとどまる。3b 以降で英訳が入った時点から意味のある検査になる。
    """
    assert len(EN.get(BASELINE_CAVEAT, BASELINE_CAVEAT)) < 60
    assert len(EN.get(SCORE_SCALE_CAPTION, SCORE_SCALE_CAPTION)) < 60
    assert len(EN.get(READING_LIMITS[0], READING_LIMITS[0])) < 100
    assert len(EN.get(SMOOTHING_SCOPE_NOTE, SMOOTHING_SCOPE_NOTE)) < 60


# --- 9: 例外文言が UI 境界の訳表に在ること（FR-043） ---------------------------

#: UI が `st.error` / `st.info` / `st.warning` へそのまま渡す例外型（`ui/*.py`
#: の外で定義される）。`ui.i18n.translate_message` はこれらの `str(error)` を
#: 境界で訳す。ここでは「訳すべき原文の在庫」を静的に洗い出し、
#: `translations._errors.EN` に漏れが無いことを確かめる。
_UI_BOUNDARY_EXCEPTIONS: Final[frozenset[str]] = frozenset(
    {
        "ImageInputError",
        "LandmarkDetectionError",
        "MetricConfigError",
        "SymmetrizationError",
        "WarpMeshError",
        "DemographicInferenceError",
        "ModelStoreError",
        "ReferenceDatasetError",
    }
)


def _static_exception_literals() -> set[str]:
    """`domain` / `application` / `adapters` の `raise <Exc>("...")` のうち、

    引数が静的な文字列リテラル（f-string や `str(error)` のような実行時の
    組み立てではない）であるものの原文集合。動的な組み立ては実行のたびに
    文字列が変わるため鍵として引けず、訳せないまま残る（許容範囲、
    `translations/_errors.py` の module docstring）。ここで対象にするのは
    「訳せるはずなのに訳表へ入れ忘れた」静的な文言だけである。
    """
    literals: set[str] = set()
    for sub in ("domain", "application", "adapters"):
        for path in sorted((SRC_ROOT / sub).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Raise) or not isinstance(node.exc, ast.Call):
                    continue
                func = node.exc.func
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    name = func.attr
                else:
                    name = None
                if name not in _UI_BOUNDARY_EXCEPTIONS or not node.exc.args:
                    continue
                arg = node.exc.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    literals.add(arg.value)
    return literals


def test_static_exception_literals_have_an_error_boundary_translation() -> None:
    """罠: 静的な例外文言を新設したのに `_errors.EN` への追加を忘れると、

    英語表示のときだけそこが日本語のまま残る（AC-I18N-01 の穴）。この走査が
    無いと、その抜けは画面を実際に壊してみるまで分からない。
    """
    missing = sorted(_static_exception_literals() - set(_errors_table.EN))
    assert missing == [], f"_errors.EN に無い例外文言があります: {missing}"
