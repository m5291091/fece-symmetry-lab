#!/bin/sh
# PreToolUse(Bash) hook: git commit / git push を機械的にブロックする。
#
# CLAUDE.md §7 の「commit と push は必ず人間が実行する」を、文章ではなく
# 実行時に強制するためのフック。stdin から PreToolUse の JSON を受け取り、
# tool_input.command が git commit / git push に該当したら exit 2 で止める。
#
# 判定は安全側（過剰ブロック側）に倒してある。誤爆したら人間が実行すればよいが、
# 見逃すとエージェントが勝手に履歴を書き換えるため。

set -u

payload=$(cat)

# tool_input.command の取り出し。jq → python3 → 素の grep の順に縮退する。
command_text=""
if command -v jq >/dev/null 2>&1; then
	command_text=$(printf '%s' "$payload" | jq -r '.tool_input.command // ""' 2>/dev/null)
elif command -v python3 >/dev/null 2>&1; then
	command_text=$(printf '%s' "$payload" | python3 -c 'import json,sys
try:
    print(json.load(sys.stdin).get("tool_input", {}).get("command", ""))
except Exception:
    pass' 2>/dev/null)
else
	# JSON parser が無い環境。payload 全体を対象にする（description 等での誤爆を許容）。
	command_text="$payload"
fi

[ -n "$command_text" ] || exit 0

# `git ... commit` / `git ... push` を 1 コマンド境界（; & |）内で探す。
# `git -C /path commit` や `foo && git push` も拾う。
if printf '%s' "$command_text" \
	| tr '\n' ' ' \
	| grep -Eq '(^|[^[:alnum:]_.-])git[^;&|]*[[:space:]](commit|push)([[:space:]]|$)'; then
	cat >&2 <<'EOF'
[blocked] コミット/プッシュは人間が行います。

このリポジトリでは git commit / git push をエージェントが実行しません
（CLAUDE.md §7）。実行せず、CLAUDE.md §8 の実装後レポートの「📝 コミット案」
として提示してください。対象ファイルは省略せず全て列挙すること。

調査用の git status / git diff / git log はそのまま使えます。
EOF
	exit 2
fi

exit 0
