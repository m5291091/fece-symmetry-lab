"""「このローカル runtime が起動しなかった」を表す共通文言（§14.4）。

2 つの MediaPipe adapter は同じ種類の失敗を包んでおり、そこで生成される
メッセージが、ほとんどの読み手にとって唯一目にするものである。stack trace は
利用者へは表示されず、CI では連鎖した traceback の下に埋もれる。
"""

from __future__ import annotations


def init_failure_detail(error: BaseException) -> str:
    """初期化に失敗した runtime について、1 行で対処可能な原因を示す。

    §14.4 は、エラーが「何が起きたか」と「何ができるか」を述べることを求める。
    単なる「初期化できません」では、まっさらな Linux 機や CI runner で最も多い
    原因が隠れてしまう。MediaPipe のネイティブライブラリは EGL/GLES へリンク
    するため、loader が `.so` の不足を報告するが、メッセージの他の部分ではそれを
    説明できない。ここは 1 文に留める。stack trace ではない（§5.3）。
    """
    text = str(error).strip()
    if isinstance(error, OSError) and "cannot open shared object file" in text:
        return (
            f"必要な共有ライブラリが見つかりません（{text}）。"
            "Linux では libegl1 と libgles2 を導入してください。"
        )
    return text or type(error).__name__
