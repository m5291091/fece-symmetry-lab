"""Shared wording for "this local runtime would not start" (§14.4).

Both MediaPipe adapters wrap the same class of failure, and the message they
produce is the only thing most readers ever see — a stack trace is not shown
to the user and, in CI, is buried under a chained traceback.
"""

from __future__ import annotations


def init_failure_detail(error: BaseException) -> str:
    """A one-line, actionable cause for a runtime that failed to initialize.

    §14.4 asks an error to say what went wrong and what can be done about it.
    A bare "could not initialize" hides the most common cause on a fresh Linux
    box or CI runner: MediaPipe's native library links against EGL/GLES, so the
    loader reports a missing `.so` that nothing else in the message explains.
    This stays one sentence — it is not a stack trace (§5.3).
    """
    text = str(error).strip()
    if isinstance(error, OSError) and "cannot open shared object file" in text:
        return (
            f"必要な共有ライブラリが見つかりません（{text}）。"
            "Linux では libegl1 と libgles2 を導入してください。"
        )
    return text or type(error).__name__
