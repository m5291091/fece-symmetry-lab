"""§14.3g の取り込みバッファ単体のテスト。

pipeline を通さずに検証する。「要求されるまで 1 枚もコピーしない」「ちょうど
1 回だけ消費される」という契約は、このオブジェクト自身の性質である。
"""

import numpy as np

from face_symmetry_lab.application.frame_capture import CaptureBuffer, CapturedFrame
from face_symmetry_lab.domain.landmarks import build_landmark_frame


def _frame(face_count: int = 1):
    points = ((0.5, 0.5),) if face_count == 1 else ()
    return build_landmark_frame(frame_id=1, face_count=face_count, points=points)


def _image(value: int = 30) -> np.ndarray:
    return np.full((8, 8, 3), value, dtype=np.uint8)


def test_nothing_is_kept_until_a_capture_is_requested() -> None:
    """要求が無い限りコピーは発生しない。これがこの機能を無償に保つ条件である。"""
    buffer = CaptureBuffer()

    buffer.store(_image(), _frame())

    assert buffer.take() is None
    assert buffer.pending is False


def test_a_requested_capture_is_fulfilled_by_a_single_face_frame() -> None:
    buffer = CaptureBuffer()
    buffer.request()

    assert buffer.pending is True
    buffer.store(_image(), _frame())

    captured = buffer.take()
    assert isinstance(captured, CapturedFrame)
    assert buffer.pending is False


def test_the_held_frame_is_consumed_exactly_once() -> None:
    """§10.4: 生画像は UI が使うまでの間しかメモリへ残さない。"""
    buffer = CaptureBuffer()
    buffer.request()
    buffer.store(_image(), _frame())

    assert buffer.take() is not None
    assert buffer.take() is None


def test_a_multi_face_frame_never_fulfils_the_request() -> None:
    """対称化には正中線が 1 本必要で、要求だけ消費して後で失敗させない。"""
    buffer = CaptureBuffer()
    buffer.request()

    buffer.store(_image(), _frame(face_count=2))

    assert buffer.pending is True
    assert buffer.take() is None


def test_the_image_is_copied_because_the_caller_reuses_its_buffer() -> None:
    """呼び出し元は WebRTC のバッファを所有し、次のフレームで再利用する。"""
    buffer = CaptureBuffer()
    buffer.request()
    image = _image(value=30)
    buffer.store(image, _frame())

    image[:] = 200  # 呼び出し元がバッファを使い回した状況を再現する

    captured = buffer.take()
    assert captured is not None
    assert int(captured.rgb_image.max()) == 30


def test_clearing_drops_both_the_request_and_the_held_frame() -> None:
    """§4.4, §10.4: 破棄後に顔由来の画像を一切残さない。"""
    buffer = CaptureBuffer()
    buffer.request()
    buffer.store(_image(), _frame())

    buffer.clear()

    assert buffer.pending is False
    assert buffer.take() is None


def test_clearing_an_unfulfilled_request_disarms_it() -> None:
    buffer = CaptureBuffer()
    buffer.request()

    buffer.clear()

    assert buffer.pending is False
