import pytest

from face_symmetry_lab.adapters.inference.mediapipe_landmarker import LandmarkDetectionError
from face_symmetry_lab.adapters.inference.runtime_errors import init_failure_detail


def test_a_missing_shared_library_names_the_packages_to_install() -> None:
    """The CI failure mode: MediaPipe's native lib links EGL/GLES, and a bare
    "could not initialize" left nothing to act on."""
    error = OSError("libGLESv2.so.2: cannot open shared object file: No such file or directory")

    detail = init_failure_detail(error)

    assert "libGLESv2.so.2" in detail
    assert "libegl1" in detail
    assert "libgles2" in detail


def test_any_other_failure_is_reported_verbatim() -> None:
    assert init_failure_detail(ValueError("壊れたモデルです")) == "壊れたモデルです"


def test_a_silent_exception_still_names_its_type() -> None:
    """An empty message must not produce a dangling sentence."""
    assert init_failure_detail(RuntimeError()) == "RuntimeError"


def test_the_detail_stays_a_single_line() -> None:
    """§5.3: a cause, not a stack trace."""
    detail = init_failure_detail(OSError("libEGL.so.1: cannot open shared object file"))

    assert "\n" not in detail


@pytest.mark.parametrize(
    "adapter_module",
    [
        "face_symmetry_lab.adapters.inference.mediapipe_landmarker",
        "face_symmetry_lab.adapters.inference.mediapipe_segmenter",
    ],
)
def test_both_mediapipe_adapters_report_the_cause(adapter_module: str) -> None:
    """Both wrap the same class of startup failure, so both must explain it."""
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(adapter_module))

    assert "init_failure_detail(error)" in source


def test_the_landmarker_error_carries_the_detail_through() -> None:
    detail = init_failure_detail(
        OSError("libGLESv2.so.2: cannot open shared object file: No such file or directory")
    )
    raised = LandmarkDetectionError(f"ランドマークモデルを初期化できません。{detail}")

    assert "libgles2" in str(raised)
