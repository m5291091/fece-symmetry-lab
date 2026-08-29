"""Image 入力モード一式（アップロードした写真 1 枚の分析と表示）。

`ui.camera` が Camera 入力モードに対して持つのと同じ位置付けであり、`ui.app`
から分けたのはそのためである。以前は `ui.app` が「同意 → 入力方法の選択 →
分析」というページ構成と、Image モードの中身（アップロード検証・landmarker の
セッション保持・年齢性別の自動推論・4 つのタブ）の両方を持っていた。ページ
構成は 2 つのモードの上位にあるものなので、片方のモードと同じ file に置くと
Camera と Image が非対称に見える。

§5.1 のとおり Camera と同じ並びにする——画像を最上部へ置き、その隣に見出し
結果の区画を持たず（数値は画像上の吹き出しが出す、§14.3e）、詳細結果・
左右対称との比較・参考年齢・エクスポートは下のタブへ 1 つずつ入れる。

**状態は Camera 側と分ける。** landmarker も recorder も、Camera が持つものとは
別のセッションキーに置く（`ui.camera` の pipeline とは寿命が違う）。入力モード
を切り替えたときは `release_image_landmarker` で明示的に解放する（FR-004、§10.4）。
"""

from __future__ import annotations

import time
from hashlib import sha256
from pathlib import Path
from typing import Final

import numpy as np
import streamlit as st

from face_symmetry_lab import __version__
from face_symmetry_lab.adapters.imaging.piecewise_warp import OpenCvPiecewiseWarper
from face_symmetry_lab.adapters.inference.mediapipe_landmarker import (
    LandmarkDetectionError,
    MediaPipeImageLandmarker,
)
from face_symmetry_lab.adapters.inference.opencv_demographic_estimator import (
    DemographicInferenceError,
    OpenCvDemographicEstimator,
)
from face_symmetry_lab.adapters.io.metric_config import (
    SYMMETRY_3D_METRIC_CONFIG_PATH,
    load_metric_config,
)
from face_symmetry_lab.adapters.io.model_store import LOCK_PATH, file_sha256
from face_symmetry_lab.application.age_gate import (
    AgeGateOutcome,
    AgeGateService,
    evaluate_age_gate_for_frame,
)
from face_symmetry_lab.application.image_input import (
    ImageInputError,
    decode_uploaded_image,
    extract_face_crop,
)
from face_symmetry_lab.application.session import SessionRecorder
from face_symmetry_lab.application.still_image import (
    STILL_IMAGE_THRESHOLDS,
    StillImageAnalysis,
    analyze_still_image,
)
from face_symmetry_lab.application.symmetrization import SymmetrizationView, symmetrize_frame
from face_symmetry_lab.domain.export import InputMode
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.metrics import PART_NAMES, MetricConfig, MetricConfigError
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.domain.smoothing import DEFAULT_TAU_SECONDS
from face_symmetry_lab.domain.symmetrization import SymmetrizationError
from face_symmetry_lab.ui.age_gate_notice import (
    age_gate_notice,
    resolve_reference_age_view,
)
from face_symmetry_lab.ui.baseline import resolve_baseline
from face_symmetry_lab.ui.demographic_panel import render_demographic_panel
from face_symmetry_lab.ui.detail_panel import render_detail_panel
from face_symmetry_lab.ui.export_panel import render_export_panel
from face_symmetry_lab.ui.i18n import t, translate_message
from face_symmetry_lab.ui.model_setup import (
    build_age_gate_service,
    landmark_asset,
    resolve_ready_demographic,
    resolve_ready_landmarker,
)
from face_symmetry_lab.ui.readings import (
    gate_color,
    resolve_equal_weight_distance,
    resolve_overall_distance,
    resolve_part_distance_3d,
)
from face_symmetry_lab.ui.report import build_image_report
from face_symmetry_lab.ui.report_figures import build_report_figures
from face_symmetry_lab.ui.result_tabs import ResultTabs, render_result_tabs
from face_symmetry_lab.ui.scores import render_tracking_status
from face_symmetry_lab.ui.still_overlay import render_image_with_overlay
from face_symmetry_lab.ui.symmetrization_panel import (
    SYMMETRIZATION_HEADING,
    render_symmetrization_view,
    render_warp_checkbox,
)

ESTIMATOR_STATE_KEY = "demographic_estimator"
ESTIMATOR_PATH_KEY = "demographic_estimator_paths"
RESULT_STATE_KEY = "demographic_estimate"
IMAGE_LANDMARKER_STATE_KEY = "image_landmarker"
IMAGE_LANDMARKER_PATH_KEY = "image_landmarker_path"
IMAGE_RECORDER_STATE_KEY = "image_session_recorder"
IMAGE_RECORDED_INPUT_KEY = "image_recorded_input"
#: H2: 「この画像に対して年齢・性別の推論を既に試みたか」を示す fingerprint。
#: `IMAGE_RECORDED_INPUT_KEY` が landmark 記録の重複を防ぐのと同じ方式で、
#: 同じ写真に対する再実行のたびに ONNX 推論を走らせないためのキャッシュ鍵。
IMAGE_DEMOGRAPHIC_INPUT_KEY = "image_demographic_input"
#: H2: 直近の推定試行が失敗した理由。`RESULT_STATE_KEY` と排他的に持つ。
DEMOGRAPHIC_ERROR_KEY = "image_demographic_error"
#: FR-038: セッション中 1 度だけ構築する年齢ゲートの use case（ONNX セッションの
#: 再生成を避ける、`ESTIMATOR_STATE_KEY` と同じ理由）。
AGE_GATE_SERVICE_STATE_KEY = "image_age_gate_service"

_IMAGE_STATUS_MESSAGES: Final[dict[TrackingStatus, str]] = {
    TrackingStatus.NO_FACE: (
        "顔を検出できませんでした。明るさ、顔の大きさ、向きを確認してください。"
    ),
    TrackingStatus.SINGLE_FACE: "1 名の顔を検出しました。",
    TrackingStatus.MULTI_FACE: "2 名以上が写っています。1 名だけの写真を使用してください。",
    TrackingStatus.ERROR: "この画像を処理できませんでした。",
}


# --- セッション状態 -----------------------------------------------------------


def clear_session_result() -> None:
    """写真を差し替えたときに、前の写真から導いた状態をすべて捨てる。"""
    st.session_state.pop(RESULT_STATE_KEY, None)
    st.session_state.pop(IMAGE_RECORDER_STATE_KEY, None)
    st.session_state.pop(IMAGE_RECORDED_INPUT_KEY, None)
    st.session_state.pop(IMAGE_DEMOGRAPHIC_INPUT_KEY, None)
    st.session_state.pop(DEMOGRAPHIC_ERROR_KEY, None)


def release_image_landmarker() -> None:
    """静止画用の landmarker を解放する（FR-004, §10.4）。"""
    landmarker = st.session_state.pop(IMAGE_LANDMARKER_STATE_KEY, None)
    st.session_state.pop(IMAGE_LANDMARKER_PATH_KEY, None)
    if isinstance(landmarker, MediaPipeImageLandmarker):
        landmarker.close()


def _session_estimator(onnx_path: Path, model_version: str) -> OpenCvDemographicEstimator:
    onnx_stat = onnx_path.stat()
    path_fingerprint = (
        f"{onnx_path.resolve()}:{onnx_stat.st_size}:{onnx_stat.st_mtime_ns}:{model_version}"
    )
    if (
        st.session_state.get(ESTIMATOR_PATH_KEY) != path_fingerprint
        or ESTIMATOR_STATE_KEY not in st.session_state
    ):
        st.session_state[ESTIMATOR_STATE_KEY] = OpenCvDemographicEstimator(
            onnx_path=onnx_path,
            model_version=model_version,
        )
        st.session_state[ESTIMATOR_PATH_KEY] = path_fingerprint
    estimator = st.session_state[ESTIMATOR_STATE_KEY]
    if not isinstance(estimator, OpenCvDemographicEstimator):
        raise DemographicInferenceError("年齢・性別推定セッションを初期化できません。")
    return estimator


def _session_image_landmarker(model_path: Path) -> MediaPipeImageLandmarker:
    """Image モードの landmarker をセッションに 1 つ保つ。映像側とは分ける。"""
    fingerprint = str(model_path.resolve())
    if (
        st.session_state.get(IMAGE_LANDMARKER_PATH_KEY) != fingerprint
        or IMAGE_LANDMARKER_STATE_KEY not in st.session_state
    ):
        release_image_landmarker()
        st.session_state[IMAGE_LANDMARKER_STATE_KEY] = MediaPipeImageLandmarker(
            model_path=model_path,
            model_version=landmark_asset().version,
        )
        st.session_state[IMAGE_LANDMARKER_PATH_KEY] = fingerprint
    landmarker = st.session_state[IMAGE_LANDMARKER_STATE_KEY]
    if not isinstance(landmarker, MediaPipeImageLandmarker):
        raise LandmarkDetectionError("ランドマーク推論を初期化できません。")
    return landmarker


def _session_image_recorder(
    landmarker: MediaPipeImageLandmarker,
    config: MetricConfig,
    config_3d: MetricConfig,
) -> SessionRecorder:
    recorder = st.session_state.get(IMAGE_RECORDER_STATE_KEY)
    if not isinstance(recorder, SessionRecorder):
        recorder = SessionRecorder(
            metric_config=config,
            metric_config_3d=config_3d,
            thresholds=STILL_IMAGE_THRESHOLDS,
            tau_seconds=DEFAULT_TAU_SECONDS,
            landmark_model_version=landmarker.model_version,
            landmark_model_sha256=landmarker.model_sha256,
            app_version=__version__,
            lock_digest=file_sha256(LOCK_PATH),
        )
        st.session_state[IMAGE_RECORDER_STATE_KEY] = recorder
    return recorder


def _session_age_gate_service() -> AgeGateService:
    """FR-038: セッション中 1 度だけ構築する（ONNX セッションの再生成を避ける）。

    `build_age_gate_service` はモデル・spec が欠けても例外を投げず、
    `classifier=None` の service へ静かに倒れる（AC-AGEGATE-03）。
    """
    service = st.session_state.get(AGE_GATE_SERVICE_STATE_KEY)
    if not isinstance(service, AgeGateService):
        service = build_age_gate_service()
        st.session_state[AGE_GATE_SERVICE_STATE_KEY] = service
    return service


def _evaluate_image_age_gate(analysis: StillImageAnalysis | None) -> AgeGateOutcome | None:
    """FR-038: この写真について、参考年齢を表示してよいかを判定する。

    左右差の解析結果（`analysis`）は読むだけで一切変えない（不変条件 12）。
    ゲート自体の失敗（spec 読み込み・正規化）は `UNDETERMINED` へ倒れて
    参考年齢を隠す側（安全側）へ働くため、ここでは呼び出しの前提
    （`analysis is not None`）だけを見る。
    """
    if analysis is None:
        return None
    try:
        config_3d = load_metric_config(SYMMETRY_3D_METRIC_CONFIG_PATH)
    except MetricConfigError:
        return None
    return evaluate_age_gate_for_frame(
        _session_age_gate_service(),
        frame=analysis.landmarks.frame,
        outcome=analysis.outcome,
        config_3d=config_3d,
    )


# --- 分析 ---------------------------------------------------------------------


def _analyze_uploaded_image(
    rgb: np.ndarray,
    *,
    input_fingerprint: str,
    model_path: Path,
    show_midline: bool,
) -> tuple[SessionRecorder, StillImageAnalysis]:
    """アップロードされた写真 1 枚を検出・採点・記録する（状態を持たない、§8.7）。

    描画は一切しない。レイアウトは呼び出し側が所有するため、ここでの失敗が
    無関係なパネルを道連れにすることはない。
    """
    config = load_metric_config()
    # §8.5.1/§8.6.5, FR-036: 静止画も camera と同じく 3D を主指標として扱う。
    # `config_3d` を省略可能にすると渡し忘れが 3D の欠測と区別できなくなるため、
    # `ui.camera._build_pipeline` と同じ形で明示的に読み込む。
    config_3d = load_metric_config(SYMMETRY_3D_METRIC_CONFIG_PATH)
    landmarker = _session_image_landmarker(model_path)
    started = time.monotonic()
    analysis = analyze_still_image(
        landmarker,
        rgb,
        config=config,
        config_3d=config_3d,
        show_midline=show_midline,
    )
    processing_ms = (time.monotonic() - started) * 1000.0

    recorder = _session_image_recorder(landmarker, config, config_3d)
    if st.session_state.get(IMAGE_RECORDED_INPUT_KEY) != input_fingerprint:
        if recorder.records():
            recorder.reset()
        recorder.append_analysis(
            analysis.landmarks.frame,
            analysis.outcome,
            elapsed_ms=0,
            processing_ms=processing_ms,
            input_mode=InputMode.IMAGE,
        )
        st.session_state[IMAGE_RECORDED_INPUT_KEY] = input_fingerprint
    return recorder, analysis


def _maybe_estimate_image_demographics(
    age_input: np.ndarray | None,
    *,
    input_fingerprint: str,
    recorder: SessionRecorder | None,
) -> bool:
    """H2/§10.8: 単一顔を検出できたら、ボタン操作なしに年齢・性別を推定する。

    戻り値は「年齢・性別モデルが使える状態か」である。`resolve_ready_demographic`
    は未配置のとき導線（取得元・sha256・実行するコマンド）を**自分で描画する**
    ため、同じ判定を参考年齢の区画からもう一度呼ぶと、その導線が画面に 2 度出る。
    ここで 1 度だけ呼び、結果を呼び出し側が区画へ渡す（2026-08-27）。

    `IMAGE_RECORDED_INPUT_KEY` が landmark 記録の重複を防ぐのと同じ方式で、
    同じ写真（アップロード bytes の sha256 = ``input_fingerprint``）に対しては
    1 度しか推論を試みない。以後の再実行はセッション状態のキャッシュ
    （`RESULT_STATE_KEY` / `DEMOGRAPHIC_ERROR_KEY`）を読むだけで、ONNX 推論を
    走らせない。分析前の明示同意は Analyze ページ入口の同意ゲートが Camera・
    Image 共通で満たす（§10.8 項目 2）ため、推定ごとの追加同意は要求しない。

    不変条件 12・FR-024: 失敗はここで必ず捕まえ、左右差側の描画へ伝播させない。
    呼び出し側（`render_image_section`）はこの関数の後に左右差の吹き出し・詳細を
    描画するため、ここで例外を握りつぶすことがそのまま失敗分離の実装になる。
    """
    demographic_model = resolve_ready_demographic()
    if demographic_model is None:
        return False
    if age_input is None:
        return True
    if st.session_state.get(IMAGE_DEMOGRAPHIC_INPUT_KEY) == input_fingerprint:
        return True  # この写真に対しては既に 1 度試みている。

    st.session_state[IMAGE_DEMOGRAPHIC_INPUT_KEY] = input_fingerprint
    onnx_path, model_version = demographic_model
    try:
        estimate = _session_estimator(onnx_path, model_version).estimate(age_input)
        st.session_state[RESULT_STATE_KEY] = estimate
        st.session_state.pop(DEMOGRAPHIC_ERROR_KEY, None)
        if recorder is not None:
            recorder.attach_demographic_estimate(estimate, observed_elapsed_ms=0)
    except Exception as error:
        # 不変条件 12: 年齢・性別の失敗が左右差側を止めてはならない。
        # そのため例外をここで捕まえ、呼び出し側へ伝播させない。
        st.session_state.pop(RESULT_STATE_KEY, None)
        st.session_state[DEMOGRAPHIC_ERROR_KEY] = (
            str(error)
            if isinstance(error, DemographicInferenceError)
            else t("年齢・性別を推定できませんでした。")
        )
    return True


# --- タブの中身 ---------------------------------------------------------------


def _render_image_export(
    recorder: SessionRecorder,
    *,
    analysis: StillImageAnalysis | None = None,
    view: SymmetrizationView | None = None,
    gate_outcome: AgeGateOutcome | None = None,
) -> None:
    """FR-017 の台帳と、FR-042 の読み物を同じ場所へ置く。

    §9.6.2 の内容は「画面に出ているもの」と同じでなければならない。そこで
    参考年齢・参考性別は、**吹き出しへ実際に渡したのと同じ解決**を通す——
    `_resolve_shown_estimate` を共有することで、年齢ゲートが隠した判断が
    レポートの母集団から復元されることを防ぐ（§10.12-3）。
    """
    bundle = recorder.export_bundle()
    builder = None
    if analysis is not None:
        estimate = _resolve_shown_estimate(gate_outcome)
        notice = "" if gate_outcome is None else age_gate_notice(gate_outcome.status)

        def builder(include_figures: bool) -> str:
            return build_image_report(
                outcome=analysis.outcome,
                bundle=bundle,
                figures=build_report_figures(
                    include_figures=include_figures,
                    preview=analysis.landmarks.preview,
                    view=view,
                ),
                baseline=resolve_baseline(estimate),
                estimate=estimate,
                show_demographics=estimate is not None,
                notice=notice,
            )

    render_export_panel(
        bundle,
        heading=t("この画像の分析データをエクスポート"),
        warning=t("姿勢・左右差・参考年齢などの顔由来データを含みます。"),
        # 1 枚の写真は常にちょうど 1 件の記録になるので、件数を出しても
        # 情報が増えない。
        key_prefix="download_image",
        report_builder=builder,
        report_mode="image",
    )


def _render_image_symmetrization(
    rgb: np.ndarray, analysis: StillImageAnalysis
) -> SymmetrizationView | None:
    """FR-034: Camera と同じ一発の折り返しを、アップロードされた写真へ適用する。

    アップロードはすでに静止したフレームなので取り込む対象が無く、画面上の
    写真からそのまま表示を計算する（§14.3g）。それ以外——並び・文言・変形の
    任意指定——は `ui.symmetrization_panel` が Camera と共有する。

    FR-042: 作ったビューを**返す**。`st.tabs` はどのタブが表示されていても毎回
    すべての中身を構築するため、この計算は既に毎回走っている。レポート用に
    もう一度折りたたむのは同じ費用を 2 度払うだけであり、しかも 2 つの図が
    別々の計算から出ることになる。失敗したときは `None`。
    """
    st.markdown(SYMMETRIZATION_HEADING)
    # §8.6.4 は全画素を作り直し、このパネルは再実行のたびに描き直される。
    # そのため変形は任意指定のままにし、操作のたびに費用を払わせない。
    warp = render_warp_checkbox(key="image_symmetrization_warp")
    try:
        view = symmetrize_frame(
            rgb,
            analysis.landmarks.frame,
            config=load_metric_config(),
            warper=OpenCvPiecewiseWarper() if warp else None,
        )
    except SymmetrizationError as error:
        # FR-043: UI 境界で例外文言を訳す（`translations/_errors.py`）。
        st.info(translate_message(str(error)))
        return None

    st.caption(
        t(
            "写真の顔を正中線で折りたたみ、左右の対応点の中点を「左右対称にそろえた位置」として"
            "実測点と重ねています。ズレの大きさを形で見るための表示で、"
            "どちらが正しい・望ましいという意味はありません。"
        )
    )
    render_symmetrization_view(view, warp_requested=warp)
    return view


def _render_image_demographic_detail(
    age_input: np.ndarray | None,
    gate_outcome: AgeGateOutcome | None = None,
    *,
    available: bool = True,
) -> None:
    """年齢・性別の参考値パネル全体。常に下側へ画面幅いっぱいで表示する（§5.1）。

    FR-024 はこれを左右差側から独立させている。推定は `_maybe_estimate_image_demographics`
    がこの関数より先に済ませているため、ここでは結果を読んで表示するだけである。

    区画の形（囲み・来歴の 1 行・状態を見る順序・FR-038 のゲート適用）は
    `ui.demographic_panel` が Camera と共有する（2026-08-27）。この関数に残るのは
    **session state からの読み出しと、この入力に固有の 1 文**だけである——分岐を
    ここへ書き足すと Camera 側とまた食い違う。
    """
    session_estimate = st.session_state.get(RESULT_STATE_KEY)
    error_message = st.session_state.get(DEMOGRAPHIC_ERROR_KEY)
    render_demographic_panel(
        estimate=session_estimate if isinstance(session_estimate, DemographicEstimate) else None,
        gate_outcome=gate_outcome,
        available=available,
        # 静止画は 1 枚で完結するので、待てば出るという言い方をしない。
        pending_notice=(
            t("単一顔を検出できた写真でのみ、参考年齢・参考性別を推定できます。")
            if age_input is None
            else t("この写真からは参考年齢・参考性別を推定できませんでした。")
        ),
        error=str(error_message) if error_message else None,
    )


# --- セクション全体 -----------------------------------------------------------


def _resolve_shown_estimate(gate_outcome: AgeGateOutcome | None) -> DemographicEstimate | None:
    """**実際に画面へ出す**参考年齢・参考性別を 1 か所で決める。

    吹き出し（FR-039 の母集団もここから解決される）と FR-042 のレポートが同じ
    関数を通る。片方が session state の推定をそのまま読むと、**年齢ゲートが
    隠した判断が母集団の年齢から復元されてしまう**（§10.12-3）。

    判定そのものは `ui.age_gate_notice.resolve_reference_age_view` が持つ
    （2026-08-26、§5.1）。Camera の HUD・専用タブと同じ 1 つの解決を通すことで、
    「ある区画だけがゲートを見落とす」壊れ方が構造的に起きなくなる。ここに残るのは
    session state からの読み出しだけである。
    """
    session_estimate = st.session_state.get(RESULT_STATE_KEY)
    estimate = session_estimate if isinstance(session_estimate, DemographicEstimate) else None
    return resolve_reference_age_view(estimate=estimate, gate_outcome=gate_outcome).estimate


def _render_preview_with_overlay(
    analysis: StillImageAnalysis,
    gate_outcome: AgeGateOutcome | None = None,
) -> None:
    """§5.1/§14.3e（2026-08-12）: 画像に隣接して数値を出すのは吹き出しだけ。

    隣の見出し結果は持たない。Camera が項目 26（H3）で解消したのと同じ「同じ
    測定値が 2 か所へ出る」状態を Image でも作らないためである。画像は画面幅
    いっぱいを使う（横に並べる相手がもう無い）。

    §8.6.3/§8.7: 全体の対称度・全体のスコア・部位別のいずれも、主指標
    `symmetry-3d/1.0.0` から HUD と同じ解決（平滑値優先）を通す。静止画は平滑値
    を持たない（`analyze_still_image` が明示的に欠測へ落とす）ため、自然に生値へ
    落ちる。全体のスコアはもともと生値のみである（§8.6.5・§8.7）。

    FR-038: `gate_outcome.may_show_reference_age` が偽のときは推定値を吹き出し
    へ渡さず、代わりに `ui.age_gate_notice` の固定文言を出す。左右差（対称度・
    全体のスコア・部位別）の描画は変えない（不変条件 12）。

    FR-039: 参照平均は `estimate`——実際に吹き出しへ出す推定——からだけ解決する。
    ゲートが参考年齢を隠した後の `None` をそのまま渡すので、隠した判断が母集団の
    年齢から復元されることがない（§10.12-3）。
    """
    estimate = _resolve_shown_estimate(gate_outcome)
    notice = "" if gate_outcome is None else age_gate_notice(gate_outcome.status)
    render_image_with_overlay(
        analysis.landmarks.preview,
        caption=t("検出したランドマーク"),
        overall=resolve_overall_distance(analysis.outcome),
        equal_weight=resolve_equal_weight_distance(analysis.outcome),
        parts={name: resolve_part_distance_3d(analysis.outcome, name) for name in PART_NAMES},
        estimate=estimate,
        show_demographics=estimate is not None,
        notice=notice,
        gate=gate_color(analysis.outcome.state),
        baseline=resolve_baseline(estimate),
    )
    render_tracking_status(
        analysis.landmarks.frame.status,
        {status: t(_IMAGE_STATUS_MESSAGES[status]) for status in _IMAGE_STATUS_MESSAGES},
    )


def render_image_section() -> None:
    """アップロードした写真 1 枚の分析セクション。同意の後にのみ呼ぶこと。"""
    uploaded = st.file_uploader(
        t("JPEG / PNG の顔写真"),
        type=["jpg", "jpeg", "png"],
        help=t("1 名だけが写る写真を想定しています。上限 10MB / 12MP。"),
        on_change=clear_session_result,
    )
    if uploaded is None:
        st.info(
            t(
                "18〜75歳の写真を選ぶと、左右差と年齢・性別の参考値を写真の上の吹き出しへ表示します。"
            )
        )
        return

    uploaded_bytes = uploaded.getvalue()
    try:
        decoded = decode_uploaded_image(uploaded_bytes)
    except ImageInputError as error:
        # FR-043: UI 境界で例外文言を訳す（`translations/_errors.py`）。
        st.error(translate_message(str(error)))
        return

    rgb = np.asarray(decoded.rgb, dtype=np.uint8)
    input_fingerprint = sha256(uploaded_bytes).hexdigest()
    st.subheader(t("静止画のランドマークと左右差"))
    model_path = resolve_ready_landmarker()

    recorder: SessionRecorder | None = None
    analysis: StillImageAnalysis | None = None
    if model_path is not None:
        show_midline = st.toggle(
            t("顔の中心線を表示"),
            value=True,
            key="image_midline_enabled",
            help=t("採点に使う正中点から推定した直線です。表示を変えてもスコアは変わりません。"),
        )
        try:
            recorder, analysis = _analyze_uploaded_image(
                rgb,
                input_fingerprint=input_fingerprint,
                model_path=model_path,
                show_midline=show_midline,
            )
        except (LandmarkDetectionError, MetricConfigError) as error:
            # FR-043: UI 境界で例外文言を訳す（`translations/_errors.py`）。
            st.error(translate_message(str(error)))

    age_input: np.ndarray | None = None
    if analysis is not None:
        try:
            age_input = extract_face_crop(rgb, analysis.landmarks.frame)
        except ImageInputError:
            age_input = None

    # H2: 吹き出しが `estimate` を読む前に確定させる。ボタンが無くなったため、
    # ここで済ませないと吹き出しが 1 呼吸遅れて古い値のまま描画されてしまう。
    demographic_available = _maybe_estimate_image_demographics(
        age_input,
        input_fingerprint=input_fingerprint,
        recorder=recorder,
    )
    # FR-038: 表示直前にゲートを評価する。既存の推定処理・worker 構造は変えず、
    # 表示可否だけをここで決める（判断の詳細は実装後レポートに明記）。
    age_gate_outcome = _evaluate_image_age_gate(analysis)

    if analysis is None:
        st.info(t("ランドマークを表示できません。"))
    else:
        _render_preview_with_overlay(analysis, age_gate_outcome)

    symmetrization_view: SymmetrizationView | None = None

    def _detail_tab() -> None:
        """§5.1: 枠は `ui.detail_panel` が Camera と共有する。"""
        if analysis is None:
            render_detail_panel(
                None,
                empty_notice=t("この画像からは左右差を算出できませんでした。"),
                show_parts=True,
            )
            return

        found = analysis

        def _diagnostics() -> None:
            st.caption(
                t("入力 {width} × {height} · ランドマーク点数 {count}").format(
                    width=rgb.shape[1],
                    height=rgb.shape[0],
                    count=found.landmarks.landmark_count,
                )
            )

        render_detail_panel(
            found.outcome,
            empty_notice=t("この画像からは左右差を算出できませんでした。"),
            # AC-LAYOUT-01: Image は HUD を持たないため、部位別 4 つはここに残る。
            show_parts=True,
            diagnostics=_diagnostics,
        )

    def _symmetry_tab() -> None:
        nonlocal symmetrization_view
        if analysis is None:
            st.info(t("ランドマークが取れていないため、比較を作成できません。"))
        else:
            symmetrization_view = _render_image_symmetrization(rgb, analysis)

    def _demographic_tab() -> None:
        _render_image_demographic_detail(
            age_input, age_gate_outcome, available=demographic_available
        )

    def _report_tab() -> None:
        # FR-042: 対称化タブより後に走る（`render_result_tabs` の並びがそれを
        # 保証する）ので、上で作ったビューをそのまま使い回せる。
        if recorder is None:
            st.info(t("エクスポートできる分析結果がまだありません。"))
        else:
            _render_image_export(
                recorder,
                analysis=analysis,
                view=symmetrization_view,
                gate_outcome=age_gate_outcome,
            )

    # §5.1, AC-LAYOUT-01: 画像の下はすべて画面幅いっぱいで一度に 1 つだけ表示し、
    # 上の画像がスクロールで手の届かない位置へ行かないようにする。1 本の長い
    # スクロールだと、詳細結果・対称化・参考年齢を続けて積み上げた分だけ画像が
    # 押し上げられ、数値と写真を見比べられなくなる。
    # §5.1（2026-08-27）: 名前・順序・アイコンは `ui.result_tabs` が Camera と
    # 共有する。2026-08-29（項目 33）: Image だけが持つ「左右対称との比較」を渡す
    # ——Camera からは対称化と一発取り込みを削除した（FR-034）。
    render_result_tabs(
        ResultTabs(
            detail=_detail_tab,
            symmetry=_symmetry_tab,
            demographic=_demographic_tab,
            report=_report_tab,
        )
    )
