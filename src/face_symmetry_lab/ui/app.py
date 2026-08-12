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
from face_symmetry_lab.application.symmetrization import symmetrize_frame
from face_symmetry_lab.domain.export import InputMode
from face_symmetry_lab.domain.landmarks import TrackingStatus
from face_symmetry_lab.domain.metrics import PART_NAMES, MetricConfig, MetricConfigError
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.domain.smoothing import DEFAULT_TAU_SECONDS
from face_symmetry_lab.domain.symmetrization import SymmetrizationError
from face_symmetry_lab.ui.about_page import render_about_page
from face_symmetry_lab.ui.camera import (
    release_session_pipeline,
    render_camera_section,
)
from face_symmetry_lab.ui.export_panel import render_export_panel
from face_symmetry_lab.ui.model_setup import (
    landmark_asset,
    resolve_ready_demographic,
    resolve_ready_landmarker,
)
from face_symmetry_lab.ui.scores import (
    DEVIATION_UNIT_CAPTION,
    format_distance,
    gate_color,
    render_demographic_estimate,
    render_score_details,
    render_symmetrized_image,
    render_tracking_status,
    resolve_equal_weight_distance,
    resolve_overall_distance,
    resolve_part_distance_3d,
)
from face_symmetry_lab.ui.still_overlay import render_image_with_overlay

ESTIMATOR_STATE_KEY = "demographic_estimator"
ESTIMATOR_PATH_KEY = "demographic_estimator_paths"
RESULT_STATE_KEY = "demographic_estimate"
INPUT_MODE_STATE_KEY = "input_mode"
#: 最後に *有効だった* 選択。`st.segmented_control` は選択中の項目をもう一度
#: 押すと `None` を報告するが、§15.1 は常にどれかの入力が有効であることを求める。
#: そのため解決済みのモードを widget とは別に覚えておく。
ACTIVE_INPUT_MODE_KEY = "active_input_mode"
INPUT_MODE_CAMERA = "Camera"
INPUT_MODE_IMAGE = "Image"

#: FR-004: ラベル・アイコン・そのモードが実際に何をするか。選ぶ前に読めるように
#: するためである。セッション状態が持つ値と同じキーを使うので、文言を変えても
#: 保存済みの設定は生き残る。
_INPUT_MODE_CARDS: Final[dict[str, tuple[str, str, str]]] = {
    INPUT_MODE_CAMERA: (
        "Web カメラ",
        ":material/videocam:",
        "映像を見ながら、顔の向きや距離をその場で調整できます。"
        "数値は最大 8 Hz で更新され、映像は端末内のメモリだけで処理されます。",
    ),
    INPUT_MODE_IMAGE: (
        "画像ファイル",
        ":material/image:",
        "JPEG / PNG を 1 枚アップロードして分析します。"
        "10MB・12MP 以下、EXIF の向きを適用し、平滑化のない 1 回きりの結果を表示します。",
    ),
}
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
CONSENT_PRIVACY_KEY = "consent_privacy"
CONSENT_RESEARCH_KEY = "consent_research"
CONSENT_RIGHTS_KEY = "consent_rights"

_IMAGE_STATUS_MESSAGES = {
    TrackingStatus.NO_FACE: (
        "顔を検出できませんでした。明るさ、顔の大きさ、向きを確認してください。"
    ),
    TrackingStatus.SINGLE_FACE: "1 名の顔を検出しました。",
    TrackingStatus.MULTI_FACE: "2 名以上が写っています。1 名だけの写真を使用してください。",
    TrackingStatus.ERROR: "この画像を処理できませんでした。",
}


def _session_estimator(
    onnx_path: Path,
    model_version: str,
) -> OpenCvDemographicEstimator:
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


def _clear_session_result() -> None:
    st.session_state.pop(RESULT_STATE_KEY, None)
    st.session_state.pop(IMAGE_RECORDER_STATE_KEY, None)
    st.session_state.pop(IMAGE_RECORDED_INPUT_KEY, None)
    st.session_state.pop(IMAGE_DEMOGRAPHIC_INPUT_KEY, None)
    st.session_state.pop(DEMOGRAPHIC_ERROR_KEY, None)


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


def release_image_landmarker() -> None:
    """静止画用の landmarker を解放する（FR-004, §10.4）。"""
    landmarker = st.session_state.pop(IMAGE_LANDMARKER_STATE_KEY, None)
    st.session_state.pop(IMAGE_LANDMARKER_PATH_KEY, None)
    if isinstance(landmarker, MediaPipeImageLandmarker):
        landmarker.close()


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


def _render_image_export(recorder: SessionRecorder) -> None:
    render_export_panel(
        recorder.export_bundle(),
        heading="この画像の分析データをエクスポート",
        warning=(
            "姿勢・左右差・肌表面参考値・参考年齢・参考性別などの顔由来データを含みます。"
            "共有先と利用目的を確認してから保存してください。"
        ),
        # 1 枚の写真は常にちょうど 1 件の記録になるので、件数を出しても
        # 情報が増えない。
        key_prefix="download_image",
    )


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
    # `_build_pipeline` と同じ形で明示的に読み込む。
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
) -> None:
    """H2/§10.8: 単一顔を検出できたら、ボタン操作なしに年齢・性別を推定する。

    `IMAGE_RECORDED_INPUT_KEY` が landmark 記録の重複を防ぐのと同じ方式で、
    同じ写真（アップロード bytes の sha256 = ``input_fingerprint``）に対しては
    1 度しか推論を試みない。以後の再実行はセッション状態のキャッシュ
    （`RESULT_STATE_KEY` / `DEMOGRAPHIC_ERROR_KEY`）を読むだけで、ONNX 推論を
    走らせない。分析前の明示同意は Analyze ページ入口の同意ゲートが Camera・
    Image 共通で満たす（§10.8 項目 2）ため、推定ごとの追加同意は要求しない。

    不変条件 12・FR-024: 失敗はここで必ず捕まえ、左右差側の描画へ伝播させない。
    呼び出し側（`_render_image_section`）はこの関数の後に左右差の見出し・詳細を
    描画するため、ここで例外を握りつぶすことがそのまま失敗分離の実装になる。
    """
    demographic_model = resolve_ready_demographic()
    if age_input is None or demographic_model is None:
        return
    if st.session_state.get(IMAGE_DEMOGRAPHIC_INPUT_KEY) == input_fingerprint:
        return  # この写真に対しては既に 1 度試みている。

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
            else "年齢・性別を推定できませんでした。"
        )


def _render_image_symmetrization(
    rgb: np.ndarray,
    analysis: StillImageAnalysis,
) -> None:
    """FR-034: Camera と同じ一発の折り返しを、アップロードされた写真へ適用する。

    アップロードはすでに静止したフレームなので取り込む対象が無く、画面上の
    写真からそのまま表示を計算する（§14.3g）。
    """
    st.markdown("#### 左右対称にそろえた位置との比較")
    # §8.6.4 は全画素を作り直し、このパネルは再実行のたびに描き直される。
    # そのため変形は任意指定のままにし、操作のたびに費用を払わせない。
    warp = st.checkbox(
        "左右対称にそろえた画像も生成する（幾何変形・処理が重くなります）",
        key="image_symmetrization_warp",
        help=(
            "ランドマークを結んだ三角形ごとに画素を変形し、左右対称にそろえた顔画像を作ります。"
            "加工画像であり、無加工の写真ではありません。"
        ),
    )
    try:
        view = symmetrize_frame(
            rgb,
            analysis.landmarks.frame,
            config=load_metric_config(),
            warper=OpenCvPiecewiseWarper() if warp else None,
        )
    except SymmetrizationError as error:
        st.info(str(error))
        return

    st.caption(
        "写真の顔を正中線で折りたたみ、左右の対応点の中点を「左右対称にそろえた位置」として"
        "実測点と重ねています。ズレの大きさを形で見るための表示で、"
        "どちらが正しい・望ましいという意味はありません。"
    )
    st.image(
        view.overlay,
        caption="実測点と、左右対称にそろえた位置（色違い）を線で結んだ表示",
        width="stretch",
    )
    mean_column, max_column = st.columns(2)
    mean_column.metric("平均のズレ", format_distance(view.mean_deviation))
    max_column.metric("最大のズレ", format_distance(view.max_deviation))
    st.caption(
        f"対称化した点 {view.symmetrized_point_count} 点。"
        "対応する左右のペアが定義されていない点は、実測のまま動かしていません。"
    )
    st.caption(DEVIATION_UNIT_CAPTION)
    render_symmetrized_image(view, requested=warp)


def _render_image_demographic_detail(age_input: np.ndarray | None) -> None:
    """年齢・性別の参考値パネル全体。常に下側へ画面幅いっぱいで表示する（§5.1）。

    FR-024 はこれを左右差側から独立させている。推定は `_maybe_estimate_image_demographics`
    がこの関数より先に済ませているため、ここでは結果を読んで表示するだけである。
    黙って消えるのではなく、常に何が起きているかを説明する（顔が検出できない・
    モデルが未配置・推定が成功した・推定が失敗した、のいずれの状態でも）。
    """
    st.caption("runtime: OpenCV DNN CPU · Open Model Zoo 2023.0 FP16由来ONNX / 対象18〜75歳")
    if age_input is None:
        st.info("単一顔を検出できた写真でのみ、参考年齢・参考性別を推定できます。")
        return

    session_estimate = st.session_state.get(RESULT_STATE_KEY)
    if isinstance(session_estimate, DemographicEstimate):
        render_demographic_estimate(session_estimate)
        return

    error_message = st.session_state.get(DEMOGRAPHIC_ERROR_KEY)
    if error_message:
        st.error(error_message)
        return

    st.info("年齢・性別モデルが未配置のため、参考値を表示できません。")


def _render_image_section() -> None:
    uploaded = st.file_uploader(
        "JPEG / PNG の顔写真",
        type=["jpg", "jpeg", "png"],
        help="1 名だけが写る写真を想定しています。上限 10MB / 12MP。",
        on_change=_clear_session_result,
    )
    if uploaded is None:
        st.info(
            "18〜75歳の写真を選ぶと、左右差と年齢・性別の参考値を写真の上の吹き出しへ表示します。"
        )
        return

    uploaded_bytes = uploaded.getvalue()
    try:
        decoded = decode_uploaded_image(uploaded_bytes)
    except ImageInputError as error:
        st.error(str(error))
        return

    rgb = np.asarray(decoded.rgb, dtype=np.uint8)
    input_fingerprint = sha256(uploaded_bytes).hexdigest()
    st.subheader("静止画のランドマークと左右差")
    model_path = resolve_ready_landmarker()

    recorder: SessionRecorder | None = None
    analysis: StillImageAnalysis | None = None
    if model_path is not None:
        show_midline = st.toggle(
            "顔の中心線を表示",
            value=True,
            key="image_midline_enabled",
            help="採点に使う正中点から推定した直線です。表示を変えてもスコアは変わりません。",
        )
        try:
            recorder, analysis = _analyze_uploaded_image(
                rgb,
                input_fingerprint=input_fingerprint,
                model_path=model_path,
                show_midline=show_midline,
            )
        except (LandmarkDetectionError, MetricConfigError) as error:
            st.error(str(error))

    landmark_frame = analysis.landmarks.frame if analysis is not None else None
    age_input: np.ndarray | None = None
    if landmark_frame is not None:
        try:
            age_input = extract_face_crop(rgb, landmark_frame)
        except ImageInputError:
            age_input = None

    # H2: 見出し（render_headline_metrics）が `estimate` を読む前に確定させる。
    # ボタンが無くなったため、ここで済ませないと見出しが 1 呼吸遅れて古い値の
    # まま描画されてしまう。
    _maybe_estimate_image_demographics(
        age_input,
        input_fingerprint=input_fingerprint,
        recorder=recorder,
    )

    session_estimate = st.session_state.get(RESULT_STATE_KEY)
    estimate = session_estimate if isinstance(session_estimate, DemographicEstimate) else None
    # §8.6.3/§8.7: 全体の対称度・全体のスコアは主指標 symmetry-3d/1.0.0 から作る。
    # 静止画は平滑値を持たない（`analyze_still_image` が明示的に欠測へ落とす）
    # ため、全体の対称度は自然に生値へ落ちる。全体のスコアはもともと生値のみ
    # （§8.6.5・§8.7）。
    overall = resolve_overall_distance(analysis.outcome) if analysis is not None else None
    equal_weight = resolve_equal_weight_distance(analysis.outcome) if analysis is not None else None

    # §5.1/§14.3e（2026-08-12）: 画像に隣接して数値を出すのは画像上の吹き出し
    # だけとし、隣の見出し結果は持たない。Camera が項目 26（H3）で解消したのと
    # 同じ「同じ測定値が 2 か所へ出る」状態を Image でも作らないためである。
    # 画像は画面幅いっぱいを使う（横に並べる相手がもう無い）。
    if analysis is None:
        st.info("ランドマークを表示できません。")
    else:
        # §8.6.3/§8.7: 部位別も見出し・HUD と同じ解決（平滑値優先、静止画は
        # 平滑値を持たないため自然に生値へ落ちる）を通す。
        parts = {name: resolve_part_distance_3d(analysis.outcome, name) for name in PART_NAMES}
        render_image_with_overlay(
            analysis.landmarks.preview,
            caption="検出したランドマーク",
            overall=overall,
            equal_weight=equal_weight,
            parts=parts,
            estimate=estimate,
            show_demographics=estimate is not None,
            gate=gate_color(analysis.outcome.state),
        )
        render_tracking_status(analysis.landmarks.frame.status, _IMAGE_STATUS_MESSAGES)

    # §5.1, AC-LAYOUT-01: 画像の下はすべて画面幅いっぱいで一度に 1 つだけ表示し、
    # 上の画像がスクロールで手の届かない位置へ行かないようにする。**Camera と
    # 同じタブ構成にする**（§5.1「Image 側だけ異なる並びにしない」）。1 本の長い
    # スクロールだと、詳細結果・対称化・参考年齢を続けて積み上げた分だけ画像が
    # 押し上げられ、数値と写真を見比べられなくなる。
    detail_tab, symmetry_tab, demographic_tab, export_tab = st.tabs(
        ["詳細結果", "左右対称との比較", "参考年齢・参考性別", "エクスポート"]
    )
    with detail_tab:
        if analysis is None:
            st.info("この画像からは左右差を算出できませんでした。")
        else:
            st.caption(
                f"入力 {rgb.shape[1]} × {rgb.shape[0]} · "
                f"ランドマーク点数 {analysis.landmarks.landmark_count}"
            )
            # 静止画は状態を持たないため、生の値をそのまま表示する（§8.7）。
            render_score_details(analysis.outcome, show_smoothed=False)
    with symmetry_tab:
        if analysis is None:
            st.info("ランドマークが取れていないため、比較を作成できません。")
        else:
            _render_image_symmetrization(rgb, analysis)
    with demographic_tab:
        _render_image_demographic_detail(age_input)
    with export_tab:
        if recorder is None:
            st.info("エクスポートできる分析結果がまだありません。")
        else:
            _render_image_export(recorder)


def _render_input_mode_selector() -> str:
    """FR-004: 入力方法を選ぶ。各選択肢は選ぶ前に説明を読めるようにする。

    `st.segmented_control` は選択中の項目を解除でき、そのままではページに入力
    が 1 つも無い状態になる。§15.1 は常にどれかが有効であることを求めるため、
    `None` を読んだときは空の画面を描かずに直前のモードへ戻す。選択自体は
    ネイティブ widget のままなので、キーボード操作は変わらない（AC-A11Y-01）。
    """
    st.markdown("#### 入力方法")
    # widget のキーではなく自前のキーから読む。widget が生成された後、その
    # キーは「解除」を意味する `None` を保持することがあり、それこそが
    # fallback が乗り越えなければならない値だからである。
    previous = st.session_state.get(ACTIVE_INPUT_MODE_KEY, INPUT_MODE_CAMERA)
    selected = st.segmented_control(
        "入力方法",
        options=list(_INPUT_MODE_CARDS),
        format_func=lambda mode: _INPUT_MODE_CARDS[mode][0],
        key=INPUT_MODE_STATE_KEY,
        on_change=_switch_input_mode,
        label_visibility="collapsed",
    )
    mode = selected if selected in _INPUT_MODE_CARDS else previous
    st.session_state[ACTIVE_INPUT_MODE_KEY] = mode

    for column, (name, (label, icon, description)) in zip(
        st.columns(len(_INPUT_MODE_CARDS)),
        _INPUT_MODE_CARDS.items(),
        strict=True,
    ):
        with column, st.container(border=True):
            heading, badge = st.columns([3, 1], vertical_alignment="center")
            heading.markdown(f"{icon} **{label}**")
            if name == mode:
                badge.badge("選択中", color="blue")
            st.caption(description)
    return mode


def _switch_input_mode() -> None:
    """FR-004: モードが変わったら、直前の入力の runtime 状態を破棄する。"""
    release_session_pipeline()
    release_image_landmarker()
    _clear_session_result()


def _render_about_footer() -> None:
    """概要と About ページへのリンク（§4.1, §5.1）。

    算出方法の全体、誤差要因、禁止用途、モデルの版は、ここではなく About
    ページに置く。このページを「同意 → 入力選択 → 分析」に集中させるためである。
    """
    st.divider()
    st.subheader("Face Symmetry Lab について")
    st.write(
        "顔のランドマーク、撮影補正後の肌表面参考値、モデル推定の参考年齢・参考性別を端末内で確認します。"
        "リアルタイム映像にはランドマークと中心線を重ね、スコアは映像の隣に表示します。"
        "数値履歴は再現性メタデータ付きの CSV / JSON として保存できます。"
    )
    boundary, metric, cadence = st.columns(3)
    boundary.metric("処理境界", "端末内のみ")
    metric.metric("左右差指標", "相対比較専用")
    cadence.metric("カメラ解析", "最大 8 Hz")
    st.info(
        "左右差は両眼の間隔に対する対応点のずれとして表示します（0〜100 のスコアではありません）。"
        "数値は、同一人物・同一環境での相対変化だけを参照してください。"
    )
    st.page_link(ABOUT_PAGE, label="対称度の算出方法・制約について詳しく見る", icon="📘")


def render_app() -> None:
    """Analyze ページ: 同意 → 入力選択 → 分析（§5.1）。"""
    st.title("Face Symmetry Lab")
    st.caption("LOCAL PROCESSING · NO IMAGE STORAGE · RESEARCH USE")

    # すでに同意済みなら（直前の実行の widget 状態から判断して）畳んでおく。
    # 再訪した利用者が毎回この節をスクロールで通り過ぎずに済むようにする。
    already_consented = all(
        st.session_state.get(key, False)
        for key in (CONSENT_PRIVACY_KEY, CONSENT_RESEARCH_KEY, CONSENT_RIGHTS_KEY)
    )
    with st.expander("利用前に確認してください", expanded=not already_consented):
        consent_privacy = st.checkbox(
            "映像と画像は端末内のメモリだけで処理され、アプリは保存・外部送信しません。",
            key=CONSENT_PRIVACY_KEY,
        )
        consent_research = st.checkbox(
            "私は18〜75歳です。結果は参考値であり、実年齢、自己認識・法的性別、"
            "肌質・医学的状態、人物の価値を示さないことに同意します。",
            key=CONSENT_RESEARCH_KEY,
        )
        consent_rights = st.checkbox(
            "本人の映像・画像、または分析する明示的な権限・同意があるものだけを使用します。",
            key=CONSENT_RIGHTS_KEY,
        )

    consented = consent_privacy and consent_research and consent_rights
    if not consented:
        st.info("3 項目を確認すると、カメラまたは画像を選択できます。")
        return

    input_mode = _render_input_mode_selector()
    st.divider()

    if input_mode == INPUT_MODE_CAMERA:
        render_camera_section()
    else:
        release_session_pipeline()
        _render_image_section()

    _render_about_footer()


#: Streamlit `st.navigation` の multipage 入口（§4.1）。import 時に構築する。
#: `st.Page` は実行中のスクリプトコンテキストを必要としないため、モジュールの
#: 入り方に関わらず `render_app` が `st.page_link` から `ABOUT_PAGE` を参照
#: できる。
ANALYZE_PAGE = st.Page(render_app, title="Analyze", icon="🪞", default=True)
ABOUT_PAGE = st.Page(render_about_page, title="使い方・算出方法", icon="📘")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Face Symmetry Lab",
        page_icon="🪞",
        layout="wide",
    )
    st.navigation([ANALYZE_PAGE, ABOUT_PAGE]).run()
