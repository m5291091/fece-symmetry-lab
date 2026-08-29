"""英語の訳表: UI 境界で `translate_message(str(error))` として訳す例外文言（FR-043）。

`domain` / `application` / `adapters` が持つ例外文言のうち、`ui` が
`st.error` / `st.info` / `st.warning` へそのまま渡す**静的な**ものだけをここへ
入れる。値を埋め込んで組み立てる f-string の例外文言は原文と一致しないため
鍵として引けず、英語表示でも日本語のまま残る（許容範囲、ADR-0010 Consequences）。

**この表の鍵は `t("...")` という形の静的呼び出しでは参照されない**
（`translate_message(str(error))` のように実行時の値を包むため）。そのため
`tests/ui/test_i18n.py` の孤児判定（`test_no_orphan_english_entries`）は
`translations.RUNTIME_KEYS`（この表の鍵集合）を除外して判定する。

`tests/ui/test_i18n.py::test_static_exception_literals_have_an_error_boundary_translation`
が、`domain` / `application` / `adapters` の `raise <Exc>("...")` を AST 走査して
ここへの入れ忘れを検出する。新しい静的な例外文言を追加したら、この表へも追加する。
"""

from __future__ import annotations

from typing import Final

EN: Final[dict[str, str]] = {
    # --- application/image_input.py（ImageInputError） ------------------------
    "画像を読み込めません。空のファイルです。": "Cannot load the image. The file is empty.",
    "ファイルサイズが上限を超えています。": "The file size exceeds the limit.",
    "JPEG または PNG を選択してください。": "Please choose a JPEG or PNG file.",
    "アニメーション画像は使用できません。": "Animated images are not supported.",
    "画像の画素数が上限を超えています。": "The image's pixel count exceeds the limit.",
    "画像を読み込めません。破損していないか確認してください。": (
        "Cannot load the image. Please check that it is not corrupted."
    ),
    "顔 crop には uint8 の RGB 画像が必要です。": "The face crop requires a uint8 RGB image.",
    "顔モデル推論には単一顔のlandmarkが必要です。": (
        "Face model inference requires single-face landmarks."
    ),
    "顔 crop の余白率が不正です。": "The face crop margin ratio is invalid.",
    "顔 crop の範囲を決定できません。": "Could not determine the face crop bounds.",
    # --- domain/metrics.py（MetricConfigError） --------------------------------
    "landmark_count は正の値である必要があります。": ("landmark_count must be a positive value."),
    # --- domain/warp_mesh.py（WarpMeshError） ----------------------------------
    "画像サイズは正の値である必要があります。": "The image size must be a positive value.",
    "三角形は (3, 2) の配列である必要があります。": "Triangles must be a (3, 2) array.",
    "変形元と変形先の点数が一致しません。": (
        "The source and destination point counts do not match."
    ),
    "変形にはランドマークが必要です。": "Warping requires landmarks.",
    "両眼間隔は正の有限値である必要があります。": (
        "The inter-eye distance must be a positive, finite value."
    ),
    "変位の上限比は正の有限値である必要があります。": (
        "The displacement limit ratio must be a positive, finite value."
    ),
    # --- application/symmetrization.py（SymmetrizationError） ------------------
    "単一顔のフレームだけを対称化できます。": "Only a single-face frame can be symmetrised.",
    # --- adapters/inference/mediapipe_landmarker.py（LandmarkDetectionError） --
    "MediaPipe を読み込めません。": "Cannot load MediaPipe.",
    "ランドマーク出力が不正です。": "The landmark output is invalid.",
    "ランドマークモデルが見つかりません。": "The landmark model was not found.",
    "ランドマーク推論は停止済みです。": "Landmark inference has already stopped.",
    "ランドマーク推論には uint8 の RGB 画像が必要です。": (
        "Landmark inference requires a uint8 RGB image."
    ),
    "ランドマークを検出できませんでした。": "Could not detect landmarks.",
    # --- adapters/inference/opencv_demographic_estimator.py（DemographicInferenceError）
    "年齢・性別モデルを初期化できません。": "Cannot initialise the age/gender model.",
    "年齢・性別推定にはuint8のRGB画像が必要です。": (
        "Age/gender estimation requires a uint8 RGB image."
    ),
    "年齢・性別モデルのONNXが見つかりません。": ("The age/gender model's ONNX file was not found."),
    "年齢・性別推定を実行できませんでした。": "Could not run age/gender estimation.",
    "年齢・性別モデルの出力shapeが不正です。": ("The age/gender model output shape is invalid."),
    "年齢・性別モデルの出力が不正です。": "The age/gender model output is invalid.",
    # --- adapters/io/metric_config.py（MetricConfigError） ---------------------
    "metric config の形式が不正です。": "The metric config format is invalid.",
    # --- adapters/io/model_store.py（ModelStoreError） --------------------------
    "モデル取得は HTTPS のみを許可します。": "Model downloads are only allowed over HTTPS.",
    "model manifest に models 配列がありません。": ("The model manifest has no models array."),
    "model manifest に derived_models 配列がありません。": (
        "The model manifest has no derived_models array."
    ),
    "取得したモデルの sha256 が manifest と一致しません。": (
        "The downloaded model's sha256 does not match the manifest."
    ),
}
