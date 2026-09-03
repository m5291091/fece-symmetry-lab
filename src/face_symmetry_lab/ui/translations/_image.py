"""英語の訳表: `ui/image_section.py`（FR-043）。

`_IMAGE_STATUS_MESSAGES`（`TrackingStatus -> str` の辞書）は
`t(_IMAGE_STATUS_MESSAGES[status])` という辞書添字の形で消費される
（`tests/ui/test_i18n.py`）。この形は辞書の値のすべてが訳表に在ることを
要求するため、4 メンバー分の訳をここへ持つ。

`st.error(str(error))` / `st.info(str(error))` の 3 経路（動的な例外文言）は
訳せない（原文がリクエストごとに変わるため `t()` の鍵として引けない）。これは
`ui/translations/_errors.py` が静的な例外文言だけを扱う設計上の既知の穴で
あり、`ui/image_section.py` 側でも意図的に `t()` を通していない。
"""

from __future__ import annotations

from typing import Final

EN: Final[dict[str, str]] = {
    # --- ファイル選択と同意前の案内 -------------------------------------------
    "JPEG / PNG の顔写真": "JPEG / PNG face photo",
    "1 名だけが写る写真を想定しています。上限 10MB / 12MP。": (
        "Assumes a photo of exactly one person. Up to 10MB / 12MP."
    ),
    "18〜75歳の写真を選ぶと、左右差と年齢・性別の参考値を写真の上の吹き出しへ表示します。": (
        "Choose a photo of someone aged 18-75 to show the symmetry and the "
        "reference age/gender values in a bubble over the photo."
    ),
    "静止画のランドマークと左右差": "Still-image landmarks and symmetry",
    "顔の中心線を表示": "Show the face midline",
    # `_camera.py` も同じ原文を持つ（同種のトグル）。訳を合わせて統合時の衝突を避ける。
    "採点に使う正中点から推定した直線です。表示を変えても対称度は変わりません。": (
        "A line estimated from the midpoints used for"
        " scoring. Toggling this does not change the "
        "symmetry value."
    ),
    "ランドマークを表示できません。": "Landmarks cannot be shown.",
    # --- 追跡状態のメッセージ（`_IMAGE_STATUS_MESSAGES`）----------------------
    "顔を検出できませんでした。明るさ、顔の大きさ、向きを確認してください。": (
        "No face detected. Check the brightness, face size, and orientation."
    ),
    "1 名の顔を検出しました。": "Detected one face.",
    "2 名以上が写っています。1 名だけの写真を使用してください。": (
        "2 or more people are in frame. Please use a photo of one person only."
    ),
    "この画像を処理できませんでした。": "This image could not be processed.",
    # --- 検出したランドマークの吹き出し ----------------------------------------
    "検出したランドマーク": "Detected landmarks",
    # --- 年齢・性別の推定失敗（静的な文言のみ）---------------------------------
    "年齢・性別を推定できませんでした。": "Could not estimate age or gender.",
    "単一顔を検出できた写真でのみ、参考年齢・参考性別を推定できます。": (
        "Reference age and gender can only be estimated from a photo "
        "where a single face was detected."
    ),
    "この写真からは参考年齢・参考性別を推定できませんでした。": (
        "Reference age and gender could not be estimated from this photo."
    ),
    # --- 対称化パネル ----------------------------------------------------------
    "写真の顔を正中線で折りたたみ、左右の対応点の中点を「左右対称にそろえた位置」として"
    "実測点と重ねています。ズレの大きさを形で見るための表示で、"
    "どちらが正しい・望ましいという意味はありません。": (
        "The photo's face is folded along the midline, and the midpoints "
        "of left-right corresponding points are overlaid on the measured "
        'points as "symmetrised positions". This shows the size of the '
        "deviation as a shape; neither side is correct or preferable."
    ),
    "ランドマークが取れていないため、比較を作成できません。": (
        "The comparison cannot be created because no landmarks were obtained."
    ),
    # --- 詳細結果タブ ------------------------------------------------------------
    "この画像からは左右差を算出できませんでした。": (
        "Symmetry could not be calculated from this image."
    ),
    "入力 {width} × {height} · ランドマーク点数 {count}": (
        "Input {width} × {height} · landmark points {count}"
    ),
    # --- エクスポート ------------------------------------------------------------
    "この画像の分析データをエクスポート": "Export this image's analysis data",
    "姿勢・左右差・参考年齢などの顔由来データを含みます。": (
        "Includes face-derived data such as pose, symmetry, and reference age."
    ),
    "エクスポートできる分析結果がまだありません。": "There is no analysis result to export yet.",
}
