"""英語の訳表: `ui/result_tabs.py`, `ui/detail_panel.py`, `ui/demographic_panel.py`,
`ui/guidance.py`, `ui/model_setup.py`, `ui/age_gate_notice.py`,
`ui/symmetrization_panel.py`（FR-043）。

`ui/guidance.py` は含まれていない。`REASON_GUIDANCE` は
`ReasonCode -> str` の辞書で、実際の表示は `ui.scores` / `ui.report` /
`ui.summary_panel`（いずれも別ファイル）が動的なキーで読む。`t()` の原文解決は
「文字列リテラル」か「同じモジュールの module-level でリテラルへ束縛された名前」
にしか対応しないため、辞書引きの結果を `t()` へ渡しても解決できない
（`test_translation_calls_use_a_resolvable_literal` が落ちる）。この辞書を
翻訳するには、参照元 3 ファイルの側で 1 件ずつ静的に解決できる形へ書き換える
設計変更が要り、担当外ファイルへまたがるためここでは行わない。
"""

from __future__ import annotations

from typing import Final

EN: Final[dict[str, str]] = {
    # --- result_tabs.py ------------------------------------------------------
    "📊 詳細結果": "📊 Details",
    "🪞 左右対称との比較": "🪞 Symmetry comparison",
    "👤 参考年齢・参考性別": "👤 Reference age & gender",
    "📄 レポート": "📄 Report",
    # --- detail_panel.py -------------------------------------------------------
    "処理の状態（診断用）": "Processing status (diagnostics)",
    # --- demographic_panel.py ---------------------------------------------------
    "年齢・性別モデルが未配置のため、参考値を表示できません。": (
        "The age and gender model is not installed, so a reference value cannot be shown."
    ),
    "映像の下の ⚙️ 表示設定で「年齢・性別の参考値を表示」を入れると、ここに表示します。": (
        "Turn on “Show age/gender reference values” in the ⚙️ "
        "Display settings below the video to see it here."
    ),
    # --- age_gate_notice.py ------------------------------------------------------
    "この映像・画像では参考年齢・参考性別を表示できません。左右差の分析結果には影響しません。": (
        "A reference age and gender cannot be shown for this video or image. "
        "This does not affect the left-right symmetry analysis results."
    ),
    # --- model_setup.py ----------------------------------------------------------
    "ランドマークモデルが未配置です。アプリは自動取得しません。": (
        "The landmark model is not installed. The app does not fetch it automatically."
    ),
    "取得元 {url} / {size:,} bytes / ライセンス {license} / sha256 {sha256}": (
        "Source {url} / {size:,} bytes / License {license} / sha256 {sha256}"
    ),
    "別の場所に置く場合は環境変数 `{env_var}` で `{path}` を指定します。": (
        "To use a different location, set the environment variable `{env_var}` to `{path}`."
    ),
    "配置済みモデルの sha256 が manifest と一致しません。"
    "改変または破損の可能性があるため読み込みません。": (
        "The sha256 of the installed model does not match the manifest. "
        "It will not be loaded because it may be modified or corrupted."
    ),
    "検証済み": "verified",
    "年齢・性別モデルのXML/BINバージョンが一致しません。": (
        "The XML and BIN versions of the age/gender model do not match."
    ),
    "年齢・性別モデルのXML/BINライセンスが一致しません。": (
        "The XML and BIN licenses of the age/gender model do not match."
    ),
    "年齢・性別ONNXの元asset指定がmanifestと一致しません。": (
        "The source asset references for the age/gender ONNX do not match the manifest."
    ),
    "年齢・性別ONNXと元assetのライセンスが一致しません。": (
        "The license of the age/gender ONNX does not match its source assets."
    ),
    "年齢・性別モデルの派生ONNXが未配置です。アプリは自動生成しません。": (
        "The derived ONNX for the age/gender model is not installed. "
        "The app does not generate it automatically."
    ),
    "年齢・性別モデルのONNXがmanifestと一致しません。"
    "改変または破損の可能性があるため読み込みません。": (
        "The ONNX for the age/gender model does not match the manifest. "
        "It will not be loaded because it may be modified or corrupted."
    ),
    # --- symmetrization_panel.py --------------------------------------------------
    "左右対称にそろえた画像も生成する（幾何変形・処理が重くなります）": (
        "Also generate a symmetrised image (geometric warp; heavier processing)"
    ),
    "ランドマークを結んだ三角形ごとに画素を変形し、左右対称にそろえた顔画像を作ります。"
    "加工画像であり、無加工の写真ではありません。": (
        "Warps pixels triangle by triangle between landmarks to create a symmetrised "
        "face image. It is a processed image, not an unedited photo."
    ),
    "対称化した画像は生成されていません。": "A symmetrised image has not been generated.",
    "左右対称にそろえた画像（ランドマークを基準に画素を変形した加工画像）": (
        "Symmetrised image (a processed image with pixels warped from the landmarks)"
    ),
    "対応点を左右対称の位置へ移動させ、その間の画素を三角形ごとに変形した合成画像です。"
    "実際の顔写真ではなく、より正しい・望ましい顔でもありません。"
    "加工画像であることを明示せずに提示・共有しないでください。": (
        "This is a composite image made by moving corresponding points to symmetric "
        "positions and warping the pixels between them triangle by triangle. It is not "
        "an actual photo of the face, and it does not represent a more correct or more "
        "desirable face. Do not present or share it without disclosing that it is a "
        "processed image."
    ),
    "ランドマークが覆わない髪・耳・首・背景は、画像の外周を固定したまま"
    "顔の変形に緩やかに追従するため、境界付近に不自然さが残ることがあります。": (
        "Hair, ears, neck, and background not covered by landmarks are held fixed at "
        "the image's outer edge while loosely following the face's warp, so unnatural "
        "artefacts can remain near the boundary."
    ),
    "実測点と、左右対称にそろえた位置（色違い）を線で結んだ表示": (
        "The measured points connected by lines to the symmetrised positions "
        "(shown in a different colour)"
    ),
    "平均のズレ": "Mean deviation",
    "最大のズレ": "Max deviation",
    "対称化した点 {count} 点。"
    "対応する左右のペアが定義されていない点は、実測のまま動かしていません。": (
        "Symmetrised points: {count}. Points without a defined left-right pair "
        "are left at their measured positions."
    ),
    "ズレは、両眼の間隔を 100% としたときの、実測点と左右対称にそろえた位置との距離です。"
    "小さいほど両者が近いことだけを意味し、良し悪しを表しません。": (
        "The deviation is the distance between the measured point and the symmetrised "
        "position, expressed as a percentage of the interocular distance. A smaller "
        "value only means the two points are closer; it does not indicate good or bad."
    ),
}
