"""英語の訳表: `ui/report.py`, `ui/report_figures.py`, `ui/summary_panel.py`,
`ui/export_panel.py`（FR-043）。

`ui/report.py` と `ui/summary_panel.py` は `ui.guidance.REASON_GUIDANCE` の値を
`t(REASON_GUIDANCE[code])` という辞書添字の形で消費する。`tests/ui/test_i18n.py`
の AST 走査は、この形の呼び出しに対して**辞書の値のすべて**が訳表に在ることを
要求する（実行時にどの `code` が来るかを AST では決められないため）。したがって
`REASON_GUIDANCE`（16 件の理由コード指示文）の全訳をここへ持つ。`ui/scores.py`
（3e 担当）も同じ辞書を別の消費形で読むことがあるが、**同じ原文に同じ訳である
限り**、訳表をまたいで重複しても `translations/__init__.py` の統合は落ちない
（値が食い違ったときだけ落ちる）。
"""

from __future__ import annotations

from typing import Final

EN: Final[dict[str, str]] = {
    # --- ui/report.py: 見出し・ラベル ---------------------------------------
    "リアルタイム計測のまとめ": "Live measurement summary",
    "写真 1 枚の計測結果": "Measurement result for one photo",
    "{version} · 時刻はセッション開始からの相対値です。": (
        "{version} · Times are relative to the start of the session."
    ),
    "{version} · この結果は 1 枚の写真だけから作られています。": (
        "{version} · This result is built from a single photo."
    ),
    "測定結果": "Measurement results",
    "このセッションの要約": "This session's summary",
    "##### このセッションの要約": "##### This session's summary",
    "区間ごとの内訳": "Breakdown by segment",
    "区間 {index}": "Segment {index}",
    "区間 {index}（{short}）": "Segment {index} ({short})",
    "**区間 {index}**": "**Segment {index}**",
    "**区間 {index}**（`{short}`）": "**Segment {index}** (`{short}`)",
    "区間ごとの内訳（{count} 区間）": "Breakdown by segment ({count} segments)",
    "計測条件を満たしたフレーム": "Frames that met the measurement conditions",
    "記録した行数": "Recorded rows",
    "うち有効な行数": "Valid rows among them",
    "有効だった長さ": "Valid duration",
    # `_camera.py` も同じテンプレートを使う。訳を合わせて統合時の衝突を避ける。
    "{value:.1f} 秒": "{value:.1f} sec",
    "参考年齢を表示できたフレーム": "Frames where a reference age could be shown",
    "項目": "Item",
    "値": "Value",
    "計測条件を満たさなかった主な理由": "Main reasons the measurement conditions were not met",
    "{guidance}（{count:,} 回 · {share:.0%}）": "{guidance} ({count:,} times · {share:.0%})",
    "- {guidance}（{count:,} 回 · {share:.0%}）": "- {guidance} ({count:,} times · {share:.0%})",
    "中央値": "Median",
    "変動幅（真ん中50%）": "Spread (middle 50%)",
    "最小 〜 最大": "Min - max",
    "件数": "Count",
    "参照統計の平均": "Reference statistics mean",
    "対称度の推移": "Symmetry over time",
    "全体のスコアの推移": "Overall score over time",
    "横軸は記録された順序です。縦軸の範囲外は端に貼り付きます。": (
        "The horizontal axis is the recorded order. "
        "Values outside the vertical range are clamped to the edge."
    ),
    " 点": " pts",
    "{value:.1f} 点": "{value:.1f} pts",
    "平均: 参照統計の {span}・{genders}（人種で絞り込まず {count:,} 件）。探索値です。": (
        "Mean: reference statistics for {span} · {genders} "
        "({count:,} cases, not filtered by race). An exploratory value."
    ),
    "{median} 歳": "{median} years",
    "{median} 歳（真ん中50%: {low}〜{high} 歳）": (
        "{median} years (middle 50%: {low}-{high} years)"
    ),
    "表示できた判定 {count:,} 件": "{count:,} readings shown",
    "真ん中50%: {low}〜{high} 歳 · {meta}": "Middle 50%: {low}-{high} years · {meta}",
    "{value:.0f} 歳": "{value:.0f} years",
    "参考年齢（中央値）": "Reference age (median)",
    "参考年齢": "Reference age",
    "参考性別": "Reference gender",
    "参考年齢・参考性別": "Reference age & gender",
    "モデルが推定した参考値です。"
    "断定された年齢・性別ではなく、左右差の結果とは無関係な別々の値です。": (
        "A model-estimated reference value. It is not a definitive age or gender, "
        "and it is a separate value unrelated to the symmetry results."
    ),
    "画像": "Images",
    "このファイルには顔写真が含まれます。保存後の共有先はアプリからは制御できません。": (
        "This file includes photos of a face. The app cannot control where "
        "it is shared after you save it."
    ),
    "全体の対称度": "Overall symmetry",
    "全体のスコア": "Overall score",
    "参照統計の平均（差）": "Reference statistics mean (difference)",
    "測定": "Measured",
    "女性": "Female",
    "男性": "Male",
    "輪郭": "Contour",
    "目": "Eyes",
    "鼻": "Nose",
    "口": "Mouth",
    "「全体のスコア」の求め方": 'How the "Overall score" is calculated',
    "部位が 1 つでも欠測しているため、この画像では計算できません。": (
        "This cannot be calculated for this image because at least one part is missing."
    ),
    "4 部位の二乗平均平方根": "Root-mean-square of the 4 parts",
    "部位": "Part",
    # `_scores.py` も同じ語を持つ（列幅制約で "Dev" を選んでいる）。訳を合わせる。
    "ずれ": "Dev",
    "目盛りは 2 点で決まります。ずれ {full_score_pct:.2f}% が"
    " 100 点、ずれ {zero_score_pct:.2f}% が 0 点です。"
    "ずれが 1 ポイント動くと {points_per_deviation:.0f} 点動きます。": (
        "The scale is fixed by two points: a deviation of {full_score_pct:.2f}% is "
        "100 pts, and {zero_score_pct:.2f}% is 0 pts. Moving 1 point of deviation "
        "moves the score by {points_per_deviation:.0f} pts."
    ),
    "撮影の状態": "Capture conditions",
    "顔の向き yaw": "Face angle: yaw",
    "顔の向き pitch": "Face angle: pitch",
    "顔の向き roll": "Face angle: roll",
    "画面に占める顔の面積比": "Face area ratio of the frame",
    "肌表面の参考値（実験値）": "Skin-surface reference value (experimental)",
    # --- ui/summary_panel.py -------------------------------------------------
    "撮り直すだけで中央値は数ポイント動きます（実測で全幅 6.8 点）。": (
        "Just retaking the shot can move the median by a few points (an observed range of 6.8 pts)."
    ),
    "設定が違う区間の値は、同じ中央値へまとめられません。": (
        "Segments with different settings cannot be combined into the same median."
    ),
    "顔を検出して年齢の判定を行ったフレームのうち、参考年齢を表示できた割合です。"
    "顔が写っていないフレームは数えていません。": (
        "The share of frames where a face was detected and age was evaluated, "
        "for which a reference age could be shown. Frames with no face are not counted."
    ),
    "記録した全フレームのうち、品質ゲートを通った割合です。": (
        "The share of all recorded frames that met the quality gate."
    ),
    "最初と最後の有効フレームの間隔です。連続して有効だった時間ではありません。": (
        "The interval between the first and last valid frame. "
        "Not the duration of continuous validity."
    ),
    "記録 {total:,} 行 · うち有効 {ok:,} 行": "Recorded {total:,} rows · {ok:,} of them valid",
    # --- ui/export_panel.py --------------------------------------------------
    "顔写真を含む図をレポートへ入れる": "Include figures with face photos in the report",
    "ランドマークや対称化の図をレポートへ埋め込みます。数値と来歴は変わりません。": (
        "Embeds the landmark and symmetrisation figures in the report. "
        "The numbers and provenance do not change."
    ),
    "レポート（HTML）を保存": "Save report (HTML)",
    # --- ui/report_figures.py -------------------------------------------------
    "検出したランドマークを重ねたプレビュー": "Preview with detected landmarks overlaid",
    "左右対称にそろえた位置との比較": "Comparison with symmetrised positions",
    "実測した点と、左右対称にそろえた位置の点を重ねた図です。"
    "どちらが正しい・望ましいという意味はありません。": (
        "Overlay of the measured points and the symmetrised positions. "
        "Neither is correct or preferable."
    ),
    "左右対称にそろえた画像（幾何変形）": "Geometrically symmetrised image (processed)",
    "画素を幾何変形して作った加工画像です。無加工の写真として提示・共有しないでください。": (
        "A processed image created by geometrically warping the pixels. "
        "Do not present or share it as an unedited photo."
    ),
    # --- ui.guidance.REASON_GUIDANCE（`t(REASON_GUIDANCE[code])` の消費側分） ---
    "顔がカメラに映るようにしてください": "Position your face so the camera can see it.",
    "1人だけがカメラに映るようにしてください": "Make sure only one person is in view.",
    "顔を正面に向けてください": "Face the camera directly.",
    "カメラとの距離を調整してください": "Adjust your distance from the camera.",
    "撮影条件を確認してください": "Check the shooting conditions.",
    "顔を正面（左右方向）に向けてください": "Face forward (left-right direction).",
    "顔を正面（上下方向）に向けてください": "Face forward (up-down direction).",
    "頭の傾きをまっすぐにしてください": "Keep your head level.",
    "もう少しカメラから離れてください": "Move a little farther from the camera.",
    "もう少しカメラに近づいてください": "Move a little closer to the camera.",
    "そのままの位置で少し静止してください": "Hold still in this position for a moment.",
    "顔がはっきり映るようにしてください": "Make sure your face is clearly visible.",
    "明るい場所へ移動するか、照明を調整してください": (
        "Move to a brighter location or adjust the lighting."
    ),
    "光が強すぎます。照明を弱めてください": "The light is too strong. Reduce the lighting.",
    "カメラを固定し、ピントを合わせてください": "Hold the camera steady and bring it into focus.",
    "処理が遅延しています。しばらくお待ちください": "Processing is delayed. Please wait a moment.",
}
