"""英語の訳表: `ui/scores.py`, `ui/readings.py`（FR-043）。

訳語は `unit3-i18n-spec.md`「訳語の固定」表に従う。特に「対称度（実験値・
両眼距離比）」「試験中の指標です」「輪郭・目・鼻・口」「待機中」「計測中」
「顔が見つかりません」は固定訳を使う。

**「全体のスコア」の計算過程開示は撤回した**（2026-09-03、§0 項目 34）。
関連する訳語もここから削除した。
"""

from __future__ import annotations

from typing import Final

EN: Final[dict[str, str]] = {
    # --- ui/scores.py: 見出し・常時表示の注意 --------------------------------
    "対称度（実験値・両眼距離比）": "Symmetry (experimental, interocular-distance ratio)",
    "試験中の指標です。": "This is an experimental metric.",
    "対称度は両眼間隔に対するずれを 100% から引いた値です。目標値ではありません。": (
        "Symmetry is 100% minus the deviation relative to the interocular "
        "distance. It is not a target value."
    ),
    "対称度は、両眼の間隔を 100% としたときの左右対応点のずれを 100% から引いた値です。"
    "100% は対応点距離がすべて 0 になる数学上の上限で、目標値ではありません。"
    "高い・低いは良し悪しを表しません。"
    "実写では表示値が数パーセントポイントの狭い範囲に集まりやすく、"
    "その中の違いを対称・非対称の判定として読まないでください。": (
        "Symmetry is 100% minus the deviation of left-right corresponding "
        "points, taking the interocular distance as 100%. 100% is the "
        "mathematical upper bound where all point distances are 0; it is "
        "not a target. Higher or lower does not mean better or worse. In "
        "real photos, displayed values tend to cluster in a narrow range "
        "of a few percentage points; please do not read differences "
        "within that range as a symmetric/asymmetric judgement."
    ),
    "同一人物・同一環境の中で傾向を見るための値です。他人との比較や医学的評価には使えません。": (
        "For trends within one person and setting. Not for comparing people or medical use."
    ),
    "部位別の値は、同じ条件で撮り直すだけでも動きます。"
    "撮影のたびに動く幅より小さい変化は、部位別の値だけでは左右差の変化として読み取れません。"
    "この幅は部位ごとに異なり、個人差は未検証です。": (
        "Per-region values move just from retaking a photo under the "
        "same conditions. A change smaller than that per-shot movement "
        "cannot be read as a change in left-right difference from the "
        "per-region values alone. This range differs by region, and "
        "individual differences are unverified."
    ),
    # --- ui/scores.py: 状態・姿勢 ---------------------------------------------
    "計測中です。": "Measuring.",
    "撮影条件を調整してください。": "Please adjust the capture conditions.",
    "顔が見つかりません。": "No face detected.",
    "2 名以上が写っています。": "2 or more people are in frame.",
    "このフレームは処理できませんでした。": "This frame could not be processed.",
    "姿勢: 取得できません": "Pose: unavailable",
    "姿勢 yaw {yaw:.1f}° / pitch {pitch:.1f}° / roll {roll:.1f}°": (
        "Pose yaw {yaw:.1f}° / pitch {pitch:.1f}° / roll {roll:.1f}°"
    ),
    " · 顔面積比 {ratio:.2f}": " · Face area ratio {ratio:.2f}",
    # --- ui/scores.py: 部位別の内訳 -------------------------------------------
    "部位別の対称度（全体値の内訳ではなく、ずれの分布を見るための独立した値です）": (
        "Per-region symmetry (an independent value for seeing how "
        "deviation is distributed, not a breakdown of the overall value)"
    ),
    "ずれ {deviation}": "Deviation {deviation}",
    "生値 {value}": "Raw value {value}",
    # --- ui/scores.py: 総合値の併記・版 ---------------------------------------
    "全体の対称度のずれ {deviation}（対称度との和は 100%）": (
        "Overall symmetry deviation {deviation} (sums to 100% with symmetry)"
    ),
    "平滑化前の生値 {symmetry}（ずれ {deviation}）": (
        "Raw value before smoothing {symmetry} (deviation {deviation})"
    ),
    "metric {metric_3d} · 2D 診断 {metric_2d} · landmark map {landmark_map}": (
        "metric {metric_3d} · 2D diagnostic {metric_2d} · landmark map {landmark_map}"
    ),
    # --- ui/scores.py: 参考年齢・参考性別 -------------------------------------
    "モデル推定の参考年齢・参考性別": "Reference age & gender (model estimate)",
    "参考年齢": "Reference age",
    "{age:.1f}歳": "{age:.1f} years",
    "外見由来の参考性別": "Reference gender (from appearance)",
    "性別モデル確率 {confidence:.1%}・対象18〜75歳。"
    "年齢や本人の自己認識・法的性別を断定せず、本人確認や人物評価には使用できません。": (
        "Gender model probability {confidence:.1%} · for ages 18-75. It "
        "does not determine age, self-identified or legal gender, and "
        "cannot be used for identity verification or evaluating a person."
    ),
    "モデル出力 female {female:.3f} / male {male:.3f}": (
        "Model output female {female:.3f} / male {male:.3f}"
    ),
    "同一区間内の直近推論を再表示しています。": (
        "Showing the most recent inference from this interval."
    ),
    "モデル情報": "Model information",
    # --- ui/readings.py: 部位・性別・ゲート状態のラベル -----------------------
    "輪郭": "Contour",
    "目": "Eyes",
    "鼻": "Nose",
    "口": "Mouth",
    "女性": "Female",
    "男性": "Male",
    "計測中": "Measuring",
    "調整中": "Adjusting",
    "顔が見つかりません": "No face detected",
    "顔が複数あります": "Multiple faces detected",
    "処理に失敗しました": "Processing failed",
    "待機中": "Waiting",
    # --- ui/readings.py: 整形 -------------------------------------------------
    "更新 {seconds:.1f} 秒ごと": "Updates every {seconds:.1f}s",
    "平滑化 {seconds:.1f} 秒": "Smoothing {seconds:.1f}s",
    "{female} {female_pct}% ・ {male} {male_pct}%": "{female} {female_pct}% / {male} {male_pct}%",
}
