"""英語の訳表: `ui/scores.py`, `ui/readings.py`, `ui/baseline.py`（FR-043）。

`ui/baseline.py` は表示文言を持たない（値の解決のみ）ため、この表の中身は
`ui/scores.py` と `ui/readings.py` の分だけである。

訳語は `unit3-i18n-spec.md`「訳語の固定」表に従う。特に「対称度（実験値・
両眼距離比）」「全体のスコア」「試験中の指標です」「輪郭・目・鼻・口」
「平均」「待機中」「計測中」「顔が見つかりません」は固定訳を使う。
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
    # --- ui/scores.py: 「全体のスコア」--------------------------------------
    "全体のスコア": "Overall score",
    "{label}の計算過程を見る": "{label}: how it is calculated",
    "部位を 1 票ずつ数えた対称度を、実測が集まる範囲へ合わせた目盛りで読み直した値です。"
    "鼻のようにペア数の少ない部位も、目のように多い部位と同じ重みで入ります。"
    "ずれが両眼間隔の 1.0% で 100 点、5.0% で 0 点になります。"
    "全体の対称度（対応点を 1 票ずつ）とは重みも目盛りも違うため、数字の大小を直接比べられません。"
    "較正された感度係数や部位の重み付けを持たない値であり、採点ではありません。"
    "母集団の分布を持たないため、順位や正常範囲を表しません。"
    "高い・低いは良し悪しを表しません。": (
        "Symmetry counted one vote per region, re-read on a scale matched "
        "to the range where measurements cluster. Regions with fewer "
        "pairs, such as the nose, count with the same weight as regions "
        "with many, such as the eyes. A deviation of 1.0% of the "
        "interocular distance gives 100 points, and 5.0% gives 0 points. "
        "Its weighting and scale differ from the overall symmetry (which "
        "counts one vote per point), so the two numbers cannot be "
        "compared directly. It has no calibrated sensitivity coefficient "
        "or region weighting, and it is not a form of grading. It has no "
        "population distribution, so it does not indicate where you fall "
        "in an ordering or a typical range. Higher or lower does not "
        "mean better or worse."
    ),
    # `SCORE_SCALE_CAPTION`（「撮り直すだけで 6.8 点ほど動く目盛りです。」）は
    # 現行の描画コードのどこからも呼ばれていない既存の未使用定数（本 unit の
    # 変更前から）。訳を足すと `test_no_orphan_english_entries` に引っかかる
    # ため、ここでは訳を追加しない——fail-open で日本語原文（23 字）のまま
    # 表示され、`test_i18n.py` の長さテスト（<60）も原文の長さで満たされる。
    "この目盛りは、実測が集まる範囲（ずれ 1.0〜5.0%）へ合わせて引き伸ばしてあります。"
    "引き伸ばした分だけばらつきも拡大し、同じ人を同じ条件で撮り直しただけでも 6.8 点ほど動きます"
    "（`docs/symmetry-metric-measurements.md` §14.4、被写体 1 名）。"
    "それより小さい点差を、人と人の違いとして読まないでください。": (
        "This scale is stretched to match the range where measurements "
        "cluster (deviation 1.0-5.0%). Stretching it also stretches the "
        "variability by the same amount: retaking a photo of the same "
        "person under the same conditions alone moves the score by about "
        "6.8 points (`docs/symmetry-metric-measurements.md` §14.4, one "
        "subject). Please do not read point differences smaller than "
        "that as a difference between people."
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
    # --- ui/scores.py: 計算過程の 3 ステップ ---------------------------------
    "ステップ 1 — 4 つの部位それぞれの「ずれ」を読む": (
        'Step 1 — Read the "deviation" for each of the 4 regions'
    ),
    "ステップ 2 — 4 つのずれを 1 つにまとめる": "Step 2 — Combine the 4 deviations into one",
    "ステップ 3 — まとめたずれを「点」へ置き換える": (
        'Step 3 — Convert the combined deviation into "points"'
    ),
    "ここでいう「ずれ」は、両目の間隔を100%としたときの、左右で対応する"
    "点どうしの離れ具合です。両目の間隔が6cmの人なら、ずれ1%はおよそ0.6mmに"
    "あたります。輪郭・目・鼻・口の4か所について、それぞれの値と、次のステップで"
    "使う二乗を並べます。": (
        'The "deviation" here is how far apart left-right corresponding '
        "points are, taking the distance between the eyes as 100%. For "
        "someone with a 6cm eye separation, a 1% deviation is about "
        "0.6mm. For the 4 regions - contour, eyes, nose, and mouth - we "
        "list each value alongside the square used in the next step."
    ),
    "4つのずれを、二乗して平均し、平方根をとって1つの数にまとめます"
    "（二乗平均平方根）。単純に足して4で割らないのは、1か所だけ大きくずれている"
    "ときに、その分が残り3部位に薄められてしまうためです。二乗すると大きいずれ"
    "ほど強く残り、最後に平方根をとることで単位が元の「ずれ」に戻ります。": (
        "The 4 deviations are squared, averaged, and square-rooted into "
        "a single number (root mean square). We do not simply add them "
        "and divide by 4, because if only one region has a large "
        "deviation, that would be diluted by the other 3 regions. "
        "Squaring keeps larger deviations more prominent, and taking "
        "the square root at the end returns the unit to the original "
        '"deviation".'
    ),
    "まとめたずれを、2つの目印にあてはめて点へ置き換えます。ずれが両眼間隔の"
    "1.0%のとき100点、5.0%のとき0点とし、その間を直線でつなぎます。この2つの"
    "目印は実測と文献から決めた固定の値で、母集団の中の位置を表すものでは"
    "ありません。": (
        "The combined deviation is converted into points using 2 fixed "
        "markers. A deviation of 1.0% of the interocular distance gives "
        "100 points, and 5.0% gives 0 points, with a straight line "
        "connecting the two. These 2 markers are fixed values decided "
        "from measurements and the literature, and do not represent a "
        "position within a population."
    ),
    "この目盛りでは、ずれが1ポイント（両眼間隔の1%）動くと点が"
    "{points:.0f}点動きます。"
    "点の細かさは、測定の細かさではありません。": (
        "On this scale, moving the deviation by 1 point (1% of the "
        "interocular distance) moves the score by {points:.0f} points. "
        "The fineness of the points is not the fineness of the "
        "measurement."
    ),
    "部位別の値のいずれかが欠測しているため、今は数値を当てはめられません。": (
        "One or more per-region values are missing, so no numbers can be filled in right now."
    ),
    "全体の対称度は対応点126個を1個ずつ数えるため、点の多い部位（目26組）ほど"
    "強く効きます。全体のスコアは部位を1つずつ数えるため、点の少ない部位"
    "（鼻7組）も同じ重みで入ります。この重みの違いが、全体の対称度と全体の"
    "スコアが一致しない理由です。": (
        "Overall symmetry counts all 126 corresponding points one by "
        "one, so regions with more points (26 pairs for the eyes) have "
        "a stronger effect. The overall score counts regions one by "
        "one, so regions with fewer points (7 pairs for the nose) count "
        "with the same weight. This difference in weighting is why "
        "overall symmetry and the overall score do not agree."
    ),
    "成人の顔の左右のずれは、両眼の間隔の1〜2%ほどです。100からずれを引くだけの"
    "目盛りでは、これが「98%前後」に集まって差が読めません。上の式はその1〜2%の"
    "あたりを広げて読むための目盛りであり、点数そのものは母集団の中の位置を表しません。": (
        "An adult face's left-right deviation is typically about 1-2% "
        "of the interocular distance. On a scale that simply subtracts "
        'the deviation from 100, this clusters around "98%" and the '
        "differences become unreadable. The formula above is a scale "
        "that stretches that 1-2% range for reading, and the score "
        "itself does not represent a position within a population."
    ),
    "同じ人を撮り直した実測（被写体 1 名）では、セッションごとの中央値が81〜88点、"
    "フレーム単位では46〜93点に散りました"
    "（`docs/symmetry-metric-measurements.md` §12.1・§14.4）。文献で「対称」とされた"
    "成人と「非対称」とされた成人の差2.11ポイントは、この目盛りでは52.8点にあたります"
    "（同§12.0）。撮影条件だけで数点動くため、小さな点差を人物の違いとして読めません。": (
        "In repeated measurements of the same person (one subject), "
        "per-session medians ranged from 81 to 88 points, and per-frame "
        "values ranged from 46 to 93 points "
        "(`docs/symmetry-metric-measurements.md` §12.1, §14.4). The "
        "2.11-point difference in the literature between adults classed "
        'as "symmetric" and "asymmetric" corresponds to 52.8 points on '
        "this scale (same §12.0). Because capture conditions alone move "
        "the score by several points, small point differences cannot be "
        "read as differences between people."
    ),
    "まとめたずれ = √( (輪郭のずれ² + 目のずれ² + 鼻のずれ² + 口のずれ²) ÷ 4 )": (
        "Combined deviation = √( (Contour deviation² + Eyes deviation² "
        "+ Nose deviation² + Mouth deviation²) ÷ 4 )"
    ),
    "  ずれ² の合計 = {squares} = {total}": "  Sum of deviation² = {squares} = {total}",
    "  {count} 部位で平均 = {total} ÷ {count} = {mean}": (
        "  Mean over {count} regions = {total} ÷ {count} = {mean}"
    ),
    "  平方根をとる = √{mean} = {combined}  ← まとめたずれ": (
        "  Take the square root = √{mean} = {combined}  ← combined deviation"
    ),
    "{label} = 100 × ({zero_anchor} - まとめたずれ) ÷ ({zero_anchor} - {full_anchor})": (
        "{label} = 100 × ({zero_anchor} - combined deviation) ÷ ({zero_anchor} - {full_anchor})"
    ),
    "部位": "Part",
    # `_report.py` も同じ原文「ずれ」を持つため、値を合わせる（`_merge` は
    # 同じ鍵に違う訳が付くと落ちる）。3 文字 = 3 桁で `_VALUE_COLUMN_CELLS`
    # に十分収まる。
    "ずれ": "Dev",
    "ずれ²": "Dev²",
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
    # --- ui/readings.py: 整形・参照平均の併記 ---------------------------------
    "{points:.1f} 点": "{points:.1f} points",
    "平均 {mean:.2f}%": "mean {mean:.2f}%",
    "平均 {mean:.2f}%（{delta}）": "mean {mean:.2f}% ({delta})",
    "平均 {mean:.1f} 点": "mean {mean:.1f} points",
    "平均 {mean:.1f} 点（{delta}）": "mean {mean:.1f} points ({delta})",
    "括弧内は表示値と平均の差です。探索値であり、基準値ではありません。": (
        "Bracket = diff from the mean. Exploratory, not a benchmark."
    ),
    "平均: 参照統計の {age} 歳・{gender}（人種で絞り込まず {count:,} 件）。{caveat}": (
        "mean: reference statistics for {age}-year-old {gender} (not "
        "filtered by race, {count:,} records). {caveat}"
    ),
    "更新 {seconds:.1f} 秒ごと": "Updates every {seconds:.1f}s",
    "平滑化 {seconds:.1f} 秒": "Smoothing {seconds:.1f}s",
    "{female} {female_pct}% ・ {male} {male_pct}%": "{female} {female_pct}% / {male} {male_pct}%",
}
