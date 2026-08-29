"""英語の訳表: `ui/camera.py`, `ui/smoothing_panel.py`（FR-043）。
鍵は日本語の原文そのもの（ADR-0010 (a)）。`_CHOICE_LABELS` / `_HOLD_LABELS` /
`_BACKGROUND_MODE_LABELS` の値（`options` に渡す識別子）はここに含めない
——widget の選択肢は翻訳しない（ADR-0010 (c)）。表示だけを切り替える
`format_func`（`_tau_choice_label` など）が別に持つテンプレートだけを載せる。

`camera.py` は `ui.age_gate_notice.SESSION_HIDDEN_NOTICE` を消費するので、
その訳はここへ入れる（消費側の表に置く。`_shared.py` などへは書かない）。
"左上" / "右上" / "左下" / "右下" / "非表示"（`ui.hud.HUD_POSITION_OPTIONS`）は
`_hud.py` が既に登録済みなので、ここでは重複登録しない
（同じ原文に同じ訳なら重複してもマージ時に落ちないが、二重管理を避ける）。
"""

from __future__ import annotations

from typing import Final

EN: Final[dict[str, str]] = {
    " · ズーム目標 {target_px}px": " · Zoom target {target_px}px",
    "##### 数値の落ち着き（移動平均の時間）": ("##### Display stability (moving-average window)"),
    (
        "18〜75歳向けモデル推定の参考年齢・参考性別を最大1Hzで独立推論します。左右差の計算"
        "とは分離しており、失敗してもカメラ映像は止まりません。"
    ): (
        "Independently infers reference age & gender "
        "for ages 18-75 at up to 1 Hz. It is separate"
        " from the symmetry calculation, and a "
        "failure here does not stop the camera video."
    ),
    "63% 追いつくまで": "Time to reach 63%",
    "95% 追いつくまで": "Time to reach 95%",
    "`START` を押して計測を始めると、ここに計算過程などの数値が表示されます。": (
        "Press `START` to begin measuring; the calculation details will appear here."
    ),
    "`START` を押して計測を始めると、参考年齢・参考性別をここに表示します。": (
        "Press `START` to begin measuring; the reference age & gender will appear here."
    ),
    "`START` を押すと、ブラウザがカメラの許可を求めます。": (
        "Press `START` and your browser will ask for camera permission."
    ),
    "{median} 歳": "{median} years",
    "{median} 歳（真ん中50%: {low}〜{high} 歳）": (
        "{median} years (middle 50%: {low}-{high} years)"
    ),
    "{value:.1f} 個": "{value:.1f} samples",
    # `_report.py` も同じテンプレートを使うため、訳を合わせる（merge が
    # 「同じ原文に違う訳」を検出して落とすのを避ける）。
    "{value:.1f} 秒": "{value:.1f} sec",
    "⚙️ 表示設定": "⚙️ Display settings",
    "✕ 全画面を終了": "✕ Exit full-screen",
    (
        "いまの解析周期（{interval_ms} "
        "ms）での目安です。解析周期はAIが自動調整するため、同じ設定でもこの本数は動きます。"
    ): (
        "A guide for the current analysis interval "
        "({interval_ms} ms). Because the interval is "
        "auto-adjusted by AI, this count moves even "
        "with the same setting."
    ),
    (
        "このセッションでは、品質ゲートを満たすフレームがありませんでした。明るさ・顔の大きさ・向"
        "きを整えて `START` から取り直してください。"
    ): (
        "No frame in this session met the quality "
        "gate. Adjust the lighting, face size, and "
        "pose, then start again with `START`."
    ),
    (
        "このセッションでは参考年齢・参考性別を表示できたフレームがありませんでした。左右差の分析"
        "結果には影響しません。"
    ): (
        "No frame in this session showed a reference "
        "age & gender. It does not affect the "
        "symmetry results."
    ),
    (
        "このセッションでは参考年齢・参考性別を表示できませんでした。左右差の分析結果には影響しま"
        "せん。"
    ): (
        "No reference age & gender was shown in this "
        "session. It does not affect the symmetry "
        "results."
    ),
    ("このセッションで表示できた参考年齢 {span} · 参考性別 {genders} · {count} 件"): (
        "Reference age shown this session {span} · gender {genders} · {count} samples"
    ),
    "しない": "Off",
    "ほぼ追いついたと言える時間です。時定数のおよそ 3.0 倍になります。": (
        "The time to be considered fully settled, about 3.0 times the time constant."
    ),
    (
        "まだ書き出せる記録がありません。カメラを開始し、品質ゲートを満たすフレームが得られると記"
        "録が貯まります。"
    ): (
        "No records to export yet. Start the camera; "
        "records accumulate once frames meet the "
        "quality gate."
    ),
    "セッションデータのエクスポート": "Session data export",
    "プレビュー専用フレーム": "Preview-only frames",
    "ランドマーク点数": "Landmark point count",
    "人物以外を単色にします。左右差・年齢性別・肌表面参考値の判定には使いません。": (
        "Turns everything but the person into a flat "
        "colour. Not used for symmetry, age/gender, "
        "or skin-surface reference values."
    ),
    "内訳: ": "Breakdown: ",
    "処理が続けて失敗しています。カメラを停止して再開してください。": (
        "Processing keeps failing. Stop the camera and start again."
    ),
    "処理中スキップ": "Skipped while busy",
    "品質ゲートを満たすと、ここに計算過程などの数値が表示されます。": (
        "Once the quality gate is met, the calculation details appear here."
    ),
    "均している解析サンプル数": "Analysis samples averaged",
    "変わるのは画面の数値が動く速さだけで、測定値は変わりません。": (
        "This changes display speed only, not the measurements."
    ),
    "姿勢・左右差などの顔由来の派生データを含みます。": (
        "Includes face-derived data such as pose and symmetry."
    ),
    "平均する時間": "Averaging window",
    "平均する時間を伸ばしても動く回数は変わりません。数字を止めたいときはこちらを長くします。": (
        "Extending the averaging window does not "
        "reduce how often it updates. Lengthen this "
        "instead to hold the number still."
    ),
    "年齢・性別の参考値を表示": "Show reference age & gender",
    "年齢・性別モデルが未配置のため使用できません。": (
        "Unavailable because the age/gender model is not installed."
    ),
    "採点に使う正中点から推定した直線です。表示を変えてもスコアは変わりません。": (
        "A line estimated from the midpoints used for"
        " scoring. Toggling this does not change the "
        "score."
    ),
    "推論を停止してモデルを解放": "Stop inference and release the model",
    "数値を入れ替える間隔": "Value refresh interval",
    (
        "明るさ・追跡ズーム・cadence は AI が自動調整します（FR-029, "
        "FR-030）。明るさ補正は landmark 検出にだけ使い、表示される映像は変わりま"
        "せん。品質ゲート閾値は固定既定値で、左右差の計算には調整できる係数がないため、自動調整が"
        "数値を動かすことはありません。"
    ): (
        "Brightness, tracking zoom, and cadence are "
        "auto-adjusted by AI (FR-029, FR-030). "
        "Brightness gain is only used for landmark "
        "detection; the displayed video is unchanged."
        " The quality-gate thresholds are fixed "
        "defaults, and the symmetry calculation has "
        "no adjustable coefficients, so "
        "auto-adjustment never changes the numbers."
    ),
    ("明るさ補正 ×{gain:.2f}{zoom} · 解析周期 {interval_ms} ms"): (
        "Brightness gain ×{gain:.2f}{zoom} · Analysis interval {interval_ms} ms"
    ),
    (
        "映像へ重ねる HUD の吹き出し（状態・平滑化時間・全体の対称度・全体のスコア・部位別4"
        "つ・参考年齢・参考性別・折れ線グラフ）の位置です。映像自体は加工しません。HUD "
        "は掴んで動かせます。プリセットを選び直すと元の角へ戻ります。"
    ): (
        "Position of the HUD overlay on the video "
        "(status · smoothing window · overall "
        "symmetry · overall score · the 4 parts · "
        "reference age & gender · trend chart). The "
        "video itself is not altered. Drag the HUD to"
        " move it; choosing a preset returns it to "
        "that corner."
    ),
    ("映像をブラウザ画面いっぱいに広げます。ブラウザの F11 と併用すると完全な全画面になります。"): (
        "Expands the video to fill the browser "
        "window. Combine with your browser's F11 for "
        "true full-screen."
    ),
    "映像を全画面表示": "Show video full-screen",
    "映像上の HUD の位置": "HUD position on the video",
    "毎回": "Every frame",
    (
        "画面の数値を、選んだ時間のあいだ入れ替えずに置きます。測定・記録・エクスポートはこの設定"
        "に関わらず今までどおり続きます。"
    ): (
        "Holds the on-screen value unchanged for the "
        "chosen duration. Measurement, recording, and"
        " export continue as usual regardless of this"
        " setting."
    ),
    "白": "White",
    "短いほど反応が速くぶれも大きく、長いほど落ち着きますが追随が遅れます。": (
        "Shorter reacts faster with more jitter; longer settles but lags more."
    ),
    "結果・履歴・平滑化をリセット": "Reset results, history & smoothing",
    "背景を置き換える（表示専用）": "Replace the background (display only)",
    "背景置換 {ms:.1f} ms": "Background replacement {ms:.1f} ms",
    "背景置換モデルが未配置のため、この機能は使用できません。": (
        "Unavailable because the background-replacement model is not installed."
    ),
    "表示が新しい水準へ 63% 近づくまでの時間です（この値が時定数そのものです）。": (
        "Time for the display to reach 63% of a new level (this value is the time constant itself)."
    ),
    (
        "表示中の数値は、直近の解析結果を時間で重み付けして平均したものです（時間ベースEMA）。"
        "ここで選ぶのはその平均の時間の長さで、測定そのものの精度や計算式は変わりません。"
    ): (
        "The value shown is a time-weighted average "
        "of recent results (a time-based EMA). This "
        "sets only the averaging window's length; it "
        "does not change the measurement's precision "
        "or formula."
    ),
    ("解析サンプル #{frame_id} · 処理 {processing_ms:.1f} ms · 最大 {max_hz} Hz"): (
        "Analysis sample #{frame_id} · processing {processing_ms:.1f} ms · max {max_hz} Hz"
    ),
    "計測を停止しました。以下はこのセッションで最後に解析したフレームの値です。": (
        "Measurement stopped. Below are the values from the last analysed frame in this session."
    ),
    "追跡状態": "Tracking status",
    (
        "選び直すと、それまでの平均をいったん捨てて新しい時間で測り直します。エクスポートも新しい"
        "区間として記録され、映像上のグラフは描き直しになります。"
    ): (
        "Reselecting discards the running average and"
        " starts timing again with the new window. "
        "Exports record this as a new segment, and "
        "the on-screen chart redraws."
    ),
    "顔が 1 名だけ正面に近い向きで写っている間に、参考年齢・参考性別を推定します。": (
        "While a single face is looking roughly forward, the reference age & gender are estimated."
    ),
    (
        "顔が外れたときや計測条件を満たさなくなったときは、この間隔を待たずにすぐ「—」へ戻ります"
        "。止まっている数値が、すでに測れていない状態のまま残ることはありません。"
    ): (
        "If the face is lost or no longer meets the "
        'measurement conditions, this reverts to "—" '
        "immediately without waiting out the "
        "interval. A held value never remains once it"
        " stops being measurable."
    ),
    "顔の中心線を表示": "Show the face midline",
    "黒": "Black",
}
