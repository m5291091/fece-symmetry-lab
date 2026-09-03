"""英語の訳表: `ui/about_page.py`, `ui/reference_page.py`（FR-043）。

**翻訳しないもの**（担当ファイルの docstring 参照）。
- `ui/reference_page.py` の `metadata.warning`（`st.text` で出す配布元注意書き
  の原文引用）。`tests/ui/test_reference_page.py` が原文一致を要求する。
- `domain.reference_stats` の `AGE_BANDS` / `RACE_LABELS` / `GENDER_LABELS`
  （データ列の値そのもの）。
- `dataset_name` / `license_scope` / `dataset_url` / 各 version / sha256。
"""

from __future__ import annotations

from typing import Final

EN: Final[dict[str, str]] = {
    # --- ui/about_page.py: モデルとバージョン --------------------------------
    "モデルとバージョン": "Models and versions",
    "- **顔ランドマーク**: {version} · sha256 `{sha256}…`": (
        "- **Face landmarks**: {version} · sha256 `{sha256}…`"
    ),
    "顔ランドマークモデルのmanifestを読み込めません。": (
        "The face landmark model manifest could not be loaded."
    ),
    "- **参考年齢・参考性別**: {version} · sha256 `{sha256}…`": (
        "- **Reference age & gender**: {version} · sha256 `{sha256}…`"
    ),
    "年齢・性別モデルのmanifestを読み込めません。": (
        "The age/gender model manifest could not be loaded."
    ),
    "- **背景置換（表示専用）**: {version} · sha256 `{sha256}…`": (
        "- **Background replacement (display only)**: {version} · sha256 `{sha256}…`"
    ),
    "背景置換モデルのmanifestを読み込めません。": (
        "The background-replacement model manifest could not be loaded."
    ),
    "- **参考年齢の表示ゲート**: {version} · spec_revision {revision} · ONNX sha256 `{sha256}…`": (
        "- **Reference age display gate**: {version} · spec_revision {revision} "
        "· ONNX sha256 `{sha256}…`"
    ),
    "年齢ゲートのspecを読み込めません。": "The age gate spec could not be loaded.",
    "版とsha256は結果のエクスポートにも記録されます。"
    "詳細は docs/model-cards/ を参照してください。": (
        "The version and sha256 are also recorded in the exported results. "
        "See docs/model-cards/ for details."
    ),
    # --- ui/about_page.py: render_about_page ---------------------------------
    "使い方と対称度の算出方法": "How it works and how symmetry is calculated",
    (
        "Face Symmetry Lab は、Web カメラまたは静止画から顔のランドマークを端末内で解析し、"
        "左右差をわかりやすく可視化する研究・セルフチェック用ツールです。"
        "画像は外部へ送信・保存されず、同一人物・同一環境内での左右差の相対的な傾向、"
        "およびモデル推定の参考年齢・参考性別を確認できます。"
    ): (
        "Face Symmetry Lab is a research and self-check tool that analyses facial "
        "landmarks on-device from a webcam or a still photo and visualises "
        "left-right differences. Images are never sent externally or stored; "
        "you can review relative trends in symmetry within the same person "
        "and environment, along with a model-estimated reference age and gender."
    ),
    (
        "試験中の指標です。撮影条件で値が動くため、"
        "数値の絶対値ではなく同一条件での相対変化だけを参照してください。"
    ): (
        "This is an experimental metric. Because the value moves with shooting "
        "conditions, treat it only as a relative change under the same "
        "conditions rather than as an absolute number."
    ),
    "左右差の算出方法": "How left-right differences are calculated",
    "幾何左右差（4部位+全体、主指標は3D）": (
        "Geometric left-right difference (4 parts + overall, primary metric is 3D)"
    ),
    (
        "1. MediaPipe Face Landmarker が検出する 478 点のランドマークのうち、"
        "版管理された左右対応点 63 組（126 点）だけを輪郭・目・鼻・口の 4 部位で使用します。\n"
        "2. 両眼の目頭・目尻の中点を結ぶ線が水平になるよう回転補正し、"
        "両眼の中点間距離を 1.0 として拡大縮小を正規化します"
        "（虹彩中心は視線で動くため基準にしません）。\n"
        "3. **主指標は 3D の object symmetry 解析です。** 折る平面（中心線）を先に決めず、"
        "対応点を鏡映してペア名を入れ替えた配置を、元の配置へ並進と回転だけで重ね合わせます"
        "（大きさは合わせません）。重ね合わせが最適な折り方を自動で見つけるため、"
        "姿勢（向き・傾き）に依存しません。\n"
        "4. 重ねきれずに残った残差の RMS（二乗平均平方根）が距離です。"
        "126 点をまとめた残差から全体の値を、部位ごとに読んだ残差から部位別の値を得ます。"
        "部位別は全体の内訳ではなく、ズレの分布を見るための並列の診断値です。\n"
        "5. 画面には**対称度**として表示します。両眼の間隔を 100% としたときのずれを、"
        "100% から引いた値です（例: ずれ 2.15% → 対称度 97.85%）。"
        "係数のない引き算だけの表示変換で、指標そのもの・エクスポートの値・`metric_version` は"
        "変わりません。ずれが両眼間隔を超える検出破綻時のみ 0% で下げ止まります。\n"
        "6. 実写では表示値が数パーセントポイントの狭い範囲に集まりやすく、"
        "その中の違いだけで対称・非対称を判定することはできません。部位別の値も、"
        "同じ条件で撮り直すだけで動く幅があり、その幅より小さい変化は部位別の値だけでは"
        "左右差の変化として読み取れません（部位ごとの個人差は未検証です）。\n"
        "7. 中心線（正中点の Total Least Squares 直線）で折りたたむ従来の 2D 指標も並行して"
        "算出し、エクスポートに `distance_*_raw`（`metric_version_2d`）として診断目的で残します。"
        "この 2D 指標は、目の左右差が正規化基準（眼アンカー）へ吸収されて距離へ現れないことと、"
        "鼻周辺の非対称が中心線を引きずって総合値を下げうることが実測で確認されているため、"
        "画面には表示しません。以前の版にあった感度係数 `k` と部位の重み `w` は、"
        "固定評価セットで校正されていない恣意的な値だったためどちらの指標からも廃止しました。\n"
        "8. 数学上、対応点の距離がすべて 0 なら対称度 100% になりますが、これは望ましさや"
        "優劣を意味せず、実写で 100% になることも保証しません。"
    ): (
        "1. Of the 478 landmarks detected by the MediaPipe Face Landmarker, we "
        "use only the 63 version-controlled left-right corresponding pairs "
        "(126 points), across the 4 parts: contour, eyes, nose, and mouth.\n"
        "2. We rotate so that the line joining the midpoints of the inner and "
        "outer eye corners is horizontal, and normalise scale so the distance "
        "between the two eye midpoints is 1.0 (the iris centre is not used as "
        "a reference because it moves with gaze direction).\n"
        "3. **The primary metric is a 3D object-symmetry analysis.** Rather "
        "than fixing a folding plane (a centre line) in advance, we mirror the "
        "corresponding points, swap their pair names, and align that "
        "arrangement to the original using only translation and rotation (not "
        "scale). Because the alignment automatically finds the best fold, the "
        "result does not depend on head pose (orientation or tilt).\n"
        "4. The distance is the RMS (root-mean-square) of the residual left "
        "after alignment. The overall value comes from the residual across all "
        "126 points combined, and the per-part values come from reading the "
        "residual within each part. The per-part values are not a breakdown of "
        "the overall value; they are a parallel diagnostic for looking at how "
        "the deviation is distributed.\n"
        "5. On screen we display this as **symmetry**: the deviation, "
        "expressed as a percentage of the eye-to-eye distance, subtracted from "
        "100% (example: a deviation of 2.15% becomes a symmetry of 97.85%). "
        "This is purely a display conversion — a subtraction with no "
        "coefficient — and it does not change the metric itself, the exported "
        "values, or `metric_version`. It floors at 0% only when detection has "
        "broken down and the deviation exceeds the eye-to-eye distance.\n"
        "6. In real photos, the displayed value tends to cluster within a "
        "narrow range of a few percentage points, so differences within that "
        "range alone cannot be used to judge symmetric versus asymmetric. The "
        "per-part values also move just from retaking the shot under the same "
        "conditions, and a change smaller than that range cannot be read as a "
        "change in left-right difference from the per-part values alone "
        "(per-part individual variation has not been verified).\n"
        "7. We also compute, in parallel, the previous 2D metric that folds "
        "along a centre line (a Total Least Squares line through the "
        "midpoints), and keep it in the export as `distance_*_raw` "
        "(`metric_version_2d`) for diagnostic purposes. We do not display this "
        "2D metric on screen, because measurements have confirmed that (a) eye "
        "left-right differences are absorbed into the normalisation reference "
        "(the eye anchor) and do not show up in the distance, and (b) "
        "asymmetry around the nose can drag the centre line and lower the "
        "combined value. The sensitivity coefficient `k` and the part weights "
        "`w` from an earlier version have been removed from both metrics, "
        "because they were arbitrary values that were never calibrated against "
        "a fixed evaluation set.\n"
        "8. Mathematically, symmetry would be 100% if every corresponding-"
        "point distance were 0, but this does not imply desirability or "
        "superiority, and there is no guarantee that a real photo will ever "
        "reach 100%."
    ),
    "左右対称にそろえた位置・画像": "Symmetrised positions and image",
    (
        "- 対応する左右のペアについて、実測点とその鏡映点の**中点**を「左右対称にそろえた位置」"
        "として算出します。正中点は中心線への垂線の足になります。\n"
        "- 対応するペアが定義されていない点（478 点のうち大半）は動かしません。推測で対応を"
        "補うことはしません。\n"
        "- 実測点と対称化点の距離は、上の RMS 距離のちょうど半分です。新しい指標ではなく、"
        "同じ測定値を形として見せているだけです。\n"
        "- 任意で、対応点を対称化位置へ移動させ、その間の画素を Delaunay 三角形ごとに変形した"
        "**対称化画像**も生成できます（既定 OFF）。画像の四隅と辺の中点を固定するため、"
        "ランドマークが覆わない髪・耳・首・背景は外周を保ったまま緩やかに追従し、"
        "境界付近に不自然さが残ることがあります。\n"
        "- 変形量が両眼間隔の 25% を超える場合は、画像を生成せず理由を表示します"
        "（検出が破綻した状態で人物として成立しない画像を作らないためです）。\n"
        "- 生成されるのは幾何変形された加工画像です。無加工の写真ではなく、より正しい・望ましい"
        "顔でもありません。加工画像であることを明示せずに提示・共有しないでください。"
    ): (
        "- For each corresponding left-right pair, we compute the **midpoint** "
        'of the measured point and its mirrored point as the "symmetrised '
        'position". The midpoint is the foot of the perpendicular to the '
        "centre line.\n"
        "- Points with no defined corresponding pair (most of the 478 points) "
        "are left unchanged. We do not guess a correspondence to fill this "
        "in.\n"
        "- The distance between a measured point and its symmetrised point is "
        "exactly half of the RMS distance above. This is not a new metric; it "
        "is simply the same measured value shown as a shape.\n"
        "- Optionally, you can also generate a **symmetrised image** (default "
        "OFF) by moving the corresponding points to their symmetrised "
        "positions and warping the pixels between them, triangle by triangle "
        "(Delaunay). Because the four corners and the edge midpoints of the "
        "image are fixed, hair, ears, neck, and background that the landmarks "
        "do not cover follow along loosely while keeping the outer boundary, "
        "and some unnatural artefacts can remain near the boundary.\n"
        "- If the amount of warping exceeds 25% of the eye-to-eye distance, "
        "we do not generate the image and instead show the reason (so we do "
        "not produce an image that no longer looks like a person, from a "
        "broken detection).\n"
        "- What is generated is a geometrically warped, processed image. It "
        "is not an unedited photo, and it is not a more correct or more "
        "desirable face. Do not present or share it without stating that it "
        "is a processed image."
    ),
    "参考年齢・参考性別": "Reference age & gender",
    (
        "- Open Model Zoo 2023.0 FP16 `age-gender-recognition-retail-0013` を、"
        "検証済みの公式 XML/BIN から再現可能な開発時変換で生成した固定 ONNX として "
        "`opencv-contrib-python` の DNN 機能で端末内推論します。\n"
        "- 入力は検出した単一顔の landmark bbox から作る BGR 62×62 の crop で、"
        "顔埋め込みは生成・保持しません。\n"
        "- 出力は連続値の年齢と `[female, male]` の 2 クラス確率で、対象は 18〜75 歳です。"
        "公開ベンチマークでは年齢平均誤差 6.99 年、性別accuracy 95.80% ですが、"
        "内部validation datasetの値であり、実利用時の精度を保証しません。\n"
        "- 年齢は「モデル推定の参考年齢」、性別は「外見からのモデル推定の参考性別」であり、"
        "本人の自己認識・法的性別・実年齢を表しません。左右差とは分離して扱います。"
    ): (
        "- We run Open Model Zoo 2023.0 FP16 `age-gender-recognition-retail-"
        "0013` on-device, using the DNN feature of `opencv-contrib-python`, as "
        "a fixed ONNX file generated by a reproducible development-time "
        "conversion from the verified official XML/BIN.\n"
        "- The input is a BGR 62x62 crop built from the landmark bounding box "
        "of the single detected face; we do not generate or retain a face "
        "embedding.\n"
        "- The output is a continuous age value and a 2-class `[female, "
        "male]` probability, and the target range is ages 18-75. On the "
        "published benchmark, the mean age error is 6.99 years and gender "
        "accuracy is 95.80%, but these come from an internal validation "
        "dataset and do not guarantee accuracy in actual use.\n"
        '- Age is treated as "Reference age (model estimate)" and gender as '
        '"Reference gender (model estimate from appearance)"; neither '
        "represents self-identified gender, legal gender, or actual age. They "
        "are handled separately from symmetry."
    ),
    "主な誤差要因": "Main sources of error",
    (
        "- 顔の向き、照明、影、レンズ歪み、ランドマーク検出誤差\n"
        "- 化粧、髭、眼鏡、頭部被覆\n"
        "- カメラの自動露出・ホワイトバランス・圧縮処理\n"
        "- 年齢・性別推定は、肌色、年齢層、照明条件によって精度が変わり得ます"
    ): (
        "- Head pose, lighting, shadows, lens aberration, landmark-detection "
        "error\n"
        "- Makeup, facial hair, glasses, head coverings\n"
        "- The camera's auto-exposure, white balance, and compression "
        "processing\n"
        "- Age and gender estimation accuracy can vary with skin tone, age "
        "group, and lighting conditions"
    ),
    "精度向上のために必要なこと": "What would help improve accuracy",
    (
        "- 固定評価セットでの距離分布の把握（参考統計であり、係数の校正ではありません）\n"
        "- 実写での姿勢角符号の確認\n"
        "- 3D 正規化との比較評価\n"
        "- 照明・ぼけを検知するゲートの追加\n"
        "- 同一端末での再撮影評価"
    ): (
        "- Understanding the distance distribution on a fixed evaluation set "
        "(a reference statistic, not coefficient calibration)\n"
        "- Verifying the sign convention of pose angles in real photos\n"
        "- Comparative evaluation against 3D normalisation\n"
        "- Adding a gate that detects lighting and blur\n"
        "- Re-capture evaluation on the same device"
    ),
    "使用しないでください": "Do not use this for",
    (
        "- 医療、本人確認、採用、保険、教育、監視、人物比較・人物評価への利用\n"
        "- **年齢認証、未成年判定、入場制限への利用。** "
        "参考年齢を表示してよいかを決める内部ゲートがありますが、"
        "これは年齢を判定する仕組みではありません。"
        "乳幼児の顔の幾何を見分けるだけで、学童期以上の年齢や 76 歳以上は判定できません。\n"
        "- 本人以外、または明示的な権限・同意のない人物の画像の使用\n"
        "- 専用の同意・保護設計なしの未成年者への利用\n"
        "- 対称度の高低による感情的な祝福・警告、人物間比較、平均顔との比較、ランキング\n"
        "- **利用者自身の測定結果と参照統計を結び付けること。** "
        "参照統計のページ（参照ページ）は利用者の測定値を持ち込まない閲覧専用で、"
        "両者を同じ画面へ並べません。"
    ): (
        "- Medical use, identity verification, hiring, insurance, education, "
        "surveillance, or comparing/evaluating people\n"
        "- **Use for age verification, determining legal adulthood, or "
        "restricting entry.** There is an internal gate that decides whether "
        "a reference age may be shown, but it is not a mechanism for "
        "determining age. It can only distinguish the facial geometry of "
        "infants, and cannot determine ages from school age upward, or ages "
        "76 and over.\n"
        "- Using images of anyone other than yourself, or of a person "
        "without explicit authorisation and consent\n"
        "- Use with children without a dedicated consent and safeguarding "
        "design\n"
        "- Emotional celebration or warning based on how high or low the "
        "symmetry score is, comparing people to each other, comparing "
        "against an average face, or turning it into a leaderboard\n"
        "- **Linking your own measurement result to the reference "
        "statistics.** The reference statistics page is a view-only page "
        "that does not bring in your own measurement values, and the two "
        "are never placed on the same screen."
    ),
    "プライバシーとデータの扱い": "Privacy and data handling",
    (
        "- 画像・映像・ランドマーク・年齢帯・左右差の数値は外部へ送信せず、暗黙に保存もしません。\n"
        "- カメラは `127.0.0.1` のみで待ち受け、STUN/TURN や利用統計は既定で無効です。\n"
        "- 画像アップロードやカメラ利用は、事前の同意画面を完了するまで開始しません。\n"
        "- モデルは自動取得・自動更新せず、版と SHA-256 を追跡してエクスポート結果に記録します。\n"
        "- **結果レポート（HTML）を保存すると、顔写真を含む図がファイルへ入ります。**"
        "アプリが自動でファイルを作ることはなく、保存はダウンロード操作のときだけです。"
        "図を入れるかどうかは保存の直前で切り替えられます。"
        "保存した後のファイルはアプリの管理外で、共有先や複製をアプリからは制御できません。"
    ): (
        "- Images, video, landmarks, age band, and symmetry values are never "
        "sent externally and are never saved implicitly.\n"
        "- The camera listens only on `127.0.0.1`, and STUN/TURN and usage "
        "telemetry are disabled by default.\n"
        "- Image upload and camera use do not start until you complete the "
        "consent screen beforehand.\n"
        "- Models are never fetched or updated automatically; their version "
        "and SHA-256 are tracked and recorded in the exported results.\n"
        "- **Saving the result report (HTML) puts figures that include face "
        "photos into the file.** The app never creates a file automatically; "
        "saving happens only when you use the download action. You can "
        "toggle whether figures are included right before saving. Once "
        "saved, the file is outside the app's control, and the app cannot "
        "control where it is shared or copied."
    ),
    # --- ui/reference_page.py -------------------------------------------------
    "すべて（絞り込まない）": "All (no filter)",
    "絞り込み": "Filters",
    "未選択の軸は絞り込みに使いません（全件が対象になります）。": (
        "An axis left unselected is not used to filter (all cases are included)."
    ),
    "軸をまたぐ条件は同時に満たすものだけが残り、"
    "同じ軸の複数選択はそのいずれかに一致するものが残ります。": (
        "Conditions across axes must all be met at once; multiple selections "
        "within the same axis keep anything matching any one of them."
    ),
    "人種": "Race",
    "性別": "Gender",
    "年齢の指定方法": "How to specify age",
    "年齢帯で選ぶ": "Choose by age band",
    "1歳刻みの範囲で選ぶ": "Choose by a 1-year range",
    "年齢帯": "Age band",
    "年齢の範囲（1歳刻み）": "Age range (1-year steps)",
    "一致したセル一覧（{count} 件）": "List of matching cells ({count})",
    "一致するセルがありません。": "No matching cells.",
    "一致した件数": "Matching count",
    "条件に一致するデータがありません。人種・性別・年齢の条件を変更してください。": (
        "No data matches the conditions. Change the race, gender, or age conditions."
    ),
    "一致した件数が {threshold} 件未満のため、平均・標準偏差は表示しません"
    "（件数が少ない集計は誤差が大きく、参考になりません）。": (
        "The matching count is below {threshold}, so the mean and standard "
        "deviation are not shown (a small sample has too much error to be "
        "useful as a reference)."
    ),
    "部位別・全体の平均対称度（%）": "Mean symmetry by part and overall (%)",
    "全体": "Overall",
    "この画面についての注意": "About this page",
    (
        "- この結果は探索値であり、"
        "正常値・目標値・個人評価・人物間比較には使用できません。\n"
        "- 群間の差は、撮影条件・注釈・構成比、"
        "ランドマーク検出の群ごとの挙動差を含みうるため、原因を断定できません。\n"
        "- 人種・性別・年齢は UTKFace の配布時の注釈であり、"
        "本アプリやモデルの推定ではありません。\n"
        "- この画面は平均と標準偏差だけを扱います。"
        "中央値と 95% 信頼区間はセル集計から復元できないため、v1 では表示しません。\n"
    ): (
        "- This result is an exploratory value and cannot be used as a "
        "typical value, a target, for individual evaluation, or for "
        "comparing people.\n"
        "- Differences between groups can include shooting conditions, "
        "annotation, composition, and differences in how landmark detection "
        "behaves for each group, so the cause cannot be determined.\n"
        "- Race, gender, and age are annotations from the UTKFace "
        "distribution, not estimates made by this app or its models.\n"
        "- This page handles only the mean and standard deviation. The "
        "median and a 95% confidence interval cannot be recovered from the "
        "cell aggregates, so v1 does not show them.\n"
    ),
    "- {name} は非商用研究限定のデータセットです（{scope} · {url}）。": (
        "- {name} is a dataset for non-commercial research use only ({scope} · {url})."
    ),
    "配布元の注意書き（原文）": "Source notice (original text)",
    "版とデータ出典": "Version and data provenance",
    "- 指標の版: `{value}`": "- Metric version: `{value}`",
    "- 参照データの schema: `{value}`": "- Reference data schema: `{value}`",
    "- セル CSV の sha256: `{value}…`": "- Cell CSV sha256: `{value}…`",
    "- ランドマーク対応表の版: `{value}`": "- Landmark map version: `{value}`",
    "参照データを読み込めないため、この画面は利用できません（{reason}）。": (
        "This page is unavailable because the reference data could not be loaded ({reason})."
    ),
    "参照統計ブラウザ（UTKFace）": "Reference statistics browser (UTKFace)",
    "探索的な記述統計です。正常値・目標値・個人評価・人物間比較には使えません。": (
        "An exploratory descriptive statistic. It cannot be used as a typical "
        "value, a target, for individual evaluation, or for comparing people."
    ),
    "事前に集計した参照データセット（UTKFace、非商用研究限定）を表示します。"
    "この画面にはカメラ・画像の分析結果を表示しません。": (
        "Shows a pre-aggregated reference dataset (UTKFace, non-commercial "
        "research use only). This page does not show Camera or Image "
        "analysis results."
    ),
    "結果": "Results",
}
