# Test Index

このファイルは production code、利用者フロー、要件 ID と test の対応を示す。test を追加・移動・削除した場合は同じ変更で更新する。

| Test | Production / flow | Requirements |
|---|---|---|
| `tests/domain/test_models.py` | `domain.models` の分析状態・左右差距離DTO（0以上・上限なし・負値拒否） | FR-008, FR-011, NFR-013 |
| `tests/domain/test_demographics.py` | 18〜75歳の連続年齢、性別2確率、時刻、model metadata検証 | FR-022, FR-023, AC-AGE-01 |
| `tests/domain/test_landmarks.py` | landmark DTO の検証と画素座標変換 | FR-008, FR-009, NFR-013 |
| `tests/domain/test_overlay.py` | ランドマーク・TLS中心線・対称化ズレ描画の純関数性・境界クランプ・不正入力拒否 | FR-013, FR-034 |
| `tests/domain/test_skin_quality.py` | camera色補正、頬の色調・texture参考値、欠測、未校正状態 | FR-025, AC-APPEARANCE-01, NFR-013 |
| `tests/domain/test_export.py` | export record v3.0の有限値・欠測・距離列（`score_*`廃止、負値拒否、上限なし）・肌表面／年齢・性別contract | FR-017, FR-023, FR-025, NFR-013, NFR-020 |
| `tests/domain/test_metrics.py` | 指標設定の検証（対応マップのみ。感度・重みfieldの不在、`all_pairs`のプールと`overall_min_pairs`） | FR-011, FR-018, NFR-013, AC-SCORE-04, §15.2 |
| `tests/domain/test_geometry.py` | 正規化の不変性と TLS 中心線・鏡映 | FR-011, AC-SCORE-02, AC-SCORE-03 |
| `tests/domain/test_warp_mesh.py` | 境界アンカー生成、変位ガード（両眼間隔比・拒否）、範囲clamp、退化三角形判定、不正入力拒否 | FR-035, AC-SYMM-02, §8.6.4 |
| `tests/domain/test_symmetrization.py` | 正規化逆変換の往復一致、対称化点＝ペア中点／正中線への射影、対応未定義点の不変、ズレ＝§8.5距離の半分、入力非破壊 | FR-034, AC-SYMM-01, §8.6.2 |
| `tests/domain/test_symmetry.py` | 部位別RMS距離と全ペアプールの全体距離、単調性、不変性、欠測（部位欠測が全体を欠測にしないこと）、上限なし | FR-011, AC-SCORE-01〜04 |
| `tests/domain/test_pose.py` | 極分解と intrinsic Y-X-Z の符号 contract | FR-009, §8.2.1, OD-04 |
| `tests/domain/test_smoothing.py` | 時間ベース EMA と reset 条件、`tau_seconds`読み出し | FR-012, §8.7, §8.8 |
| `tests/domain/test_tracking_zoom.py` | 追跡ズームの窓計算・座標写像・拡大率の純粋幾何 | FR-026, §14.3a, AC-ZOOM-01 |
| `tests/domain/test_frame_adjustments.py` | 明るさgainの乗算・clip、背景合成のmask blend・検証、mean_luminance | FR-027, FR-028, §14.3b, AC-DISPLAY-01 |
| `tests/domain/test_auto_tuning.py` | 明るさ（生映像の輝度から絶対gainを直接算出）・追跡ズーム・cadence（analysis/age_interval_ms・history_capacity）の自動調整heuristicの範囲clamp・不正入力拒否（感度調整は§8.6で廃止） | FR-026, FR-028, FR-029, FR-030, §14.3b, §14.3c, §14.3d, AC-AUTOTUNE-01, AC-AUTOTUNE-02 |
| `tests/application/test_image_input.py` | JPEG / PNG uploadの形式・容量・画素・animation・推論用顔crop検証 | FR-006, FR-007, FR-022, AC-INPUT-01 |
| `tests/application/test_realtime.py` | latest-frame cadence、frame drop、単調時刻、失敗分離、immutable snapshot、年齢・性別workerの提出/収集/失敗分離、Advanced設定（metric_config・history_capacity・age_interval_ms・analysis_interval_ms）の伝播、自動調整ON/OFFの明るさ・zoom policy反映と失敗許容、cadence自動調整（segment/EMA非破壊）と自動調整がanalyzer・recorderへ届かないこと（採点に影響する経路の不在）、§14.3gの一発キャプチャ（要求時のみ保持・生frameのcopy・1回で消費・単一顔待ち・失敗frame越しの存続・reset/closeで破棄）、幾何・mask共用のdisplay cache TTLとauto-tune連動、処理段階別タイミング（landmark_ms/segmentation_ms/scoring_ms）の内訳計測、明るさ補正がlandmark検出入力にのみ届きdisplay出力には届かないことの検証 | FR-008, FR-013, FR-022〜024, FR-028, FR-029, FR-030, FR-034, NFR-002, NFR-006, NFR-007, NFR-017, §14.3b, §15.2, §14.3c, §14.3d, §14.3f, AC-AUTOTUNE-01, AC-AUTOTUNE-02, AC-DISPLAY-01 |
| `tests/application/test_auto_tune_controller.py` | 自動調整の判断ロジック単体（明るさ・zoomの共有throttle、cadenceの最小sample数と不変時のNone、採点に影響する推奨を持たないこと、無効時は何も推奨しない、統計失敗の握り潰し） | FR-029, FR-030, §14.3c, §14.3d, AC-AUTOTUNE-01, AC-AUTOTUNE-02 |
| `tests/application/test_demographic_worker.py` | 年齢・性別推定workerのcapacity-1・独自cadence・segment世代破棄・失敗分離・`set_interval_ms`によるcadence編集 | FR-022〜024, §14.3, §15.2 |
| `tests/application/test_still_image.py` | 静止画ランドマークの stateless 検出、中心線切替、プレビュー生成 | FR-006, FR-008, FR-013 |
| `tests/application/test_symmetrization.py` | 対称化の一発処理（stateless・再現性、元画像非破壊、画素座標、複数顔・寸法なし・退化入力の拒否）、warper任意注入（未注入時は生成せず沈黙・対称化点を変形先として渡す・変位ガード超過と画像化失敗のfail-soft） | FR-034, FR-035, AC-SYMM-01, AC-SYMM-02, §14.3g |
| `tests/application/test_quality.py` | 品質ゲート、理由コード、OK/HOLD ヒステリシス | FR-010, §6.2〜6.4, AC-STATE-01 |
| `tests/application/test_analyzer.py` | ゲート→幾何距離／補正後の肌表面参考値→全体距離のEMAの統合と失敗分離、metric bundleが差し替え不可であること、`tau_seconds`読み出し | FR-011, FR-012, FR-025, AC-FACE-01, AC-APPEARANCE-01, §15.2 |
| `tests/application/test_session.py` | session履歴上限、segment、設定fingerprint（距離列・part_pair_counts）、肌表面・年齢・性別の合流、`reset`がmetric bundleを差し替えないこと・history_capacity resize、`resize_history_capacity`による非破壊resize（segment/revision不変） | FR-016, FR-017, FR-020, FR-023, FR-025, NFR-012, §15.2, §14.3d |
| `tests/runtime/test_frame_scheduler.py` | capacity 1 の latest-frame-wins gate | NFR-006, NFR-007 |
| `tests/runtime/test_display_cache.py` | 間引きフレームが再利用する幾何・mask共用のTTL slot（期限内の再読み出し、期限切れでの破棄、明示clear、並行store/readの一貫性） | FR-013, §1.2(17), AC-DISPLAY-01 |
| `tests/adapters/test_opencv_demographic_estimator.py` | 派生ONNX用BGR前処理、連続年齢・性別contract、失敗分離 | FR-022〜024, AC-AGE-02 |
| `tests/adapters/test_mediapipe_landmarker.py` | Face Landmarker contract（VIDEO / IMAGE）、単一顔方針、frame order、close | FR-008, FR-009, AC-FACE-01 |
| `tests/adapters/test_tracking_zoom_landmarker.py` | 追跡ズームのcrop/拡大wiring、見失い復帰、多顔recheck、close伝播、`policy`読み出し（fake provider） | FR-026, FR-008, §14.3a, AC-ZOOM-01 |
| `tests/adapters/test_mediapipe_segmenter.py` | Image Segmenter contract（VIDEO）、frame order、shape/finite検証、close | FR-027, §14.3b, AC-DISPLAY-01 |
| `tests/adapters/test_model_store.py` | manifest 検証、HTTPS/許可ホスト、取得再試行、sha256、atomic install | FR-002, AC-MODEL-01, AC-MODEL-02 |
| `tests/adapters/test_piecewise_warp.py` | Delaunay＋区分アフィンの画素変形（恒等メッシュでの完全一致、元画像非破壊、変位時の画素移動と方向、境界アンカーの固定、三角形のindex解決、重複点・辺上の点、サイズ/チャンネル不一致の拒否） | FR-035, AC-SYMM-02, §8.6.4 |
| `tests/adapters/test_metric_config.py` | 凍結 metric bundle の読み込みと構造不変条件 | FR-011, FR-020, AC-SCORE-04 |
| `tests/adapters/test_session_exporter.py` | CSV / JSON の同値性、manifest、ファイル名 | FR-017, AC-EXPORT-01 |
| `tests/model/test_real_landmarker.py` | 実 `.task` asset の load・検出・スコア smoke（未配置時は skip） | FR-002, FR-009, FR-011, §21.1 |
| `tests/model/test_demographic_conversion.py` | 検証済み公式IRからmanifest固定ONNXを決定的に再生成（asset／変換tool未配置時は skip）。合成IR fixtureによる、Result層の`/sink_port_0`命名衝突（重複ONNX node名）の回帰test | FR-002, FR-022〜024, AC-AGE-01〜02, §10.6, §21.1 |
| `tests/model/test_real_demographic.py` | manifest検証済み派生ONNXのOpenCV DNN load・出力shape smoke（未配置／runtime不備時は skip） | FR-002, FR-022〜024, AC-AGE-01〜02, §21.1 |
| `tests/model/test_openvino_parity.py` | OpenVINO参照runtimeと派生ONNXの固定入力parity（OD-07数値ゲート、openvino未導入時は skip） | FR-002, FR-022〜024, AC-AGE-01〜02, OD-07, §21.1 |
| `tests/model/test_onnx_runtime_agreement.py` | OpenVINO・cv2.dnn・onnxruntimeの三者比較で変換recipeのbugを切り分け（OD-07 due diligence、openvino／onnxruntime未導入時は skip） | FR-002, FR-022〜024, OD-07, §21.1 |
| `tests/model/test_real_segmenter.py` | 実asset由来のImage Segmenter load・出力shape/有限値smoke、domain合成との結合（未配置時は skip） | FR-002, FR-027, §14.3b, §21.1 |
| `tests/ui/test_about_page.py` | About ページが例外なく描画され、算出方法・禁止用途の見出しを含むこと（AppTest） | §4.1, FR-014, §10.7 |
| `tests/ui/test_app.py` | 年齢・性別モデル設定、dark theme、UI composition、入力方法選択（両モードの説明提示・既定Camera・選択解除時に直前のモードを維持・切替時の runtime 解放）、Image側対称化パネルの失敗表示と中立文言 | FR-001, FR-004, FR-022, FR-034, AC-FLOW-01, AC-A11Y-01 |
| `tests/ui/test_camera.py` | WebRTC 設定（ICE なし・映像のみ）、frame callbackの描画契約（幾何のみ描画／描くものが無ければ元frameをそのまま返し再encodeしない／解析はsnapshotへ届く／背景置換時のpts保持）、対称化キャプチャUI（arm・1回だけ消費・失敗の明示・クリアで保持破棄・価値判断語を使わない文言）、レイアウト（映像がタブより先・表示設定を映像より上に置かない・配置presetの読み書き分離） | FR-005, FR-013, FR-032, FR-034, NFR-006, NFR-009, §5.1, AC-PRIV-02, AC-LAYOUT-01, AC-SYMM-01 |
| `tests/ui/test_guidance.py` | 位置誘導メッセージの全ReasonCode網羅と、品質ゲート充足時にNoneを返すこと | FR-031, AC-GUIDE-01 |
| `tests/ui/test_scores.py` | Camera・Image共用の結果表示（追跡状態ごとの severity 割当、欠測時の `—` 表示、対称度＝`100%-ずれ%`の整形・下限clamp・両者の和が100%、見出しがずれを併記すること、中立文言、対称化画像の表示（既定OFFでは描画しない・拒否理由の明示・加工画像であることの明記・価値判断語を使わない文言）、詳細パネルが共通guidance表から理由を文言化すること） | FR-013, FR-014, NFR-013, §5.2, §5.3, §6.3, §8.6.3, §8.6.4, FR-035, AC-LAYOUT-01, AC-DISPLAY-02, AC-SYMM-02 |
| `tests/ui/test_model_setup.py` | manifest由来のlandmark・派生ONNX解決、元IR provenance・個別hash検証、状態表示 | FR-001, FR-002, FR-020, FR-024 |
| `tests/test_cli.py` | localhost 限定の Streamlit 起動 command | NFR-009, AC-SEC-01 |
| `tests/repository/test_generate_structure.py` | 安全で bounded な `STRUCTURE.md` 生成 | 開発効率・リポジトリ規約 |
| `tests/repository/test_testbox_documentation.py` | test file set と本索引の同期 | 開発効率・リポジトリ規約 |
