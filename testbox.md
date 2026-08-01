# Test Index

このファイルは production code、利用者フロー、要件 ID と test の対応を示す。test を追加・移動・削除した場合は同じ変更で更新する。

| Test | Production / flow | Requirements |
|---|---|---|
| `tests/domain/test_models.py` | `domain.models` の分析状態・左右差 result | FR-008, FR-011, NFR-013 |
| `tests/domain/test_demographics.py` | 18〜75歳の連続年齢、性別2確率、時刻、model metadata検証 | FR-022, FR-023, AC-AGE-01 |
| `tests/domain/test_landmarks.py` | landmark DTO の検証と画素座標変換 | FR-008, FR-009, NFR-013 |
| `tests/domain/test_overlay.py` | ランドマークとTLS中心線描画の純関数性・境界クランプ | FR-013 |
| `tests/domain/test_skin_quality.py` | camera色補正、頬の色調・texture参考値、欠測、未校正状態 | FR-025, AC-APPEARANCE-01, NFR-013 |
| `tests/domain/test_export.py` | export record v2.0の有限値・欠測・肌表面／年齢・性別contract | FR-017, FR-023, FR-025, NFR-013, NFR-020 |
| `tests/domain/test_metrics.py` | 指標設定の検証（重み・感度・対応マップ） | FR-011, FR-018, NFR-013 |
| `tests/domain/test_geometry.py` | 正規化の不変性と TLS 中心線・鏡映 | FR-011, AC-SCORE-02, AC-SCORE-03 |
| `tests/domain/test_symmetry.py` | RMS→スコア変換、単調性、欠測、重み | FR-011, AC-SCORE-01〜03 |
| `tests/domain/test_pose.py` | 極分解と intrinsic Y-X-Z の符号 contract | FR-009, §8.2.1, OD-04 |
| `tests/domain/test_smoothing.py` | 時間ベース EMA と reset 条件 | FR-012, §8.7, §8.8 |
| `tests/application/test_image_input.py` | JPEG / PNG uploadの形式・容量・画素・animation・推論用顔crop検証 | FR-006, FR-007, FR-022, AC-INPUT-01 |
| `tests/application/test_realtime.py` | latest-frame cadence、frame drop、単調時刻、失敗分離、immutable snapshot | FR-008, FR-013, NFR-006, NFR-007, NFR-017 |
| `tests/application/test_still_image.py` | 静止画ランドマークの stateless 検出、中心線切替、プレビュー生成 | FR-006, FR-008, FR-013 |
| `tests/application/test_quality.py` | 品質ゲート、理由コード、OK/HOLD ヒステリシス | FR-010, §6.2〜6.4, AC-STATE-01 |
| `tests/application/test_analyzer.py` | ゲート→幾何／補正後の肌表面参考値→平滑化の統合と失敗分離 | FR-011, FR-012, FR-025, AC-FACE-01, AC-APPEARANCE-01 |
| `tests/application/test_session.py` | session履歴上限、segment、設定fingerprint、肌表面・年齢・性別の合流 | FR-016, FR-017, FR-020, FR-023, FR-025, NFR-012 |
| `tests/runtime/test_frame_scheduler.py` | capacity 1 の latest-frame-wins gate | NFR-006, NFR-007 |
| `tests/adapters/test_opencv_demographic_estimator.py` | OpenVINO IR用BGR前処理、連続年齢・性別contract、失敗分離 | FR-022〜024, AC-AGE-02 |
| `tests/adapters/test_mediapipe_landmarker.py` | Face Landmarker contract（VIDEO / IMAGE）、単一顔方針、frame order、close | FR-008, FR-009, AC-FACE-01 |
| `tests/adapters/test_model_store.py` | manifest 検証、HTTPS/許可ホスト、sha256、atomic install | FR-002, AC-MODEL-01, AC-MODEL-02 |
| `tests/adapters/test_metric_config.py` | 凍結 metric bundle の読み込みと構造不変条件 | FR-011, FR-020, AC-SCORE-04 |
| `tests/adapters/test_session_exporter.py` | CSV / JSON の同値性、manifest、ファイル名 | FR-017, AC-EXPORT-01 |
| `tests/model/test_real_landmarker.py` | 実 `.task` asset の load・検出・スコア smoke（未配置時は skip） | FR-002, FR-009, FR-011, §21.1 |
| `tests/model/test_real_demographic.py` | manifest検証済みOpenVINO IRのOpenCV DNN load・出力shape smoke（未配置／runtime不備時は skip） | FR-002, FR-022〜024, AC-AGE-01〜02, §21.1 |
| `tests/ui/test_app.py` | 年齢・性別モデル設定、dark theme、UI composition | FR-001, FR-022, AC-FLOW-01, AC-A11Y-01 |
| `tests/ui/test_camera.py` | WebRTC 設定（ICE なし・映像のみ）、sample 間の直近数値表示、frame callback | FR-005, FR-013, NFR-006, NFR-009, AC-PRIV-02 |
| `tests/ui/test_camera_overlay.py` | カメラ映像内の高コントラストなスコア・姿勢・sample 状態表示 | FR-013, FR-014, AC-A11Y-01 |
| `tests/ui/test_model_setup.py` | manifest由来のlandmark・OpenVINO XML/BIN解決、個別hash検証、状態表示 | FR-001, FR-002, FR-020, FR-024 |
| `tests/test_cli.py` | localhost 限定の Streamlit 起動 command | NFR-009, AC-SEC-01 |
| `tests/repository/test_generate_structure.py` | 安全で bounded な `STRUCTURE.md` 生成 | 開発効率・リポジトリ規約 |
| `tests/repository/test_testbox_documentation.py` | test file set と本索引の同期 | 開発効率・リポジトリ規約 |
