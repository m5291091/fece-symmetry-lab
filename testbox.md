# Test Index

このファイルは production code、利用者フロー、要件 ID と test の対応を示す。test を追加・移動・削除した場合は同じ変更で更新する。

| Test | Production / flow | Requirements |
|---|---|---|
| `tests/domain/test_models.py` | `domain.models` の分析状態・左右差 result | FR-008, FR-011, NFR-013 |
| `tests/domain/test_age.py` | logits→8 年齢帯、確率・時刻・model metadata 検証 | FR-022, FR-023, AC-AGE-01 |
| `tests/domain/test_landmarks.py` | landmark DTO の検証と画素座標変換 | FR-008, FR-009, NFR-013 |
| `tests/domain/test_overlay.py` | ランドマーク描画の純関数性と境界クランプ | FR-013 |
| `tests/domain/test_metrics.py` | 指標設定の検証（重み・感度・対応マップ） | FR-011, FR-018, NFR-013 |
| `tests/domain/test_geometry.py` | 正規化の不変性と TLS 中心線・鏡映 | FR-011, AC-SCORE-02, AC-SCORE-03 |
| `tests/domain/test_symmetry.py` | RMS→スコア変換、単調性、欠測、重み | FR-011, AC-SCORE-01〜03 |
| `tests/domain/test_pose.py` | 極分解と intrinsic Y-X-Z の符号 contract | FR-009, §8.2.1, OD-04 |
| `tests/domain/test_smoothing.py` | 時間ベース EMA と reset 条件 | FR-012, §8.7, §8.8 |
| `tests/application/test_image_input.py` | JPEG / PNG upload の形式・容量・画素・animation 検証 | FR-006, FR-007, AC-INPUT-01 |
| `tests/application/test_realtime.py` | frame drop、単調時刻、失敗分離、immutable snapshot | FR-008, FR-013, NFR-007, NFR-017 |
| `tests/application/test_still_image.py` | 静止画ランドマークの stateless 検出とプレビュー生成 | FR-006, FR-008, FR-013 |
| `tests/application/test_quality.py` | 品質ゲート、理由コード、OK/HOLD ヒステリシス | FR-010, §6.2〜6.4, AC-STATE-01 |
| `tests/application/test_analyzer.py` | ゲート→スコア→平滑化の統合と失敗分離 | FR-011, FR-012, AC-FACE-01 |
| `tests/runtime/test_frame_scheduler.py` | capacity 1 の latest-frame-wins gate | NFR-006, NFR-007 |
| `tests/adapters/test_onnx_age_estimator.py` | ONNX 前処理、8 logits contract、失敗分離 | FR-022〜024, AC-AGE-02 |
| `tests/adapters/test_mediapipe_landmarker.py` | Face Landmarker contract（VIDEO / IMAGE）、単一顔方針、frame order、close | FR-008, FR-009, AC-FACE-01 |
| `tests/adapters/test_model_store.py` | manifest 検証、HTTPS/許可ホスト、sha256、atomic install | FR-002, AC-MODEL-01, AC-MODEL-02 |
| `tests/adapters/test_metric_config.py` | 凍結 metric bundle の読み込みと構造不変条件 | FR-011, FR-020, AC-SCORE-04 |
| `tests/model/test_real_landmarker.py` | 実 `.task` asset の load・検出・スコア smoke（未配置時は skip） | FR-002, FR-009, FR-011, §21.1 |
| `tests/ui/test_app.py` | 年齢モデル設定と UI composition | FR-001, FR-022, AC-FLOW-01 |
| `tests/ui/test_camera.py` | WebRTC 設定（ICE なし・映像のみ）と frame callback | FR-005, NFR-009, AC-PRIV-02 |
| `tests/ui/test_model_setup.py` | manifest 由来のモデル解決と検証状態表示 | FR-001, FR-002, FR-020 |
| `tests/test_cli.py` | localhost 限定の Streamlit 起動 command | NFR-009, AC-SEC-01 |
| `tests/repository/test_generate_structure.py` | 安全で bounded な `STRUCTURE.md` 生成 | 開発効率・リポジトリ規約 |
| `tests/repository/test_testbox_documentation.py` | test file set と本索引の同期 | 開発効率・リポジトリ規約 |
