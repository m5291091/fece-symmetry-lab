# Repository Structure

This is the compact, AI-facing repository index for `face-symmetry-lab`. It is an exploration aid, not a specification. `requirement.md` remains the product source of truth and `CLAUDE.md` remains the execution policy.

Read this file once at the start of a task. After locating the relevant module, inspect only that module, its corresponding tests, model card, and requirement sections.

## Entry points

| Concern | Path |
|---|---|
| Product requirements | `requirement.md` |
| AI and development policy | `CLAUDE.md` |
| Test traceability | `testbox.md` |
| Streamlit application | `src/face_symmetry_lab/ui/app.py` |
| About page (methodology / constraints / models / privacy) | `src/face_symmetry_lab/ui/about_page.py` |
| 参照統計ブラウザ（人種・性別・年齢での絞り込み閲覧、FR-037） | `src/face_symmetry_lab/ui/reference_page.py` |
| 参照統計の凍結セルテーブル（§9.5、アプリが読む唯一の参照データ） | `configs/reference/utkface_symmetry3d_cells_v1.csv` |
| 吹き出し・HUD へ併記する参照平均の解決（FR-039） | `src/face_symmetry_lab/ui/baseline.py` |
| 参照統計セルテーブルの生成 (dev-only、アプリからは呼ばない) | `scripts/build_reference_dataset.py` |
| 参照統計の生成元 notebook（再現性の正本。`src/` からは参照しない） | `research/utkface-symmetry/utkface_symmetry_lab.ipynb` |
| Realtime camera UI | `src/face_symmetry_lab/ui/camera.py` |
| 映像へ重ねる HUD の組み立てと配線 | `src/face_symmetry_lab/ui/hud.py` |
| HUD がブラウザへ渡す CSS・実行時 JS | `src/face_symmetry_lab/ui/hud_assets.py` |
| 表示の落ち着き（移動平均の間隔・更新間隔）を選ぶ UI（FR-040, FR-041） | `src/face_symmetry_lab/ui/smoothing_panel.py` |
| 表示値を一定時間だけ差し替えないラッチ（FR-041） | `src/face_symmetry_lab/domain/display_hold.py` |
| セッション要約の記述統計（FR-021） | `src/face_symmetry_lab/domain/session_summary.py` |
| 表示できた参考年齢・参考性別の観測（FR-039、レポートの母集団） | `src/face_symmetry_lab/application/demographic_observations.py` |
| セッション要約の描画（FR-021） | `src/face_symmetry_lab/ui/summary_panel.py` |
| 結果レポート（HTML）の組み立て（FR-042） | `src/face_symmetry_lab/ui/report.py` |
| レポートへ入れる図の組み立て（FR-042） | `src/face_symmetry_lab/ui/report_figures.py` |
| 図の縮小と data URI 符号化（FR-042） | `src/face_symmetry_lab/adapters/imaging/image_encoding.py` |
| Image 入力モード一式 | `src/face_symmetry_lab/ui/image_section.py` |
| 表示値の解決と整形（HUD・吹き出し・詳細で共有） | `src/face_symmetry_lab/ui/readings.py` |
| Camera / Image 共有の対称化パネル | `src/face_symmetry_lab/ui/symmetrization_panel.py` |
| Domain models | `src/face_symmetry_lab/domain/` |
| Upload validation | `src/face_symmetry_lab/application/image_input.py` |
| Realtime use case | `src/face_symmetry_lab/application/realtime.py` |
| 解析周期と表示キャッシュ TTL | `src/face_symmetry_lab/application/analysis_cadence.py` |
| 表示専用の合成（描画・背景置換・表示キャッシュ） | `src/face_symmetry_lab/application/preview_compositor.py` |
| 一発取り込みバッファ | `src/face_symmetry_lab/application/frame_capture.py` |
| 年齢・性別 worker との中継役 | `src/face_symmetry_lab/application/demographic_relay.py` |
| 自動調整の判断ロジック | `src/face_symmetry_lab/application/auto_tune_controller.py` |
| Realtime age/gender worker | `src/face_symmetry_lab/application/demographic_worker.py` |
| Session history / export records | `src/face_symmetry_lab/application/session.py` |
| Still-image landmarks | `src/face_symmetry_lab/application/still_image.py` |
| Runtime ports | `src/face_symmetry_lab/application/ports.py` |
| Pinned model manifest | `configs/model-manifest.json` |
| Frozen metric bundle | `configs/metrics/symmetry-2d-v1.json` |
| 3D object symmetry の凍結 bundle（並走） | `configs/metrics/symmetry-3d-v1.json` |
| Landmark map derivation | `scripts/derive_landmark_map.py` |
| Symmetry metric | `src/face_symmetry_lab/domain/symmetry.py` |
| 3D object symmetry 指標（§8.5.1、並走） | `src/face_symmetry_lab/domain/symmetry_3d.py` |
| 「全体のスコア」の表示スケール（§8.6.7） | `src/face_symmetry_lab/domain/score_scale.py` |
| 「全体のスコア」の計算過程 DTO（§8.6.3 H4） | `src/face_symmetry_lab/domain/score_explanation.py` |
| 対称化ビジュアライゼーション（純粋幾何） | `src/face_symmetry_lab/domain/symmetrization.py` |
| 対称化の一発処理 | `src/face_symmetry_lab/application/symmetrization.py` |
| 対称化画像のピクセルワーピング | `src/face_symmetry_lab/adapters/imaging/piecewise_warp.py` |
| Camera correction / skin-surface metric | `src/face_symmetry_lab/domain/skin_quality.py` |
| Quality gate / hysteresis | `src/face_symmetry_lab/application/quality.py` |
| Frame analysis orchestration | `src/face_symmetry_lab/application/analyzer.py` |
| Landmark model card | `docs/model-cards/mediapipe-face-landmarker-v1.md` |
| Model store (hash + atomic install) | `src/face_symmetry_lab/adapters/io/model_store.py` |
| Model preparation CLI | `scripts/bootstrap_models.py` |
| OpenVINO reference parity (OD-07, dev-only) | `scripts/verify_openvino_parity.py` |
| Cross-ONNX-runtime agreement (OD-07 due diligence, dev-only) | `scripts/verify_onnx_runtime_agreement.py` |
| 実写での左右差ノイズ床・左右バイアス・注入ゲイン測定 (diagnostic, dev-only) | `scripts/measure_metric_noise.py` |
| 追跡ズーム経路の `z` スケール写像の実測 (diagnostic, dev-only) | `scripts/measure_zoom_depth_scale.py` |
| セッション間の表示値ばらつきと撮影共変量の実測 (diagnostic, dev-only) | `scripts/measure_session_variation.py` |
| within-take の姿勢応答（pitch と表示値）の実測 (diagnostic, dev-only) | `scripts/measure_pose_response.py` |
| Frame admission gate | `src/face_symmetry_lab/runtime/frame_scheduler.py` |
| 間引きフレーム用のTTL cache | `src/face_symmetry_lab/runtime/display_cache.py` |
| MediaPipe landmark adapter | `src/face_symmetry_lab/adapters/inference/mediapipe_landmarker.py` |
| Tracking-zoom landmark wrapper | `src/face_symmetry_lab/adapters/inference/tracking_zoom_landmarker.py` |
| Tracking-zoom geometry | `src/face_symmetry_lab/domain/tracking_zoom.py` |
| MediaPipe segmentation adapter | `src/face_symmetry_lab/adapters/inference/mediapipe_segmenter.py` |
| Brightness / background compositing | `src/face_symmetry_lab/domain/frame_adjustments.py` |
| 明るさ・追跡ズームの自動調整heuristic | `src/face_symmetry_lab/domain/auto_tuning.py` |
| Segmentation model card | `docs/model-cards/mediapipe-selfie-segmenter.md` |
| Age / gender adapter | `src/face_symmetry_lab/adapters/inference/opencv_demographic_estimator.py` |
| Age / gender model card | `docs/model-cards/openvino-age-gender-retail-0013.md` |
| Architecture decisions | `docs/adr/` |
| Repository verification | `scripts/verify.py` |

## Requirement routing

| Capability | Production | Tests | Requirements |
|---|---|---|---|
| Analysis DTO | `domain/models.py` | `tests/domain/test_models.py` | FR-008, FR-011 |
| Landmark DTO / pixel mapping（推定奥行き `z` を含む） | `domain/landmarks.py` | `tests/domain/test_landmarks.py` | FR-008, FR-009, §8.1.1 |
| Landmark overlay painting | `domain/overlay.py` | `tests/domain/test_overlay.py` | FR-013, FR-034 |
| Export record DTO | `domain/export.py` | `tests/domain/test_export.py` | FR-017, NFR-020 |
| Metric bundle validation | `domain/metrics.py` | `tests/domain/test_metrics.py` | FR-011, FR-018 |
| Normalization / TLS midline / 3D 正規化 | `domain/geometry.py` | `tests/domain/test_geometry.py` | FR-011, FR-036, AC-SCORE-02, §8.5.1 |
| Symmetry distances | `domain/symmetry.py` | `tests/domain/test_symmetry.py` | FR-011, AC-SCORE-01, AC-SCORE-04 |
| 3D object symmetry 距離（並走） | `domain/symmetry_3d.py` | `tests/domain/test_symmetry_3d.py` | FR-036, AC-SCORE-05〜09, §8.5.1, §8.6.5 |
| 「全体のスコア」の表示スケール | `domain/score_scale.py` | `tests/domain/test_score_scale.py` | FR-013, §8.6.7, AC-DISPLAY-05 |
| 「全体のスコア」の計算過程 DTO | `domain/score_explanation.py` | `tests/domain/test_score_explanation.py` | FR-013, §8.6.3, §8.6.5, §8.6.7, AC-DISPLAY-04 |
| 対称化点群・ズレ | `domain/symmetrization.py` | `tests/domain/test_symmetrization.py` | FR-034, AC-SYMM-01, §8.6.2 |
| 変形メッシュ・安全ガード | `domain/warp_mesh.py` | `tests/domain/test_warp_mesh.py` | FR-035, AC-SYMM-02, §8.6.4 |
| Delaunay＋区分アフィン変形 | `adapters/imaging/piecewise_warp.py` | `tests/adapters/test_piecewise_warp.py` | FR-035, AC-SYMM-02, §8.6.4 |
| Skin-surface reference scoring | `domain/skin_quality.py` | `tests/domain/test_skin_quality.py` | FR-025, AC-APPEARANCE-01 |
| Head pose | `domain/pose.py` | `tests/domain/test_pose.py` | FR-009, §8.2.1 |
| Display smoothing（時定数の値域・段・導出値を含む） | `domain/smoothing.py` | `tests/domain/test_smoothing.py` | FR-012, FR-040, §8.7, §8.8 |
| 表示の落ち着きの選択 UI | `ui/smoothing_panel.py` | `tests/ui/test_smoothing_panel.py` | FR-040, FR-041, §8.7, §15.1, AC-SMOOTH-01, AC-HOLD-01 |
| 表示ラッチ（更新間隔） | `domain/display_hold.py` | `tests/domain/test_display_hold.py` | FR-041, §15.1, AC-HOLD-01, AC-HOLD-02 |
| セッション要約（記述統計） | `domain/session_summary.py` | `tests/domain/test_session_summary.py` | FR-021, §10.7, §14.4, AC-SUMMARY-01, AC-SUMMARY-02 |
| セッション要約の描画 | `ui/summary_panel.py` | `tests/ui/test_summary_panel.py` | FR-021, §5.2, §5.3, §10.7, AC-SUMMARY-01, AC-SUMMARY-02 |
| 結果レポート（HTML） | `ui/report.py` | `tests/ui/test_report.py` | FR-042, §9.6, §10.7, AC-REPORT-01, AC-REPORT-02 |
| レポートの図の組み立て | `ui/report_figures.py` | `tests/ui/test_report_figures.py` | FR-042, FR-034, FR-035, AC-REPORT-03 |
| 図の縮小と data URI 符号化 | `adapters/imaging/image_encoding.py` | `tests/adapters/test_image_encoding.py` | FR-042, §10.4, AC-REPORT-02 |
| Demographic mapping | `domain/demographics.py` | `tests/domain/test_demographics.py` | FR-022, FR-023 |
| Safe image input | `application/image_input.py` | `tests/application/test_image_input.py` | FR-006, FR-007 |
| Realtime frame pipeline | `application/realtime.py` | `tests/application/test_realtime.py` | FR-013, NFR-017 |
| 解析周期・表示 TTL | `application/analysis_cadence.py` | `tests/application/test_analysis_cadence.py` | NFR-006, §1.2(17), §14.3d |
| 表示専用の合成 | `application/preview_compositor.py` | `tests/application/test_preview_compositor.py` | FR-013, FR-027, FR-028, §14.3b |
| 一発取り込みバッファ | `application/frame_capture.py` | `tests/application/test_frame_capture.py` | FR-034, §10.4, §14.3g |
| 年齢・性別の中継役 | `application/demographic_relay.py` | `tests/application/test_demographic_relay.py` | FR-022〜024, §14.3 |
| 自動調整の判断ロジック（cadence・表示のみ） | `application/auto_tune_controller.py` | `tests/application/test_auto_tune_controller.py` | FR-029, FR-030, AC-AUTOTUNE-01, AC-AUTOTUNE-02 |
| 年齢・性別の独立worker | `application/demographic_worker.py` | `tests/application/test_demographic_worker.py` | FR-022〜024, §14.3 |
| Session recording / fingerprint | `application/session.py` | `tests/application/test_session.py` | FR-016, FR-017, FR-020 |
| Quality gate | `application/quality.py` | `tests/application/test_quality.py` | FR-010, AC-STATE-01 |
| Analysis orchestration | `application/analyzer.py` | `tests/application/test_analyzer.py` | FR-011, FR-012 |
| Latest-frame-wins gate | `runtime/frame_scheduler.py` | `tests/runtime/test_frame_scheduler.py` | NFR-006, NFR-007 |
| 幾何・mask共用の表示TTL cache | `runtime/display_cache.py` | `tests/runtime/test_display_cache.py` | FR-013, §1.2(17), AC-DISPLAY-01 |
| Landmark detection | `adapters/inference/mediapipe_landmarker.py` | `tests/adapters/test_mediapipe_landmarker.py` | FR-008, FR-009, §8.1.1 |
| Tracking-zoom geometry | `domain/tracking_zoom.py` | `tests/domain/test_tracking_zoom.py` | FR-026, §14.3a, §8.1.1, AC-ZOOM-01 |
| Tracking-zoom landmark wrapper | `adapters/inference/tracking_zoom_landmarker.py` | `tests/adapters/test_tracking_zoom_landmarker.py` | FR-026, FR-008, §8.1.1, AC-ZOOM-01 |
| Brightness / background compositing | `domain/frame_adjustments.py` | `tests/domain/test_frame_adjustments.py` | FR-027, FR-028, AC-DISPLAY-01 |
| 明るさ・追跡ズームの自動調整heuristic | `domain/auto_tuning.py` | `tests/domain/test_auto_tuning.py` | FR-026, FR-028, FR-029, AC-AUTOTUNE-01 |
| MediaPipe segmentation adapter | `adapters/inference/mediapipe_segmenter.py` | `tests/adapters/test_mediapipe_segmenter.py` | FR-027, AC-DISPLAY-01 |
| OpenCV age / gender inference | `adapters/inference/opencv_demographic_estimator.py` | `tests/adapters/test_opencv_demographic_estimator.py` | FR-022〜024 |
| Metric bundle loading | `adapters/io/metric_config.py` | `tests/adapters/test_metric_config.py` | FR-011, FR-020 |
| 詳細結果の描画と §5.3 の文言 | `ui/scores.py` | `tests/ui/test_scores.py` | FR-013, FR-014, §5.2, §5.3, §8.6.3 |
| 表示値の解決と整形（共有） | `ui/readings.py` | `tests/ui/test_readings.py` | FR-013, FR-036, §8.6.3, §8.6.7, §14.3e |
| Camera / Image 共有の対称化パネル | `ui/symmetrization_panel.py` | `tests/ui/test_symmetrization_panel.py` | FR-034, FR-035, §5.3, §10.7 |
| Image 入力モード一式 | `ui/image_section.py` | `tests/ui/test_image_section.py` | FR-004, FR-006, FR-024, §5.1, §10.8 |
| ダウンロードパネル（CSV / JSON / レポート、Camera・Image共用） | `ui/export_panel.py` | `tests/ui/test_export_panel.py` | FR-017, FR-020, FR-042 |
| Still-image landmarks | `application/still_image.py` | `tests/application/test_still_image.py` | FR-006, FR-013 |
| 対称化の一発処理 | `application/symmetrization.py` | `tests/application/test_symmetrization.py` | FR-034, FR-035, AC-SYMM-01, AC-SYMM-02, §14.3g |
| Model manifest / install | `adapters/io/model_store.py` | `tests/adapters/test_model_store.py` | FR-002, AC-MODEL-01 |
| Real-model smoke | `scripts/bootstrap_models.py` | `tests/model/test_real_landmarker.py` | FR-002, §21.1 |
| OpenVINO reference parity | `scripts/verify_openvino_parity.py` | `tests/model/test_openvino_parity.py` | OD-07, FR-022〜024, §21.1 |
| Cross-ONNX-runtime agreement | `scripts/verify_onnx_runtime_agreement.py` | `tests/model/test_onnx_runtime_agreement.py` | OD-07, FR-022〜024, §21.1 |
| Camera composition（対称化キャプチャUIを含む） | `ui/camera.py` | `tests/ui/test_camera.py` | FR-004, FR-005, FR-034, NFR-009 |
| 映像へ重ねる HUD の組み立て | `ui/hud.py` | `tests/ui/test_hud.py` | FR-013, FR-032, §5.2, §14.3e |
| HUD のブラウザ資産（CSS・実行時 JS） | `ui/hud_assets.py` | `tests/ui/test_hud_assets.py` | FR-013, FR-032, §5.2, §5.4, §14.3e |
| 位置誘導メッセージ | `ui/guidance.py` | `tests/ui/test_guidance.py` | FR-031, AC-GUIDE-01 |
| Model readiness UI | `ui/model_setup.py` | `tests/ui/test_model_setup.py` | FR-001, FR-002, FR-020 |
| Analyze ページの骨格（同意・入力選択） | `ui/app.py` | `tests/ui/test_app.py` | FR-001, FR-004, AC-FLOW-01, AC-A11Y-01 |
| About page | `ui/about_page.py` | `tests/ui/test_about_page.py` | §4.1, FR-014, §10.7 |
| 参照統計セルの絞り込みと合併（Chan の並列分散） | `domain/reference_stats.py` | `tests/domain/test_reference_stats.py` | FR-037, §9.5, AC-REFERENCE-01, AC-REFERENCE-02 |
| 参照統計の use case・入力検証・失敗分離 | `application/reference_browser.py` | `tests/application/test_reference_browser.py` | FR-037, AC-REFERENCE-02, AC-REFERENCE-03 |
| 凍結参照アセットの読み込みと sha256 検証 | `adapters/io/reference_dataset.py` | `tests/adapters/test_reference_dataset.py` | FR-037, §9.5, §10.9, AC-REFERENCE-03 |
| 参照統計ページの描画と注意書き | `ui/reference_page.py` | `tests/ui/test_reference_page.py` | FR-037, §10.7, AC-REFERENCE-02〜04 |
| 参照平均の併記（母集団の選択と差） | `domain/reference_baseline.py` | `tests/domain/test_reference_baseline.py` | FR-039, §10.12, AC-BASELINE-01, AC-BASELINE-02 |
| 参照平均の use case（失敗分離・memo 化・セッション単位） | `application/reference_baseline.py` | `tests/application/test_reference_baseline.py` | FR-039, §9.6.1, §10.12, AC-BASELINE-01, AC-BASELINE-03, AC-BASELINE-05 |
| 表示できた判定の観測 | `application/demographic_observations.py` | `tests/application/test_demographic_observations.py` | FR-039, FR-042, §10.12, AC-BASELINE-05 |
| 参照平均の画面側の解決 | `ui/baseline.py` | `tests/ui/test_baseline.py` | FR-039, §10.12, AC-BASELINE-01 |
| 個票 → セルテーブルの生成と自己検証 (dev-only) | `scripts/build_reference_dataset.py` | `tests/repository/test_build_reference_dataset.py` | FR-037, §9.5, §10.9 |

## Maintenance

- Curated tables above change when responsibilities change.
- The tree below is generated from tracked and non-ignored paths.
- Do not edit between the generated markers.
- Regenerate: `uv run python scripts/generate_structure.py --include-untracked`
- Verify: `uv run python scripts/generate_structure.py --include-untracked --check`

<!-- BEGIN GENERATED TREE -->
_File-list SHA-256: `39aa549841fcf6d611170312a5988adcd84782b4dcbbf1933a7f1f23fa3692a4`_

face-symmetry-lab/
- `.antigravity/`
  - `instructions.md`
- `.claude/`
  - `agents/`
    - `implementer.md`
  - `scripts/`
    - `block-commit.sh`
  - `settings.json`
- `.github/`
  - `workflows/`
    - `ci.yml`
  - `copilot-instructions.md`
- `.streamlit/`
  - `config.toml`
- `configs/`
  - `metrics/`
    - `symmetry-2d-v1.json`
    - `symmetry-3d-v1.json`
  - `reference/`
    - `utkface_symmetry3d_cells_v1.csv`
    - `utkface_symmetry3d_cells_v1.meta.json`
  - `age-gate-v1.json`
  - `model-manifest.json`
- `data/`
  - `README.md`
- `docs/`
  - `adr/`
    - `0001-age-estimation-runtime.md`
    - `0002-head-pose-coordinate-signs.md`
    - `0003-age-gender-and-skin-correction.md`
    - `0004-reference-statistics-browser-data.md`
    - `0005-age-gate-for-reference-age-display.md`
    - `0006-reference-baseline-alongside-user-readings.md`
    - `0007-user-adjustable-display-smoothing.md`
    - `0008-display-hold-for-live-readings.md`
    - `0009-report-export-with-embedded-images.md`
  - `model-cards/`
    - `age-gate-v1-evaluation.json`
    - `age-gate-v1.md`
    - `mediapipe-face-landmarker-v1.md`
    - `mediapipe-selfie-segmenter.md`
    - `openvino-age-gender-retail-0013.md`
    - `symmetry-2d-v1.md`
    - `symmetry-3d-v1.md`
  - `HANDOFF.md`
  - `beauty-page-plan.md`
  - `capture-protocol.md`
  - `implementation-prompts.md`
  - `symmetry-metric-analysis.md`
  - `symmetry-metric-level3.md`
  - `symmetry-metric-measurements.md`
- `research/`
  - `utkface-symmetry/`
    - `utkface_age_gate_lab.ipynb`
    - `utkface_symmetry_lab.ipynb`
- `scripts/`
  - `__init__.py`
  - `bootstrap_models.py`
  - `build_reference_dataset.py`
  - `convert_demographic_model.py`
  - `derive_landmark_map.py`
  - `generate_structure.py`
  - `measure_metric_noise.py`
  - `measure_pose_response.py`
  - `measure_session_variation.py`
  - `measure_zoom_depth_scale.py`
  - `symmetry_level3_probe.py`
  - `symmetry_metric_probe.py`
  - `verify.py`
  - `verify_onnx_runtime_agreement.py`
  - `verify_openvino_parity.py`
- `src/`
  - `face_symmetry_lab/`
    - `adapters/`
      - `export/`
        - … (1 files)
      - `imaging/`
        - … (3 files)
      - `inference/`
        - … (7 files)
      - `io/`
        - … (4 files)
      - … (1 files)
    - `application/`
      - … (19 files)
    - `domain/`
      - … (25 files)
    - `runtime/`
      - … (3 files)
    - `ui/`
      - … (21 files)
    - `__init__.py`
    - `__main__.py`
- `tests/`
  - `adapters/`
    - `test_image_encoding.py`
    - `test_mediapipe_landmarker.py`
    - `test_mediapipe_segmenter.py`
    - `test_metric_config.py`
    - `test_model_store.py`
    - `test_onnx_age_gate.py`
    - `test_opencv_demographic_estimator.py`
    - `test_piecewise_warp.py`
    - `test_reference_dataset.py`
    - `test_runtime_errors.py`
    - `test_tracking_zoom_landmarker.py`
  - `application/`
    - `__init__.py`
    - `test_age_gate.py`
    - `test_age_gate_wiring.py`
    - `test_analysis_cadence.py`
    - `test_analyzer.py`
    - `test_auto_tune_controller.py`
    - `test_demographic_observations.py`
    - `test_demographic_relay.py`
    - `test_demographic_worker.py`
    - `test_frame_capture.py`
    - `test_image_input.py`
    - `test_preview_compositor.py`
    - `test_quality.py`
    - `test_realtime.py`
    - `test_reference_baseline.py`
    - `test_reference_browser.py`
    - `test_session.py`
    - `test_still_image.py`
    - `test_symmetrization.py`
  - `domain/`
    - `__init__.py`
    - `test_age_gate.py`
    - `test_auto_tuning.py`
    - `test_demographics.py`
    - `test_display_hold.py`
    - `test_export.py`
    - `test_frame_adjustments.py`
    - `test_geometry.py`
    - `test_landmarks.py`
    - `test_metrics.py`
    - `test_models.py`
    - `test_overlay.py`
    - `test_pose.py`
    - `test_reference_baseline.py`
    - `test_reference_stats.py`
    - `test_score_explanation.py`
    - `test_score_scale.py`
    - `test_session_summary.py`
    - `test_skin_quality.py`
    - `test_smoothing.py`
    - `test_symmetrization.py`
    - `test_symmetry.py`
    - `test_symmetry_3d.py`
    - `test_tracking_zoom.py`
    - `test_warp_mesh.py`
  - `fixtures/`
    - `age_gate_contract_fixture.json`
  - `model/`
    - `conftest.py`
    - `test_age_gate_contract.py`
    - `test_demographic_conversion.py`
    - `test_onnx_runtime_agreement.py`
    - `test_openvino_parity.py`
    - `test_real_demographic.py`
    - `test_real_landmarker.py`
    - `test_real_segmenter.py`
  - `repository/`
    - `test_build_reference_dataset.py`
    - `test_generate_structure.py`
    - `test_testbox_documentation.py`
  - `runtime/`
    - `test_display_cache.py`
    - `test_frame_scheduler.py`
  - `ui/`
    - `__init__.py`
    - `conftest.py`
    - `test_about_page.py`
    - `test_age_gate_notice.py`
    - `test_app.py`
    - `test_baseline.py`
    - `test_camera.py`
    - `test_export_panel.py`
    - `test_guidance.py`
    - `test_hud.py`
    - `test_hud_assets.py`
    - `test_image_section.py`
    - `test_model_setup.py`
    - `test_readings.py`
    - `test_reference_page.py`
    - `test_report.py`
    - `test_report_figures.py`
    - `test_scores.py`
    - `test_smoothing_panel.py`
    - `test_still_overlay.py`
    - `test_summary_panel.py`
    - `test_symmetrization_panel.py`
  - `__init__.py`
  - `test_cli.py`
- `.gitignore`
- `.python-version`
- `CLAUDE.md`
- `CODEX.md`
- `NOTICE`
- `README.md`
- `STRUCTURE.md`
- `pyproject.toml`
- `requirement.md`
- `testbox.md`
- `uv.lock`
<!-- END GENERATED TREE -->
