# Test Index

このファイルは production code、利用者フロー、要件 ID と test の対応を示す。test を追加・移動・削除した場合は同じ変更で更新する。

| Test | Production / flow | Requirements |
|---|---|---|
| `tests/domain/test_models.py` | `domain.models` の分析状態・左右差 result | FR-008, FR-011, NFR-013 |
| `tests/domain/test_age.py` | logits→8 年齢帯、確率・時刻・model metadata 検証 | FR-022, FR-023, AC-AGE-01 |
| `tests/application/test_image_input.py` | JPEG / PNG upload の形式・容量・画素・animation 検証 | FR-006, FR-007, AC-INPUT-01 |
| `tests/adapters/test_onnx_age_estimator.py` | ONNX 前処理、8 logits contract、失敗分離 | FR-022〜024, AC-AGE-02 |
| `tests/ui/test_app.py` | 年齢モデル設定と UI composition | FR-001, FR-022, AC-FLOW-01 |
| `tests/test_cli.py` | localhost 限定の Streamlit 起動 command | NFR-009, AC-SEC-01 |
| `tests/repository/test_generate_structure.py` | 安全で bounded な `STRUCTURE.md` 生成 | 開発効率・リポジトリ規約 |
| `tests/repository/test_testbox_documentation.py` | test file set と本索引の同期 | 開発効率・リポジトリ規約 |
