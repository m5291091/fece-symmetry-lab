# Face Symmetry Lab

Face Symmetry Lab は、顔の左右差とモデル推定の参考年齢帯をローカルで可視化する研究用 Streamlit アプリです。画像をアプリから外部へ送信・保存せず、人物の価値、美しさ、医学的状態、本人性を判定しません。

仕様の正本は [requirement.md](requirement.md)、リポジトリ探索の入口は [STRUCTURE.md](STRUCTURE.md)、AI・開発作業の共通規約は [AGENTS.md](AGENTS.md) です。

## 現在の実装範囲

- JPEG / PNG のサイズ・形式・アニメーション検証
- UTKFace 8 年代分類 MobileNetV2 の ONNX Runtime 推論境界
- 0–9、10–19、20–29、30–39、40–49、50–59、60–69、70+ の参考年齢帯表示
- モデル版・SHA-256・8 クラス確率の表示
- モデル未配置・不正出力・推論失敗をアプリ全体から分離

左右差の静止画分析とカメラ分析は、`requirement.md` の M1 / M2 で順次実装します。

## セットアップ

```bash
uv sync --frozen
uv run face-symmetry-lab
```

Streamlit は `.streamlit/config.toml` により `127.0.0.1` だけで待ち受け、利用統計を送信しません。

## 年齢帯モデル

アプリは次のいずれかで ONNX モデルを読み込みます。

1. `models/age_mobilenetv2_utkface_v1.onnx`
2. 環境変数 `FACE_SYMMETRY_LAB_AGE_MODEL_PATH` のローカルパス

モデルは Git に含めません。創造工房セミナーの `best_age_model_mobilenetv2.pth` を別の学習環境で変換します。

```bash
python scripts/export_age_model.py /path/to/best_age_model_mobilenetv2.pth
```

変換環境だけに PyTorch / torchvision が必要です。アプリ本体は ONNX Runtime を使うため、学習依存や InsightFace を runtime lock へ混在させません。評価値と制約は [モデルカード](docs/model-cards/age-mobilenetv2-utkface-v1.md) を確認してください。

## 開発

変更は Red → Green → Refactor → Verify の順で進めます。

```bash
uv run python scripts/verify.py
```

個別コマンド:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
uv run python scripts/generate_structure.py --include-untracked --check
```

新しいテストを追加・移動した場合は [testbox.md](testbox.md) も更新してください。
