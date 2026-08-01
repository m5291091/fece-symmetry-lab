# Face Symmetry Lab

Face Symmetry Lab は、顔の左右差とモデル推定の参考年齢帯をローカルで可視化する研究用 Streamlit アプリです。画像をアプリから外部へ送信・保存せず、人物の価値、美しさ、医学的状態、本人性を判定しません。

仕様の正本は [requirement.md](requirement.md)、リポジトリ探索の入口は [STRUCTURE.md](STRUCTURE.md)、AI・開発作業の共通規約は [AGENTS.md](AGENTS.md) です。

## 現在の実装範囲

- Web カメラのリアルタイム顔ランドマーク表示（MediaPipe Face Landmarker 478 点）
- アップロードした JPEG / PNG のランドマーク検出と重ね描き
- JPEG / PNG のサイズ・形式・アニメーション検証
- 単一顔ポリシー（0 顔・2 顔以上では点を描かない）
- 固定 manifest と SHA-256 検証によるモデル準備
- UTKFace 8 年代分類 MobileNetV2 の ONNX Runtime 推論境界
- 0–9、10–19、20–29、30–39、40–49、50–59、60–69、70+ の参考年齢帯表示
- モデル版・SHA-256・8 クラス確率の表示
- モデル未配置・不正出力・推論失敗をアプリ全体から分離

左右差スコアの算出は `requirement.md` の M1 / M2 で実装します。現時点でランドマークは表示のみで、対称性の数値は出しません。

## セットアップ

```bash
uv sync --frozen
uv run python scripts/bootstrap_models.py   # manifest を表示して同意後に取得
uv run face-symmetry-lab
```

Streamlit は `.streamlit/config.toml` により `127.0.0.1` だけで待ち受け、利用統計を送信しません。カメラは STUN / TURN を使わず、音声 permission を要求しません。

## ランドマークモデル

`configs/model-manifest.json` に取得元 URL、サイズ、ライセンス、SHA-256 を固定しています。アプリは起動時にモデルを自動取得しません。

```bash
uv run python scripts/bootstrap_models.py                    # 内容を表示して確認
uv run python scripts/bootstrap_models.py --accept-manifest   # CI / 非対話
uv run python scripts/bootstrap_models.py --verify            # 配置済みファイルを検証
```

取得は一時ファイルへ書き、SHA-256 が一致した場合だけ `models/` へ atomic rename します。1 byte でも異なるモデルは配置しません。配置先を変える場合は `FACE_SYMMETRY_LAB_LANDMARKER_PATH` を設定します。

### 実環境での検出テスト

モデル配置後、実 asset を使う smoke テストが有効になります（未配置なら skip）。

```bash
uv run pytest -m model
```

合成した顔画像で 478 点検出、複数顔の拒否、VIDEO / IMAGE 両モードの動作を検証します。実写で確認する場合は、本人または同意のある写真をローカルに置き、`FACE_SYMMETRY_LAB_SMOKE_IMAGE` にパスを指定してください。実写ファイルはリポジトリへコミットしません。

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
