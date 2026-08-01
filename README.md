# Face Symmetry Lab

Face Symmetry Lab は、顔の左右差、撮影条件依存の肌表面参考値、18〜75歳向けの参考年齢・参考性別をローカルで可視化する研究用Streamlitアプリです。画像をアプリから外部へ送信・保存せず、人物の価値、美しさ、医学的状態、本人性を判定しません。

仕様の正本は [requirement.md](requirement.md)、リポジトリ探索の入口は [STRUCTURE.md](STRUCTURE.md)、AI・開発作業の共通規約は [AGENTS.md](AGENTS.md) です。

## 現在の実装範囲

- Web カメラのリアルタイム顔ランドマーク表示（MediaPipe Face Landmarker 478 点）
- アップロードした JPEG / PNG のランドマーク検出と重ね描き
- JPEG / PNG のサイズ・形式・アニメーション検証
- 単一顔ポリシー（0 顔・2 顔以上では点を描かない）
- 姿勢推定（yaw / pitch / roll）と撮影品質ゲート、OK / HOLD のヒステリシス
- 輪郭・目・鼻・口・全体の左右差スコア（実験値）と時間ベース EMA
- リアルタイム映像内の高コントラストな左右差・姿勢数値オーバーレイ
- ランドマーク推論を最大8Hzへ間引き、幾何表示を250msだけ保持する低遅延preview
- 太いmagenta中心線と、Camera / Image個別のON/OFF設定
- 顔領域のbounded white balance・明度補正と、頬の肌表面参考値（未校正）
- `metric_version` / `config_fingerprint` / 正規化 RMS 距離付き CSV・JSON エクスポート
- 版管理された左右対応マップと指標バンドル（`configs/metrics/`）
- 固定 manifest と SHA-256 検証によるモデル準備
- Open Model Zooの18〜75歳向け連続年齢・性別モデルのOpenCV DNN推論境界
- モデル版・組み合わせSHA-256・性別2クラス確率の表示
- モデル未配置・不正出力・推論失敗をアプリ全体から分離

左右差スコアは **試験中の指標** です。感度係数 `k_p` を固定評価セットで凍結していない（OD-01）ため、数値の絶対値ではなく、同一人物・同一環境での相対的な変化だけを参照してください。異なる人物の比較、医学的評価、魅力度の判断には使用できません。

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

## 左右差スコアの算出

`requirement.md` §8 の手順に従います。

1. 眼の目頭・目尻の中点を anchor とし、roll 補正・眼間距離 1.0 への正規化・原点移動を行う
2. 正中点集合から **Total Least Squares（PCA）** で中心線を推定する
3. 版管理された左右対応ペアを中心線で鏡映し、部位ごとの正規化 RMS `d_p` を求める
4. `score_p = clamp(100 × exp(-k_p × d_p), 0, 100)`、`overall = Σ(w_p × score_p)`
5. 表示用の全体値のみ時間ベース EMA（`alpha = 1 - exp(-Δt/tau)`）で平滑化する

左右対応マップは手書きせず、Face Landmarker のメッシュ構造から導出して凍結します。

```bash
uv run python scripts/derive_landmark_map.py --check   # 凍結済み設定と一致するか
uv run python scripts/derive_landmark_map.py --write   # 再導出して凍結し直す
```

品質ゲート（姿勢・顔サイズ・連続安定性）を通らないフレームではスコアを出さず、
`HOLD` の理由を行動可能な文言で表示します。欠測は 0 点に置き換えません。

カメラ分析中は、画面左上に状態、全体・4部位の左右差、yaw / pitch / roll を表示します。
同じ数値は画面下のテキスト領域でも確認できます。1件以上の分析結果があると、
セッションデータを CSV または JSON で保存できます。画像やlandmark座標は含めず、
校正に必要な部位別正規化 RMS、`metric_version`、`config_fingerprint` を含めます。

## OD-01 / OD-04 の検証状態

OD-01 の固定評価セットと、OD-04 の同意済み実写単軸映像はリポジトリに含めません。
現時点では実データが未提供のため、`sensitivity_frozen` は `false` のままです。
収集条件と完了判定は
[指標カード](docs/model-cards/symmetry-2d-v1.md) と
[姿勢ADR](docs/adr/0002-head-pose-coordinate-signs.md) に記録しています。
実データなしに係数や符号確認結果を完了扱いにはしません。

## 年齢・性別モデル

アプリはOpen Model Zoo `age-gender-recognition-retail-0013`のOpenVINO IR（XML/BIN）を読み込みます。

1. `models/age-gender-recognition-retail-0013.xml` と同名の`.bin`
2. 環境変数`FACE_SYMMETRY_LAB_DEMOGRAPHIC_XML_PATH` / `FACE_SYMMETRY_LAB_DEMOGRAPHIC_BIN_PATH`

モデルはGitに含めません。固定URL、size、XML/BIN個別SHA-256、license、runtime parityが未確認のためmanifestへも未登録です。承認前に自動取得しません。

アプリ本体は既存の`opencv-contrib-python`だけでCPU推論し、OpenVINO、PyTorch、torchvision、InsightFaceをruntime lockへ混在させません。公開評価値と制約は[モデルカード](docs/model-cards/openvino-age-gender-retail-0013.md)を確認してください。

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
