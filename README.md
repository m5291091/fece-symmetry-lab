# Face Symmetry Lab

Face Symmetry Lab は、顔の左右差、撮影条件依存の肌表面参考値、18〜75歳向けの参考年齢・参考性別をローカルで可視化する研究用Streamlitアプリです。画像をアプリから外部へ送信・保存せず、人物の価値、美しさ、医学的状態、本人性を判定しません。

仕様の正本は [requirement.md](requirement.md)、リポジトリ探索の入口は [STRUCTURE.md](STRUCTURE.md)、AI・開発作業の共通規約は [CLAUDE.md](CLAUDE.md) です。

## 現在の実装範囲

- Web カメラのリアルタイム顔ランドマーク表示（MediaPipe Face Landmarker 478 点）
- アップロードした JPEG / PNG のランドマーク検出と重ね描き
- JPEG / PNG のサイズ・形式・アニメーション検証
- 単一顔ポリシー（0 顔・2 顔以上では点を描かない）
- 姿勢推定（yaw / pitch / roll）と撮影品質ゲート、OK / HOLD のヒステリシス
- 輪郭・目・鼻・口の部位別左右差距離と、全ペアをプールした全体距離（実験値）、全体距離への時間ベース EMA
- 左右対称にそろえた位置と実測点を重ねる対称化ビジュアライゼーション（Camera はワンショットのキャプチャ、Image は表示中の写真から）
- Delaunay 三角形分割＋区分アフィン変換による対称化画像の生成（既定 OFF の明示操作。変位が大きい場合は生成せず理由を表示）
- 映像を最上部に固定し、設定・詳細・対称化・エクスポートをタブへ集約したリアルタイム画面
- ランドマーク推論を最大8Hzへ間引き、幾何表示を250msだけ保持する低遅延preview
- 太いmagenta中心線と、Camera / Image個別のON/OFF設定
- 顔領域のbounded white balance・明度補正と、頬の肌表面参考値（未校正）
- `metric_version` / `config_fingerprint` / 正規化 RMS 距離付き CSV・JSON エクスポート
- 版管理された左右対応マップと指標バンドル（`configs/metrics/`）
- 固定 manifest と SHA-256 検証によるモデル準備
- Open Model Zooの18〜75歳向け連続年齢・性別モデルのOpenCV DNN推論境界
- モデル版・組み合わせSHA-256・性別2クラス確率の表示
- モデル未配置・不正出力・推論失敗をアプリ全体から分離

左右差は **試験中の指標** です。画面には **対称度**（両眼の間隔を 100% としたときの左右対応点のずれを 100% から引いた値）を表示します。係数のない引き算だけの表示変換で、指標本体は正規化 RMS 距離そのもの、エクスポート列と `metric_version` は表示の向きに依存しません。主指標は姿勢に依存しない 3D object symmetry 解析（`symmetry-3d/1.0.0`）で、全対応点をまとめて 1 回で求めた全体の対称度、部位別の対称度、部位を 1 票ずつ数えた全体のスコアを独立に表示します。中心線ベースの旧 2D 指標（`symmetry-2d/2.0.0`）は画面から外れ、エクスポートに診断列として残ります。撮影条件で値が動くため、数値の絶対値ではなく、同一人物・同一環境での相対的な変化だけを参照してください。異なる人物の比較、医学的評価、魅力度の判断には使用できません。

## セットアップ

```bash
uv sync --frozen
uv run python scripts/bootstrap_models.py   # manifest を表示して同意後に取得
uv run face-symmetry-lab
```

Streamlit は `.streamlit/config.toml` により `127.0.0.1` だけで待ち受け、利用統計を送信しません。カメラは STUN / TURN を使わず、音声 permission を要求しません。

## モデル準備

`configs/model-manifest.json` にランドマークモデルと年齢・性別モデルの取得元 URL、サイズ、ライセンス、SHA-256 を固定しています。アプリは起動時にモデルを自動取得しません。

```bash
uv run python scripts/bootstrap_models.py                    # 内容を表示して確認
uv run python scripts/bootstrap_models.py --accept-manifest   # CI / 非対話
uv run python scripts/bootstrap_models.py --verify            # 配置済みファイルを検証
```

取得は一時ファイルへ書き、SHA-256 が一致した場合だけ `models/` へ atomic rename します。1 byte でも異なるモデルは配置しません。配置先を変える場合はランドマークに`FACE_SYMMETRY_LAB_LANDMARKER_PATH`、年齢・性別の派生モデルに`FACE_SYMMETRY_LAB_DEMOGRAPHIC_ONNX_PATH`を設定します。別パスもmanifestと同じsize・SHA-256でなければ読み込みません。

### 実環境での検出テスト

モデル配置後、実 asset を使う smoke テストが有効になります（未配置なら skip）。

```bash
uv run pytest -m model
```

合成した顔画像で 478 点検出、複数顔の拒否、VIDEO / IMAGE 両モードの動作を検証します。実写で確認する場合は、本人または同意のある写真をローカルに置き、`FACE_SYMMETRY_LAB_SMOKE_IMAGE` にパスを指定してください。実写ファイルはリポジトリへコミットしません。

## 対称度の算出

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

品質ゲート（姿勢・顔サイズ・連続安定性）を通らないフレームでは数値を出さず、
`HOLD` の理由を行動可能な文言で表示します。欠測は 0 に置き換えません。

カメラ分析中は、映像の隣に状態・全体の対称度・全体のスコア・参考年齢・参考性別を、その下の
タブに部位別・姿勢・肌表面参考値を表示します。「この瞬間を対称化して見る」を押すと、直近の解析成功
フレームを 1 枚だけ取り込み、正中線で折りたたんだ位置と実測点を重ねた静止画を表示
します。取り込んだ映像はブラウザセッションのメモリだけに残り、保存も送信もしません。
リアルタイムの連続分析はこの操作で停止しません。

チェックを入れると、対応点を左右対称の位置へ移動させ、その間の画素を三角形ごとに変形した
**対称化画像**も生成します（720p・478点でおよそ 40ms）。画像の四隅と辺中点を固定するため、
ランドマークが覆わない髪・耳・背景は外周を保ったまま緩やかに追従します。変位が両眼間隔の
25% を超える場合は生成せず理由を表示し、点の可視化はそのまま残ります。生成されるのは
**幾何変形された加工画像**であり、無加工の写真でも、より正しい・望ましい顔でもありません。1件以上の分析結果があると、
セッションデータを CSV または JSON で保存できます。画像やlandmark座標は含めず、
部位別と全体の正規化 RMS 距離、`metric_version`、`config_fingerprint` を含めます。

## OD-04 の検証状態

OD-04 の同意済み実写単軸映像はリポジトリに含めません。実データなしに符号確認結果を
完了扱いにはしません。収集条件と完了判定は
[姿勢ADR](docs/adr/0002-head-pose-coordinate-signs.md) に記録しています。

OD-01（部位別感度係数の較正）は解消しました。`symmetry-2d/2.0.0` で感度係数と部位重みを
廃止し、較正すべき自由パラメータが無くなったためです。距離分布そのものの参考統計は
[指標カード](docs/model-cards/symmetry-2d-v1.md) に任意調査として残しています。

## 年齢・性別モデル

アプリはOpen Model Zoo `age-gender-recognition-retail-0013`の公式OpenVINO IR（XML/BIN）から固定生成したONNXを読み込みます。

```bash
uv run python scripts/bootstrap_models.py --accept-manifest
uv run --script scripts/convert_demographic_model.py
```

生成先は`models/age-gender-recognition-retail-0013.onnx`です。別の場所を使う場合は環境変数`FACE_SYMMETRY_LAB_DEMOGRAPHIC_ONNX_PATH`を指定します。

モデルはGitに含めません。Open Model Zoo 2023.0 FP16の固定URL、size、Apache-2.0、XML/BIN個別SHA-256に加え、派生ONNXの変換recipe、size、SHA-256をmanifestへ登録済みです。公式assetの取得とONNX生成はいずれも明示操作だけです。固定評価の公式IR runtime parity、slice評価、3 OS smokeはリリース前の未完事項です。

アプリ本体は既存の`opencv-contrib-python`だけでCPU推論し、OpenVINO、ONNX Runtime、PyTorch、torchvision、InsightFaceをruntime lockへ混在させません。変換専用packageはPEP 723 scriptへ隔離しています。公開評価値と制約は[モデルカード](docs/model-cards/openvino-age-gender-retail-0013.md)を確認してください。

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

## 開発体制

メインスレッドはOpus、実装はimplementer(Sonnet)が担当する。
