# Face Symmetry Lab

Face Symmetry Lab は、Web カメラの**リアルタイム映像**から顔の左右差（対称性）を測る、ローカル完結の研究用 Streamlit アプリです。あわせて 18〜75 歳向けの参考年齢・参考性別を、左右差とは独立した参考値として推定します。画像・映像・派生値を外部へ送信・保存せず、人物の価値・美しさ・医学的状態・本人性を判定しません。

仕様の正本は [requirement.md](requirement.md)、リポジトリ探索の入口は [STRUCTURE.md](STRUCTURE.md)、AI・開発作業の共通規約は [CLAUDE.md](CLAUDE.md) です。本 README は概要であり、数値・受け入れ基準の正本は `requirement.md`、実測の正本は [`docs/`](docs/) と[モデルカード](docs/model-cards/)です。

## 課題

Web カメラの**リアルタイム映像**から「顔の形の左右差」を、撮影条件（頭の向き・照明）から切り離して数値化し、同一人物・同一環境内の相対的な変化として観察できるようにする。リアルタイムでは頭が絶えず動くため、姿勢と形を分離することが特に重要になる。次の 3 条件を同時に満たす必要がある。

- **プライバシー**: 画像・映像・ランドマーク・派生値を端末内に閉じ、外部送信・暗黙のファイル保存・顔埋め込みを行わない。
- **姿勢に汚染されない指標**: yaw / pitch / roll によらず「顔の形の左右差」だけを測る。
- **恣意性の排除**: 較正されていない自由パラメータや、順位・偏差値・合否・美的評価を持ち込まない。

医療・本人確認・採用・保険・教育・監視・人物比較の用途には用いない（`requirement.md` §10）。

## 既存手法の限界

一般的な顔解析プロダクト・サービスは、本アプリが満たそうとする 3 条件のいずれかを損なう。

- **クラウド型の顔解析 API・アプリ**: 画像・映像を外部サーバへ送信して処理するため、保存先・二次利用を利用者が制御できない。端末内完結・オフラインの要求と両立しない。
- **「美人度」「対称性スコア」系アプリ**: 較正されていない重み付けで 0〜100 の断定的なスコアや順位・偏差値を出し、魅力・優劣を示唆する。母集団や撮影条件を跨いだ比較を許してしまう。
- **顔認識・埋め込みベースの手法**: 顔埋め込みを生成・保存し、人物同定や人物間比較へ転用されうる。
- **単眼撮影での姿勢・照明への感受性**: 多くの手法は頭部の向き（yaw / pitch / roll）や照明で値が動き、「顔の形の左右差」と撮影条件を分離しない。頭が動き続けるリアルタイム映像では、この汚染がとりわけ表面化する。

## 提案手法

**リアルタイム映像に対する 3D object symmetry 解析（`symmetry-3d/1.0.0`）** を主指標とする（`requirement.md` §8.5.1、[指標カード](docs/model-cards/symmetry-3d-v1.md)）。各フレームで、対応ペアを鏡映してペア名を入れ替えた配置を、元の配置へ**並進と回転だけ**で Procrustes 重ね合わせ、残差 RMS を距離 `d_3d` とする。

- **姿勢不変**: 剛体重ね合わせが並進・回転（yaw / pitch / roll のすべて）を吸収するため、頭が動くリアルタイム映像でも「形の左右差」だけが残る。正規化で値へ効くのはスケールだけになる。
- **恣意性なし**: 較正されていない自由パラメータを持たず、各対応点は等しく 1 票。
- **ローカル完結**: 推論は MediaPipe と OpenCV DNN / ONNX Runtime のローカル資産に対して行い、`127.0.0.1` だけで待ち受け、利用統計・STUN / TURN を既定無効にする。顔埋め込みを生成せず、モデルは版と SHA-256 で固定する。
- **低遅延リアルタイム**: 重いランドマーク推論を最大 8Hz へ間引き、間引いたフレームは直前の幾何を 250ms だけ再利用する。数値は映像の上に重ねた DOM レイヤーの HUD が表示し、映像 pixel へは焼き込まない。
- **失敗分離**: 年齢・性別・背景などある推論の失敗が、左右差分析・表示を止めない。

## 顔対称性スコアの定義

定義の正本は `requirement.md` §8.5.1 / §8.6.5 / §8.6.7、詳細は[指標カード](docs/model-cards/symmetry-3d-v1.md)。

- **全体の対称度 `d_3d`**: 対応ペア 63 組（126 点）の 3D 座標を 1 回だけ Procrustes 重ね合わせた残差 RMS。**3 次元で測った**両眼アンカー間距離を 1.0 とする無次元量。0 が下限、上限なし。小さいほど鏡映配置がよく重なる。

  ```text
  X    = 対応ペア 126 点の 3D 座標
  M(X) = σ( mirror_x(X) )              σ はペア表そのものの置換
  d_3d = sqrt( mean_i || X_i − (M(X)·R + t)_i ||² )   R は回転、スケール自由度なし
  ```

- **部位別の対称度 `d_3d,p`**: 同じ 1 回の重ね合わせの残差を輪郭・目・鼻・口ごとに読む**並列の診断値**。総合値へは寄与しない。全部位が有効なとき恒等式 `d_3d² = Σ_p m_p·d_3d,p² / Σ_p m_p` が成立する（`m_p` は版管理された点数で、設定値ではない）。
- **全体のスコア `s_3d`**: 部位別 `d_3d,p` を**部位ごとに 1 票**として組み立てた別の総合量（`d_3d` は対応点ごとに 1 票）。重みは全部位 1 に固定で調整できない。`d_3d` とは目盛りも異なるため、両者の数字の大小を直接比べない。
- **画面表示**: 距離を **対称度 = 100% − ずれ%** として出す（係数のない可逆変換）。「全体のスコア」だけは §8.6.7 の表示スケールで 0〜100 点へ換算する: `score_points = 25 × (対称度% − 95)`、アンカーは **ずれ 1.0% = 100 点 / ずれ 5.0% = 0 点**（clamp）。実測が 98% 前後の狭い幅へ潰れるのを避けるための、線形・可逆な目盛りの読み替えである。
- **解釈の制約**: 撮影条件で値が動くため、絶対値ではなく**同一人物・同一環境内の相対変化**だけを参照する。母集団内順位・正常範囲・美的品質・人物間の差としては解釈しない。100 点は完全対称の証明ではない。欠測（`z` 欠測・有効ペア不足）は 0 で埋めず欠測のまま扱う。

## 処理フロー・システムアーキテクチャ

Web 的な MVC ではなく **層 + port/adapter** で構成する（`CLAUDE.md` §9）。依存方向は `ui → application → domain`、adapter は application の port を実装する。

| 層 | 役割 |
|---|---|
| `domain` | 純粋計算と immutable DTO（対称性・幾何・スコア定義・DTO） |
| `application` | use case・状態・入力検証・port（リアルタイム pipeline、解析周期、セッション記録、年齢ゲート） |
| `adapters` | MediaPipe / ONNX Runtime / OpenCV / export / IO |
| `ui` | Streamlit の composition と表示（HUD・タブ・レポート・i18n） |
| `runtime` | frame scheduling と表示キャッシュ |

**リアルタイム処理フロー**（`application.realtime.RealtimeLandmarkPipeline`）:

```text
WebRTC frame
  → FrameScheduler（latest-frame-wins、容量 1、解析中に届いたフレームは捨てる）
  → AnalysisCadence（ランドマーク推論を最大 8Hz へ間引く）
  → 明るさ補正（検出器入力のみ）→ Face Landmarker（478 点）
  → FrameAnalyzer（正規化 → 3D Procrustes 重ね合わせ → 品質ゲート → 時間ベース EMA）
  → age-gate（参考年齢の表示可否判定）→ PipelineSnapshot（不変ビュー）
       ├→ 映像上 HUD（5Hz、DOM オーバーレイ。数値と直近 60 秒の折れ線）
       ├→ 詳細タブ（1Hz。計算過程・姿勢・診断）
       └→ 参考年齢・参考性別タブ（1Hz）
  ┌ 独立 worker（別スレッド・独自 1Hz cadence・容量 1）: 年齢・性別推論
  └ 背景置換（表示専用、任意）
```

間引かれたフレームは直前の幾何・背景 mask を 250ms だけ再利用する（`runtime.display_cache`）。年齢・性別推論・背景置換・age-gate はいずれもクリティカルパス外で、失敗しても左右差分析を止めない（失敗分離）。数値は映像 pixel へ焼き込まず DOM レイヤーの HUD が表示する（遅延要件 FR-013）。詳細・参考年齢タブを 1Hz に抑えるのは、5Hz のフル再描画がブラウザのメインスレッドで WebRTC 映像と競合するのを避けるためである（HUD だけ NFR-003 の 4〜5Hz を維持）。

## データセット・モデル

すべて `configs/model-manifest.json` に取得元 URL・サイズ・ライセンス・SHA-256 を固定し、アプリは起動時に自動取得しない。

| 用途 | モデル / データ | 版 | ライセンス |
|---|---|---|---|
| 顔ランドマーク（478 点） | MediaPipe Face Landmarker | `float16-1` | Apache-2.0 |
| 背景置換（表示専用） | MediaPipe Selfie Segmenter | `float16-1` | Apache-2.0 |
| 参考年齢・参考性別（18〜75 歳） | Open Model Zoo `age-gender-recognition-retail-0013`（固定 IR から派生 ONNX を決定的生成） | `2023.0/FP16` → `FP16-onnx-v2` | Apache-2.0 |
| 参考年齢の表示可否ゲート | `age-gate/1.0.0`（同梱の決定木、幾何特徴 41 次元入力） | `1.0.0` | 非商用研究のみ（UTKFace 学習） |
| 参照統計（同年齢・同性別の平均の併記） | UTKFace 事前集計セルテーブル | `utkface_symmetry3d_cells_v1` | 非商用研究のみ |

年齢・性別モデルの推論は `opencv-contrib-python` の DNN と `onnxruntime`（age-gate 専用）だけで CPU 実行し、OpenVINO / PyTorch / InsightFace を runtime lock へ混在させない。各[モデルカード](docs/model-cards/)に公開評価値・入出力契約・制約を記録している。

## 定量的結果

いずれも被写体 1 名・単一環境での測定であり、母集団統計・順位・正常範囲としては解釈できない（[`docs/symmetry-metric-measurements.md`](docs/symmetry-metric-measurements.md)、未取得の証跡は[指標カード](docs/model-cards/symmetry-3d-v1.md)）。

**部位別ゲイン**（既知量の非対称を注入したときの応答。1.0 = 注入分を正しく検出）:

| 部位 | ゲイン |
|---|---:|
| 輪郭 | +0.138 |
| 目 | +0.200 |
| 鼻 | +0.057 |
| 口 | +0.238 |

- **信号 ÷ ノイズ床 = 5.52x**（人と人の差をノイズ床の上で区別できる）。ノイズ床 0.241mm、検出器そのものの再現性 0.300mm。
- **物理的妥当性**: 距離中央値 **1.157mm** は、文献（Prog Orthod 2015、3D スキャン＋鏡像重ね合わせ、n=28/27）の「対称と判定された成人 1.52mm / 非対称 2.85mm」と同じ土俵に着地する。**入力は 3D スキャンではなく単眼推定の `z`** なので、値を直接あてはめて個人を分類してはならない。
- **姿勢の分離**: 生の z 左右差 1.792mm のうち **77%** は剛体回転（＝姿勢）で、重ね合わせ後 0.404mm へ落ちる。恒等式 `d_3d² = xy² + z²` は誤差 0 で成立。
- **セッション間ばらつき**: 撮り直しで「全体のスコア」は実測で全幅 **6.8 点** 動く（絶対値ではなく相対変化を読む根拠）。
- **参考年齢・参考性別**: 上流公開値は年齢平均誤差 6.99 年・性別 accuracy 95.80%。アプリが実行する派生 ONNX は独立した 2 つの実行系（`cv2.dnn` と `onnxruntime`）で年齢差 2.4×10⁻⁵ 年と一致し、決定的に再現できる。

## 主な機能

**リアルタイム（Web カメラ）を主軸とする。**

- リアルタイム顔ランドマーク表示（MediaPipe Face Landmarker 478 点）と、ランドマーク推論を最大 8Hz へ間引き幾何表示を 250ms 保持する低遅延 preview。
- 映像に重ねる DOM レイヤーの HUD: 全体の対称度・全体のスコア・部位別 4 つ・参考年齢・参考性別と、直近 60 秒の折れ線グラフ。品質ゲートの枠色、直すべき主要因 1 件の指示（例「顔を正面に向けてください」）、明るさの助言（暗い・明るすぎる）を映像を見ながら確認できる。HUD は位置選択とドラッグ移動ができる。
- 姿勢推定（yaw / pitch / roll）・顔サイズ・連続安定性による撮影品質ゲートと OK / HOLD ヒステリシス。満たさないフレームでは数値を出さず、欠測を 0 で埋めない。明るさの非ブロッキング警告（`LOW_LIGHT` / `OVEREXPOSED`）は採点・品質ゲートを止めない。
- 全体の対称度は時間ベース EMA で平滑化し、追随の速さ（時定数）と表示の更新間隔を映像直下のポップオーバーから選べる。
- 独立 worker による 18〜75 歳の参考年齢・参考性別推論（別 cadence・容量 1）。参考性別は女性・男性の確率として表示し、対象範囲外へは参考年齢を出さない表示ゲートを通す。
- 同梱 UTKFace 事前集計による参照統計の閲覧と、表示中の参考年齢・参考性別に対する同年齢・同性別の平均・差の併記（人種では絞り込まない）。
- 背景の黒 / 白置換（表示専用・任意）。中心線の ON / OFF。
- セッションの記述統計要約（有効率・中央値・変動幅・主な HOLD 理由）と、セッション全体を 1 つの HTML レポートとして保存（`metric_version` / `config_fingerprint` 付き。画像・landmark 座標は含めない）。
- 日本語 / 英語の UI 表示言語切替。

**画像アップロード（補助）**: JPEG / PNG のサイズ・形式・アニメーション検証、ランドマーク検出・重ね描き、左右対称にそろえた点の可視化と対称化画像の生成（既定 OFF の明示操作、変位が大きい場合は生成せず理由を表示。生成物は加工画像であることを明示し、保存・送信しない）。

**共通の安全性**: 単一顔ポリシー（0 顔・2 顔以上では数値を出さない）、固定 manifest と SHA-256 検証によるモデル準備、モデル未配置・不正出力・推論失敗のアプリ全体からの分離。

左右差は **試験中の指標** です。画面には **対称度**（両眼の間隔を 100% としたときの左右対応点のずれを 100% から引いた値）を表示します。係数のない引き算だけの表示変換で、撮影条件で値が動くため、数値の絶対値ではなく同一人物・同一環境での相対的な変化だけを参照してください。異なる人物の比較、医学的評価、魅力度の判断には使用できません。

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

## 計算手法

各フレームの左右差は、姿勢に依存しない 3D object symmetry 解析（`symmetry-3d/1.0.0`、`requirement.md` §8.5.1 / §8.6.5）で求めます。

1. 478 点のうち版管理された左右対応ペア 63 組（126 点）の 3D 座標を取り出し、両眼アンカー（目頭・目尻の中点）で測った 3 次元距離を 1.0 とするスケール正規化を行う。
2. 対応ペアを鏡映し、ペア表そのものの置換でペア名を入れ替える。
3. 入れ替えた配置を、元の配置へ**並進と回転だけ**で Procrustes 重ね合わせる。残差 RMS が全体の対称度 `d_3d`。部位別 `d_3d,p` は同じ 1 回の重ね合わせの残差を部位ごとに読み、`s_3d`（全体のスコアの元）は部位を 1 票ずつ数えた総合量である。
4. 画面には距離を **対称度**（`100% − ずれ%`）として出す。全体の対称度だけを時間ベース EMA（`alpha = 1 - exp(-Δt/tau)`）で平滑化し、`tau` は映像直下のポップオーバーから選べる（既定 0.4 秒）。平滑化は表示の追随の速さだけを変え、生値・計算式を変えない。`z` 欠測や有効ペア不足のフレームは、値を欠測のまま扱い 0 で埋めない。

左右対応マップは手書きせず、Face Landmarker のメッシュ構造から導出して凍結します。

```bash
uv run python scripts/derive_landmark_map.py --check   # 凍結済み設定と一致するか
uv run python scripts/derive_landmark_map.py --write   # 再導出して凍結し直す
```

## 年齢・性別モデル

アプリはOpen Model Zoo `age-gender-recognition-retail-0013`の公式OpenVINO IR（XML/BIN）から固定生成したONNXを読み込みます。

```bash
uv run python scripts/bootstrap_models.py --accept-manifest
uv run --script scripts/convert_demographic_model.py
```

生成先は`models/age-gender-recognition-retail-0013.onnx`です。別の場所を使う場合は環境変数`FACE_SYMMETRY_LAB_DEMOGRAPHIC_ONNX_PATH`を指定します。

モデルはGitに含めません。Open Model Zoo 2023.0 FP16の固定URL、size、Apache-2.0、XML/BIN個別SHA-256に加え、派生ONNXの変換recipe、size、SHA-256をmanifestへ登録済みです。公式assetの取得とONNX生成はいずれも明示操作だけです。

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
