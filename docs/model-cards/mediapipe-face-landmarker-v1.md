# Model card — MediaPipe Face Landmarker

| 項目 | 内容 |
|---|---|
| name | `face_landmarker` |
| version | `mediapipe-face-landmarker/float16-1` |
| created_at (取得日) | 2026-08-01 |
| owner | Google（MediaPipe Authors）。本リポジトリは配布せず、参照のみ |
| asset | `face_landmarker.task`（3,758,596 bytes） |
| sha256 | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| source | `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task` |
| license | Apache-2.0 |
| runtime | MediaPipe Tasks Vision 0.10.35 系 / CPU delegate |

## 意図された用途

- 端末内での顔ランドマーク（478 点）検出と、その可視化。
- `configs/metrics/symmetry-2d-v1.json` に固定した左右対応マップを用いた、
  同一人物・同一環境内での左右差の相対比較。

## 禁止する用途

`requirement.md` §2.2 / §10.7 に従う。医療診断、本人確認、採用、保険、教育、監視、
人物間の比較・順位付け、魅力度や人格の推定には使用しない。顔埋め込みを生成・保存しない。
年齢・性別推定は本landmarkモデルではなく、別モデル・別結果として明示同意後に限り扱う。

## 入力・出力契約

| 項目 | 内容 |
|---|---|
| 入力 | RGB `uint8` 画像（`mp.Image` / `ImageFormat.SRGB`） |
| running mode | realtime は `VIDEO` + 単調増加 timestamp、静止画は `IMAGE`。インスタンスを兼用しない |
| num_faces | 2（2 顔以上を `MULTI_FACE` と判定するためであり、複数人を採点するためではない） |
| 出力 | 正規化済み 478 landmark（x, y, z）と 4×4 facial transformation matrix |
| 姿勢 | 変換行列の 3×3 部を極分解して回転を取り出し、intrinsic Y-X-Z で yaw / pitch / roll を得る |

## 評価

本リポジトリでは検出精度の再評価を行っていない。実行しているのは
`tests/model/test_real_landmarker.py` の smoke 検証のみである。

| 検証 | 結果 |
|---|---|
| asset の sha256 と manifest の一致 | 合格 |
| 空画像で `NO_FACE`（VIDEO / IMAGE 双方） | 合格 |
| 合成顔で 478 点検出、座標が 0〜1 に収まる | 合格 |
| 合成 2 顔画像で `MULTI_FACE`、点を描かない | 合格 |
| 合成対称顔の姿勢 \|yaw\|, \|roll\| < 15° | 合格 |
| 合成対称顔 → 左右差スコア（全部位が有限値） | 合格 |

参考測定（2026-08-01, macOS 14 / Apple Silicon, `symmetry-2d/1.0.0`）:

| 入力 | overall | contour | eyes | nose | mouth |
|---|---:|---:|---:|---:|---:|
| 合成対称顔 | 44.3 | 27.7 | 36.5 | 61.2 | 51.9 |
| 同じ顔を x 方向に剪断 | 13.8 | 4.1 | 3.0 | 30.9 | 17.2 |

対称な入力でも 100 にならないのは、**指標の感度係数 `k_p` が未校正**だからであり、
モデルの誤りではない。合成点群に対する幾何計算そのものは
`tests/domain/test_symmetry.py` で許容差 `1e-6` 内の 100 を確認している（AC-SCORE-01）。

## 既知の制約

- 学習データの構成、肌色・年齢・眼鏡・髭・頭部被覆などの slice 別性能は上流が
  公表しておらず、本リポジトリでも未評価である。公平性の主張はできない。
- 合成描画顔での検出は経路確認であり、実写での精度を意味しない。
- 極端な横顔、強い遮蔽、低照度、モーションブラーでは検出が落ちる。品質ゲート
  （§8.2）で `HOLD` に倒し、スコアを出さないことで対処する。
- 姿勢の符号規約は合成回転による contract test（`tests/domain/test_pose.py`）で
  固定しているが、実写映像での符号確認は未実施（OD-04 は未解決）。確認手順と
  必要な集計証跡は `docs/adr/0002-head-pose-coordinate-signs.md` に記録する。
- `z` 座標はモデルの 3D 事前分布に依存する相対値であり、MVP の 2D 指標では
  landmark map の導出時にのみ使用し、スコア計算には使わない。

## フォールバックと取り消し

- モデル未配置・sha256 不一致では読み込まず、UI に取得手順を表示する。
- 検出失敗は当該フレームのみを `ERROR` とし、ストリームは継続する。
- 差し替えは manifest の `version` / `sha256` 更新と、
  `scripts/derive_landmark_map.py --check` の再実行を条件とする。マップが変われば
  `landmark_map_version` と `metric_version` を上げる。
