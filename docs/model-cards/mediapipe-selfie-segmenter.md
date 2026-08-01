# Model card — MediaPipe Image Segmenter (selfie_segmenter)

| 項目 | 内容 |
|---|---|
| name | `selfie_segmenter` |
| version | `mediapipe-image-segmenter/selfie-segmenter-float16-1` |
| created_at (取得日) | 2026-08-01 |
| owner | Google（MediaPipe Authors）。本リポジトリは配布せず、参照のみ |
| asset | `selfie_segmenter.tflite`（249,537 bytes） |
| sha256 | `191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b` |
| source | `https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite` |
| license | Apache-2.0 |
| runtime | MediaPipe Tasks Vision 0.10.35 系 / CPU delegate |

## 意図された用途

- 端末内での人物前景confidence maskの取得と、それを用いた背景の黒/白置換表示。
- 表示専用機能であり、ON/OFF切替可能（既定OFF）。左右差スコア、年齢・性別推定、
  肌表面参考値のいずれの入力にも使わない（§14.3b）。

## 禁止する用途

`requirement.md` §2.2 / §10.7 に従う。医療診断、本人確認、採用、保険、教育、監視、
人物間の比較・順位付けには使用しない。segmentation maskや合成後の映像を暗黙保存・
外部送信しない。

## 入力・出力契約

| 項目 | 内容 |
|---|---|
| 入力 | RGB `uint8` 画像（`mp.Image` / `ImageFormat.SRGB`） |
| running mode | `VIDEO` + 単調増加 timestamp（realtimeのみ、静止画では使用しない） |
| 出力 | 単一チャンネルの前景confidence mask（`0.0`背景〜`1.0`前景、入力と同じ高さ・幅） |
| 合成 | `domain.frame_adjustments.composite_background` で表示用frameへ黒 or 白を合成する。合成結果は解析へ戻さない |

## 評価

上流の精度評価は未実施であり、本リポジトリで実施したのは
`tests/model/test_real_segmenter.py` のsmoke検証のみである。

| 検証 | 結果 |
|---|---|
| asset の sha256 と manifest の一致 | 合格 |
| 出力shapeが入力と同じ高さ・幅、単一channel | 合格 |
| 出力が有限値で `[0, 1]` に収まる | 合格 |

**既知の制約（重要）**: 本リポジトリのsmoke fixtureは、実在人物を含めないための合成描画顔
（線画ベース、§21.4）である。この合成画像に対しては前景confidenceがほぼ0（人物として
認識されない）ことを確認済みである。これは、モデルが実写selfie写真で学習されており、
line-art調の合成画像を人物と認識しないためと考えられ、モデルの不具合ではない。
したがって本リポジトリのCIでは「合成画像で前景が高confidenceになること」は検証できず、
shapeと有限値だけを検証する。実写での前景抽出精度は、`FACE_SYMMETRY_LAB_SEGMENTER_SMOKE_IMAGE`
環境変数で指定するローカル画像（リポジトリへコミットしない）でのみ手動確認できる。

## 既知の制約

- `mediapipe.tasks.python.vision.ImageSegmenter`の`numpy_view()`はMediaPipe/TFLiteが
  所有するmemoryへの参照であり、コピーせずに保持すると後続呼び出しで値が書き換わりうる
  ことを実機テストで確認した（`tests/model/test_real_segmenter.py`実行中に発生した
  flakinessから特定）。`adapters/inference/mediapipe_segmenter.py`は取得直後に
  `np.array(..., copy=True)`で確実にコピーし、この問題を回避している。
- 構築直後の初回推論はXNNPACK delegateの初期化に起因すると見られる不安定な出力を示す
  ことがあり、`MediaPipeSegmenter.__init__`内で1回分の推論を破棄するwarm-upを行う。
- 実写人物・照明・服装・背景条件別のslice評価は未実施。
- 半透明maskの縁（髪、眼鏡、手など）の精度は上流が公表しておらず未評価。
- 極端な低照度、逆光、背景と酷似した服装では前景/背景の境界が不安定になりうる。
  合成に失敗しても左右差分析・年齢性別推定は継続する（表示専用機能のため）。

## フォールバックと取り消し

- モデル未配置・sha256不一致では読み込まず、背景合成トグルはOFFのまま操作できない
  旨をUIに表示する。
- 1フレームの推論失敗は当該フレームの背景合成だけを飛ばし、ストリームや左右差分析は
  継続する（NFR-017と同じ「1フレーム失敗でstreamを止めない」方針）。
- 差し替えはmanifestの`version` / `sha256`更新を条件とする。
