# Model card: age-mobilenetv2-utkface-v1

## Status

研究用候補。配布 ONNX asset、固定 test split、license inventory のリポジトリ登録は未完了であり、release 済みモデルではない。

## Intended use

本人または明示同意を得た人物の、単一かつ正面に近い顔 crop から、次の参考年齢帯を端末内で推定する。

`0-9`, `10-19`, `20-29`, `30-39`, `40-49`, `50-59`, `60-69`, `70+`

実年齢の断定、本人確認、医療、採用、保険、教育、監視、人物評価には使用しない。

## Architecture and preprocessing

- MobileNetV2
- 8 class classifier
- 入力: RGB 224×224
- scale: `0..255` → `0..1`
- normalize mean: `[0.485, 0.456, 0.406]`
- normalize std: `[0.229, 0.224, 0.225]`
- runtime: ONNX Runtime CPU provider

## Training and reported evaluation

- Dataset: UTKFace-Cropped
- Split: train 70%、validation 15%、test 15%、`random_state=42`、stratified
- Optimizer: Adam
- Learning rate: `1e-4`
- Reported Accuracy: 約 59%
- Reported Macro F1: 約 0.53
- Reported Weighted F1: 約 0.57

これらは創造工房セミナー資料の報告値で、アプリリポジトリ内ではまだ再現検証していない。`.pth` と ONNX の同一 test split 評価差を 2 percentage points 以内にすることが AC-AGE-01 である。

## Known limitations

- 10–19 歳と 40–69 歳付近の分類が相対的に弱いと報告されている。
- class imbalance の影響がある。
- UTKFace の母集団、ラベル品質、撮影条件に由来する偏りを含み得る。
- 眼鏡、髭、頭部被覆、照明、カメラ、顔の向き、crop の違いで出力が変わり得る。
- softmax の最大値はモデル内部の相対値で、正解確率や人物についての確実性を保証しない。

## Required release evidence

- 元 checkpoint と ONNX の SHA-256
- Dataset / checkpoint / ONNX の配布・利用ライセンス
- 固定 test split の Accuracy、Macro / Weighted F1、class 別 precision / recall / F1、confusion matrix
- `.pth` と ONNX の parity report
- 年齢帯、肌色、眼鏡、髭、頭部被覆、照明、camera の slice 評価
- 対応 ONNX Runtime、opset、CPU provider、アプリ版
