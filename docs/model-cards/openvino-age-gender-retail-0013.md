# Model card — age-gender-recognition-retail-0013

| 項目 | 内容 |
|---|---|
| name | `age-gender-recognition-retail-0013` |
| version | source `open-model-zoo/2023.0/FP16` / runtime `FP16-onnx-v1` |
| upstream | Open Model Zoo / Intel |
| asset release | Open Model Zoo 2023.0 / FP16 |
| status | 公式IR・派生ONNXのmanifest登録、変換再現test、OpenCV smoke済み。公式IR parity、slice評価、3 OS smokeは未完（OD-07 open） |
| runtime | `opencv-contrib-python` DNN / CPU |
| format | source: OpenVINO IR XML + BIN / runtime: ONNX opset 13 |
| license | Apache-2.0。条件に従う商用利用・改変・再配布が可能 |

## 固定asset

| file | size | SHA-256 |
|---|---:|---|
| `age-gender-recognition-retail-0013.xml` | 62,543 bytes | `2de3182a773f31d2c2ed91197ee069feae340a505add157a4a92067d333cf067` |
| `age-gender-recognition-retail-0013.bin` | 4,276,038 bytes | `7c6e3da30e369fd870f6d208fc6b08f7cc15a137c1c9c1030b911a5fad31da13` |
| `age-gender-recognition-retail-0013.onnx` | 8,556,137 bytes | `e86b01b71cce307c1a58d9b3db3777be2d95993f702f5b6b29d02bf4efe296bc` |

- XML: `https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml`
- BIN: `https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin`
- source pair identity: `b2917f682865b378589103a46442b3ed6de97ff48e3425a192c96d690bfa5a5e`
- ONNXは`scripts/convert_demographic_model.py`の`ir11-to-onnx-opset13-v1` recipeで、検証済みIRから決定的に生成する。変換toolは`onnx 1.22.0`、`numpy 2.5.1`、`ml-dtypes 0.5.4`、`protobuf 7.35.1`へ固定する。
- 公式IRと派生ONNXはrepositoryへ同梱しない。IRを明示的なbootstrap操作で取得し、ONNXも明示操作でローカル生成する。

## 意図された用途

本人または明示同意を得た18〜75歳の人物の単一顔cropから、連続年齢と`[female, male]`の二値確率を端末内で推定する。年齢は「モデル推定の参考年齢」、性別は「外見からのモデル推定の参考性別」と表示し、左右差・肌表面参考値とは分離する。

## 入出力契約

- 入力: BGR、`float32`、NCHW `1×3×62×62`、0〜255。顔幅62px以上を想定。
- 年齢出力: `fc3_a`、shape `1×1×1×1`。値を100倍した年齢。
- 性別出力: `prob`、shape `1×2×1×1`、順序`[female, male]`。
- 18未満または75超の出力、NaN/Inf、不正shapeは結果を出さず、左右差分析を継続する。
- 実行時model identity: 派生ONNXのSHA-256。元IR pair identityと変換recipeはmanifestから追跡する。

## 上流公開評価

上流文書では、約20,000人の内部validation datasetについて年齢平均誤差6.99年、性別accuracy 95.80%とされる。dataset自体とslice内訳は公開されておらず、本アプリの撮影条件や利用者集団で同じ精度を保証しない。

## 既知の制約と禁止用途

- 子どもの顔はtraining setに含まれず、対象は18〜75歳に限定される。
- 二値ラベルは外見からのモデル分類であり、本人の自己認識、法的性別、身体的特徴の確定ではない。
- 肌色、年齢、性別、眼鏡、髭、頭部被覆、照明、camera別の性能は本リポジトリで未評価である。
- 医療、本人確認、採用、保険、教育、監視、人物比較・順位付けに使用しない。
- 顔埋め込みを生成・保存しない。入力cropや結果を暗黙保存・外部送信しない。

## Release gate

固定URL、元IRと派生ONNXのsize・個別SHA-256、変換recipe、license inventoryのmanifest登録は完了した。モデル取得・変換は引き続き明示操作だけに限定する。macOS Apple Siliconでは変換再現testとOpenCV実モデルsmokeが成功した。一般リリースには次が残る。

1. 固定評価で参照runtimeとの年齢差0.1年以内、性別確率差`1e-4`以内。
2. 年齢・性別・肌色・撮影条件slice評価とlimitations。
3. Windows / Ubuntuを含む3 OS runtime確認。
4. 非公開の内部validation datasetについて、学習データのlicense・同意根拠・保持方針を確認できない制約の法務判断。
