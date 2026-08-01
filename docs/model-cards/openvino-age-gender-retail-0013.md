# Model card — age-gender-recognition-retail-0013

| 項目 | 内容 |
|---|---|
| name | `age-gender-recognition-retail-0013` |
| version | `open-model-zoo/age-gender-recognition-retail-0013` |
| upstream | Open Model Zoo / Intel |
| status | adapter実装済み、配布asset未登録（OD-07 open） |
| runtime | `opencv-contrib-python` DNN / CPU |
| format | OpenVINO IR XML + BIN |
| license | Open Model Zoo repositoryはApache-2.0。配布assetの法務確認は未完了 |

## 意図された用途

本人または明示同意を得た18〜75歳の人物の単一顔cropから、連続年齢と`[female, male]`の二値確率を端末内で推定する。年齢は「モデル推定の参考年齢」、性別は「外見からのモデル推定の参考性別」と表示し、左右差・肌表面参考値とは分離する。

## 入出力契約

- 入力: BGR、`float32`、NCHW `1×3×62×62`、0〜255。顔幅62px以上を想定。
- 年齢出力: `fc3_a`、shape `1×1×1×1`。値を100倍した年齢。
- 性別出力: `prob`、shape `1×2×1×1`、順序`[female, male]`。
- 18未満または75超の出力、NaN/Inf、不正shapeは結果を出さず、左右差分析を継続する。
- model identity: XMLとBINの各SHA-256を文字列化した組み合わせSHA-256。

## 上流公開評価

上流文書では、約20,000人の内部validation datasetについて年齢平均誤差6.99年、性別accuracy 95.80%とされる。dataset自体とslice内訳は公開されておらず、本アプリの撮影条件や利用者集団で同じ精度を保証しない。

## 既知の制約と禁止用途

- 子どもの顔はtraining setに含まれず、対象は18〜75歳に限定される。
- 二値ラベルは外見からのモデル分類であり、本人の自己認識、法的性別、身体的特徴の確定ではない。
- 肌色、年齢、性別、眼鏡、髭、頭部被覆、照明、camera別の性能は本リポジトリで未評価である。
- 医療、本人確認、採用、保険、教育、監視、人物比較・順位付けに使用しない。
- 顔埋め込みを生成・保存しない。入力cropや結果を暗黙保存・外部送信しない。

## Release gate

次が揃うまでXML/BINをmanifestへ登録せず、正式モデルとして自動取得しない。

1. XML/BINの固定URL、size、個別SHA-256、license inventory。
2. 固定評価で参照runtimeとの年齢差0.1年以内、性別確率差`1e-4`以内。
3. 年齢・性別・肌色・撮影条件slice評価とlimitations。
4. 実モデルsmokeと3 OS runtime確認。
