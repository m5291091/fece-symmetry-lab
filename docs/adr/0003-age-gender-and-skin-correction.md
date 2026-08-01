# ADR-0003: 18〜75歳の年齢・性別推定と撮影補正後の肌表面参考値

- Status: Accepted
- Date: 2026-08-01
- Related requirements: FR-022〜025, AC-AGE-01〜02, AC-APPEARANCE-01, NFR-021
- Supersedes: ADR-0001

## Context

従来候補はUTKFaceの8年代分類で、学習checkpoint、ONNX export、配布ライセンス、固定評価が未完了だった。利用者は8年代分類の削除、18〜75歳を対象にした年齢と性別の追加、およびcamera補正を踏まえた肌品質評価を求めた。

cameraの自動露出・white balance・HDR・ISPは端末とbrowserごとに異なり、WebRTCから一律に固定・逆変換できない。したがって肌の医学的状態や絶対品質はRGB画像だけから分離できない。

## Decision

- 8年代分類と`AgeBand`を削除し、Open Model Zoo 2023.0 FP16 `age-gender-recognition-retail-0013`を採用する。
- 対象は18〜75歳に限定する。62×62 BGRの顔cropを入力し、連続年齢と`[female, male]`の二値確率を返す。
- 表示は「モデル推定の参考年齢」「外見からのモデル推定の参考性別」とし、自己認識・法的性別を断定しない。
- OpenVINO IRのXML/BINを出所の正本とする。PyPIのOpenCV wheelにはOpenVINO backendがないため、検証済みIRから開発用scriptで決定的に生成したONNXを、既存の`opencv-contrib-python` DNNでCPU推論する。
- XML/BINは暗黙取得しない。Intel公式storageの固定URL、size、Apache-2.0、個別SHA-256をmanifestへ登録し、明示操作でのみ取得する。派生ONNXも元asset、変換script・tool版、size、SHA-256をmanifestへ固定し、実行したONNX hashを結果へ記録する。
- ONNX変換用packageは開発scriptへ隔離し、アプリのruntime lockへOpenVINO、ONNX Runtime、学習runtime、別の`cv2` providerを加えない。公式IRとのparity、slice評価、3 OS smokeはリリース条件として残す。
- 顔cropには端末内のGray-world white balanceとrobust明度正規化を適用する。補正gainを結果へ含める。
- 頬patchの色調均一性と局所textureを0〜100の「撮影条件依存の肌表面参考値」として出す。疾患、肌タイプ、健康、美しさを推定せず、幾何スコアへ混合しない。

## Consequences

- 既存schemaの8年齢確率field削除は破壊的変更なので、schemaを2.0.0へ上げる。
- 公開モデルの年齢平均誤差6.99年、性別accuracy 95.80%は内部validation datasetの値であり、本アプリで別途fixed/slice評価が必要になる。
- 公式IRからONNXへの逆変換はOpenVINO公式変換APIの対象外なので、この固定graphで用いる演算だけを明示変換する。上流graph変更時は暗黙追従せず、変換contractとparityを再承認する。
- camera補正は撮影条件差を軽減できるが、物理的な色校正やcamera固有ISPの逆変換ではない。OD-08完了まで肌表面参考値は相対比較専用かつ未校正とする。
