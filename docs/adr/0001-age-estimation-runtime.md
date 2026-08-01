# ADR-0001: 年齢帯推定は ONNX Runtime adapter として統合する

- Status: Superseded by ADR-0003
- Date: 2026-07-30
- Related requirements: FR-022〜024, AC-AGE-01〜02, NFR-021

## Context

創造工房セミナーでは、UTKFace-Cropped を 8 年代に分けて MobileNetV2 を fine-tune し、InsightFace baseline と比較した。fine-tuned MobileNetV2 の報告値は Accuracy 約 59%、Macro F1 約 0.53 で、現時点の初期候補として最も良い。

一方、学習 notebook の `.pth` を直接ロードするために PyTorch / torchvision をアプリへ加えると、配布容量、起動時間、CPU / GPU wheel の解決、脆弱性対応が重くなる。InsightFace を runtime に加えると、公開モデルの利用条件に加え、MediaPipe の `opencv-contrib-python` と InsightFace の `opencv-python` が同じ `cv2` namespace を提供する問題がある。

## Decision

- 学習・比較はアプリ環境から分離する。
- `best_age_model_mobilenetv2.pth` を 8 logits 出力の ONNX へ export する。
- アプリは ONNX Runtime CPU provider だけで年齢帯を推論する。
- 入力は RGB 224×224、ImageNet mean / std、出力順は `0-9` から `70+` の 8 区分に固定する。
- モデルは暗黙に取得せず、検証済みローカル asset の SHA-256 と版を結果へ含める。
- 年齢帯の結果・失敗は左右差分析から分離する。
- 性別推定は本 ADR と MVP の対象外とする。

## Consequences

- アプリ lock が軽くなり、OpenCV provider を 1 つにできる。
- notebook から ONNX への export parity とモデル配布ライセンスの確認が M0 の release gate になる。
- 現在の静止画 vertical slice は UTKFace-Cropped 相当の単一顔 crop を前提とする。一般画像向けの顔検出・alignment は M1 で MediaPipe と統合する。
