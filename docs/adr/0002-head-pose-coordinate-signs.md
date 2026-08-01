# ADR 0002: Head pose の座標・符号規約

- Status: Proposed — 実写確認待ち
- Date: 2026-08-01
- Requirements: FR-009, §8.2.1, OD-04

## Context

MediaPipe Face Landmarker の facial transformation matrix は、表示ミラーの有無と
Euler角の軸順を混同すると、ユーザーへの左右案内が反転する。アプリは画像を保存せず、
非ミラー分析座標だけを品質判定とexportへ用いる必要がある。

## Decision

3×3部分を極分解してscale / shearを除き、`R = Ry(yaw) @ Rx(pitch) @ Rz(roll)` の
intrinsic Y-X-Z として分解する。非ミラー画像上で、次を正とする。

- yaw: 鼻が画像の右へ動く
- pitch: 鼻が上へ動く
- roll: 時計回り

表示だけをミラー化しても、保存値と品質判定の符号は変えない。UIの行動案内だけが
表示ミラーを考慮する。

## Automated evidence

`tests/domain/test_pose.py` はidentity、±単軸回転、合成回転、scale / shear除去、
gimbal lock、不正行列を検証する。この証跡は数式contractを固定するが、採用assetと
実カメラを通した経験的確認の代替にはしない。

## Required empirical check

本人の明示同意後、非ミラー表示で正面から次の順に単軸運動を行う。

1. yaw: 正面 → 鼻を画像右 → 正面 → 画像左
2. pitch: 正面 → 鼻を上 → 正面 → 下
3. roll: 正面 → 時計回り → 正面 → 反時計回り

各軸で他2軸を可能な範囲で固定し、正方向区間の中央値が正、負方向区間の中央値が負、
正面区間の中央値が0°近傍であることを確認する。映像とlandmarkは保存せず、端末、
ブラウザ、model version / sha256、各区間の件数・中央値・符号一致率だけを本ADRへ追記する。

## Empirical evidence

未実施。検証映像がリポジトリまたはローカル評価環境に提供されていないため、
OD-04は未解決とする。実写結果なしに符号規約を承認済みへ変更しない。
