# AGENTS.md — Face Symmetry Lab 共通実行規約

この規約は、このリポジトリで調査・設計・実装・テスト・文書変更を行う AI エージェントに適用する。目的は、使用する AI が変わっても、仕様、安全性、テスト、最終報告の手順を一定に保つことである。

## 1. 指示と正本

競合時は、実行環境の強制指示、現在のユーザー指示、本ファイル、`requirement.md`、`STRUCTURE.md`、テスト、実装の順に従う。

- プロダクト仕様の唯一の正本: `requirement.md`
- 探索用の索引: `STRUCTURE.md`
- テストと要件の索引: `testbox.md`
- 設計判断: `docs/adr/`
- モデルの評価・制約: `docs/model-cards/`

README や ADR に要件本文を複製しない。仕様を変える場合は、実装より先に `requirement.md` の要件 ID、受け入れ基準、マイルストーンを更新する。

## 2. 作業開始

変更前に次を行う。

1. `STRUCTURE.md` を一度読み、対象 module、対応 test、要件節を特定する。
2. `requirement.md` の対象要件、共通の §1.3、§10、受け入れ基準を読む。3 module 以上、データ契約、モデル、プライバシーへ跨る変更では全文を確認する。
3. `testbox.md` で既存 test を特定する。
4. `git status --short` と `git diff` で既存変更を確認し、ユーザーの変更を保護する。
5. 実装順と検証方法を短く決める。

`STRUCTURE.md` は索引であり仕様ではない。対象を見つけた後は、関係 module と test を優先して読み、全リポジトリ探索を繰り返さない。

## 3. 不変条件

1. 画像、映像、landmark、年齢帯、左右差スコアを外部送信または暗黙保存しない。
2. Streamlit は `127.0.0.1` だけで待ち受け、利用統計、STUN、TURN を既定無効にする。
3. 同意前に画像 upload や camera permission を開始しない。
4. 年齢帯は断定年齢ではなく「モデル推定の参考年齢帯」と表示する。
5. 年齢帯と左右差を別結果として扱い、相関・因果・人物価値を示唆しない。
6. 医療、本人確認、採用、保険、教育、監視、人物比較へ使用しない。
7. モデルを暗黙取得・更新せず、版と SHA-256 を追跡する。
8. 顔埋め込みと性別推定を実装しない。
9. `cv2` provider は `opencv-contrib-python` 1 distribution に限定する。
10. 学習用 PyTorch / torchvision / InsightFace をアプリの runtime lock に加えない。
11. 欠測、NaN、Inf、不正 shape を結果や UI へ流さない。
12. 年齢帯推論の失敗で左右差分析を停止しない。

不変条件を変える必要がある場合、実装前に `requirement.md` と ADR を更新する。

## 4. TDD

実装・修正は Red → Green → Refactor → Verify の順で行う。

- Red で要件または不具合を再現する失敗 test を追加する。
- Green は test を通す最小実装にする。
- test を削除、skip、弱体化して通してはならない。
- 外部モデルや network は unit / PR test から呼ばず、fake、fixture、contract test を使う。
- 新しい production file には対応 test を作る。
- test の追加・移動・削除時は `testbox.md` を同じ変更で更新する。

## 5. 依存方向

`ui → application → domain` を守り、adapter は application の port を実装する。

- `domain`: 純粋計算と immutable DTO。Streamlit、MediaPipe、ONNX Runtime、OpenCV を import しない。
- `application`: use case、状態、入力検証、port。UI 描画とモデル自動取得を行わない。
- `adapters`: MediaPipe、ONNX Runtime、export、model store。表示文言と価値判断を持たない。
- `ui`: composition と表示。推論式、幾何、ファイル永続化を実装しない。

学習 notebook と checkpoint はアプリに直接依存させず、ONNX asset、model card、hash、固定評価 report だけを境界にする。

## 6. 構成索引と検証

ファイルを追加・削除・移動した場合:

```bash
uv run python scripts/generate_structure.py --include-untracked
```

終了前の一括検証:

```bash
uv run python scripts/verify.py
```

個別には format、lint、`mypy src`、関連 test、全 test、構成 check、`git diff --check` を実行する。実モデル smoke、E2E、performance、offline test が環境不足で未実行なら、その理由を明記する。

## 7. Git

明示指示なしに `git add`、commit、branch 作成、merge、rebase、tag、push、PR 作成を行わない。調査用の status、diff、log は実行してよい。

ファイル変更時の最終報告では、日本語 Conventional Commit の候補を 1 つ以上示す。その際に、必ず、それぞれのCommitでの対象ファイルを全て表示しなさい。

## 8. 最終報告

次の順で簡潔に報告する。

1. 実装結果
2. 主な変更
3. テスト
4. 未解決事項
5. ユーザー確認
6. 提案コミット
7. 次の提案
