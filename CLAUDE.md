# CLAUDE.md — Face Symmetry Lab

## 1. プロジェクト概要

顔画像・カメラ映像から左右差（対称度）と参考年齢帯をローカル完結で推定する Streamlit アプリ。Python 3.13 / uv 管理で、推論は MediaPipe と ONNX Runtime をローカル資産に対して行い、顔に関するデータを一切外部へ出さない。

## 2. コマンド

すべて実在を確認済み。ここに無いコマンドを推測で使わない。

| 目的 | コマンド |
|---|---|
| 一括検証（終了前に必須） | `uv run python scripts/verify.py` |
| format | `uv run ruff format --check .` |
| lint | `uv run ruff check .` |
| 型 | `uv run mypy src` |
| テスト | `uv run pytest` |
| モデル実機テスト | `uv run pytest -m model` |
| 構成索引の検証 | `uv run python scripts/generate_structure.py --include-untracked --check` |
| 構成索引の更新 | `uv run python scripts/generate_structure.py --include-untracked` |
| セットアップ | `uv sync --frozen` → `uv run python scripts/bootstrap_models.py` |
| 起動 | `uv run face-symmetry-lab` |

`scripts/verify.py` は CI（`.github/workflows/ci.yml` の `quality` job）と同じ 6 ステップを同じ順で走らせる。ローカルで通れば CI も通る。

**IMPORTANT: ファイルを追加・削除・移動したら `generate_structure.py --include-untracked` を実行する。** 未追跡ファイルもファイル一覧 SHA-256 に混ざるので、docs を先にコミットしてから実装の作業単位へ入る。

## 3. 正本と優先順位

競合時は次の順に従う。実行環境の強制指示 > 現在のユーザー指示 > 本ファイル > `requirement.md` > `STRUCTURE.md` > テスト > 実装。

- プロダクト仕様の唯一の正本: `requirement.md`（要件 ID・受け入れ基準・マイルストーン）
- 探索索引: `STRUCTURE.md`（`scripts/generate_structure.py` の生成物。索引であって仕様ではない）
- テスト対応表: `testbox.md` ／ 設計判断: `docs/adr/` ／ モデル制約: `docs/model-cards/`

**YOU MUST: 仕様を変えるときは実装より先に `requirement.md` を更新する。** README や ADR へ要件本文を複製しない。

## 4. 作業開始の手順

1. `STRUCTURE.md` を一度読み、対象 module・対応 test・要件節を特定する
2. `requirement.md` の対象要件、§1.3、§10、受け入れ基準を読む。3 module 以上／データ契約／モデル／プライバシーに跨る変更なら全文を確認する
3. `testbox.md` で既存 test を特定する
4. `git status --short` と `git diff` で既存の未コミット変更を確認し、それを壊さない
5. 実装順と検証方法を短く決める

対象を見つけた後は関係する module と test だけを読む。全リポジトリ探索を繰り返さない。

## 5. 規約

### 依存方向

`ui → application → domain` を守り、adapter は application の port を実装する。

- `domain`: 純粋計算と immutable DTO
- `application`: use case、状態、入力検証、port
- `adapters`: MediaPipe、ONNX Runtime、export、model store
- `ui`: composition と表示

### プライバシーと AI 安全（不変条件）

**IMPORTANT: この節を変える必要が出たら、実装前に `requirement.md` と ADR を更新する。**

不変条件の正本は `requirement.md` §1.3.1 であり、番号と順序はそこが定義する。ここには要約だけを置き、全文は複製しない。

1. 画像・映像・ランドマーク・派生値は端末内に閉じ、外部送信と暗黙のファイル保存を行わない
2. Streamlit は `127.0.0.1` だけで待ち受け、利用統計・STUN・TURN を既定無効にする
3. 同意を得てから画像 upload と camera permission を開始する
4. 年齢は断定年齢ではなく「モデル推定の参考年齢」として表示する
5. 参考年齢・参考性別と左右差は別々の結果として扱う。**利用者自身の測定結果と参照統計を結び付けず、両者を同じ画面へ並べない**（`FR-039` の限定解除は 2026-09-03 に撤回、ADR-0006 は Superseded）
6. モデルは版と SHA-256 を追跡する
7. 顔埋め込みを追加しない（例外なし）。新規 ML モデルも原則追加しないが、`age-gate/1.0.0`（FR-038）だけは限定解除済みの唯一の例外（ADR-0005）
8. 対称度は「試験中の指標」、対称化画像は「加工画像」であることを UI に明記する
9. `cv2` provider は `opencv-contrib-python` 1 distribution に限定する
10. 採点に影響する自動調整を持たない
11. 欠測・NaN・Inf・不正 shape を結果や UI へ流さない
12. ある推論の失敗が、他の分析・表示・エクスポートを止めない（失敗分離）。年齢・性別推論が失敗しても左右差分析は続行する

## 6. 禁止事項

### 実装してはならないもの

- **YOU MUST NOT: 顔埋め込みを実装しない。** この禁止に例外はない（不変条件 7）
- **新規 ML モデルを追加しない。** 例外は `age-gate/1.0.0`（FR-038、ADR-0005）1 件だけで、これは 2026-08-15 に限定解除済み。他を足すときは `requirement.md` §1.3.1 を先に改訂する
- 医療・本人確認・採用・保険・教育・監視・人物比較の用途を作らない。**年齢ゲート（FR-038）を年齢認証・未成年判定・入場制限へ使わない**（§10.11）
- 年齢帯と左右差の相関・因果・人物価値を示唆する文言や UI を作らない
- **偏差値・順位・パーセンタイル・分位・合否・「平均より上／下」を作らない。** 利用者自身の測定結果と参照統計を結び付けず、両者を同じ画面へ並べない（§10.7）。参照ページ（FR-037）へ利用者自身の測定結果を持ち込まない
- **「全体のスコア」（`display-scale/1.0.0`、§8.6.7）を作らない。** 2026-09-03 に撤回済み。画面へ出す左右差は全体の対称度と部位別 4 つの対称度だけで、`s_3d` は記録列 `distance3d_parts_equal_raw` と age-gate の特徴としてのみ使う
- **モデルの内部判断を利用者へ属性として提示しない。**「18歳未満と判定されました」「未成年です」は禁止。「この画像では参考年齢を表示できません」までとする（§10.11-2）
- 価値判断語（美しい／良い／悪い）を表示文言に使わない
- 学習用 PyTorch / torchvision / LightGBM / InsightFace をアプリの runtime lock に加えない。**推論 runtime は `opencv-contrib-python` の DNN と `onnxruntime` の 2 つだけ**で、後者は FR-038 の決定木モデル専用（§10.8）
- 欠測・NaN・Inf・不正 shape を結果や UI へ流さない
- モデルを暗黙に取得・更新しない

### 層をまたいでしてはならないこと

- `domain` から Streamlit / MediaPipe / ONNX Runtime / OpenCV を import しない
- `application` で UI 描画とモデル自動取得を行わない
- `adapters` に表示文言と価値判断を持たせない
- `ui` に推論式・幾何計算・ファイル永続化を実装しない
- 学習 notebook と checkpoint をアプリから直接参照しない（境界は ONNX asset / model card / hash / 固定評価 report だけ）

### 進め方でしてはならないこと

- **YOU MUST NOT: テストを削除・skip・弱体化して通さない。**
- 実装を書いた後に、それへ合わせてテストを書かない
- 外部モデルや network を unit / PR test から呼ばない（fake・fixture・contract test を使う）
- test を追加・移動・削除したまま `testbox.md` を更新せずに終えない
- 依頼されたタスクと無関係なリファクタを混ぜない
- 「常に 0 を返す実装」でも通るテストを書かない（罠を仕込んで実際に落ちることを確かめる）

## 7. 開発フロー

### 役割分担

- **メインスレッド（Opus）**: 設計・分析・サブエージェントへの指示・実装後のレビューと報告
- **implementer サブエージェント（Sonnet）**: 実装
- **IMPORTANT: 実装は原則すべて implementer へ委譲する。** Opus が直接編集してよいのは、委譲する方が非効率な軽微な修正（数行の typo、1 ファイルの定数変更、コマンド実行）だけ

標準起動コマンド: `claude --model opus`

### TDD

Red → Green → Refactor → Verify。

- **YOU MUST: テストは実装より先に書く。実装を書いた後にそれへ合わせてテストを書かない。**
- テストは最低限でよい。全網羅より「壊れたら気づける代表ケース」を優先する
- 新しい production file には対応 test を 1 つ作る
- test を増減したら `testbox.md` を同じ変更で更新する
- 終了前に `uv run python scripts/verify.py` を通す。環境不足で実行できないテスト（実モデル smoke / E2E / performance）があれば、その理由を報告に明記する

### Git

- **YOU MUST: `git commit` と `git push` は誰も自律実行しない。必ず人間が実行する。** `.claude/scripts/block-commit.sh` が機械的にブロックする
- `git add` / branch 作成 / merge / rebase / tag / PR 作成も、明示指示なしに行わない
- 調査用の `git status` / `git diff` / `git log` は自由に実行してよい
- 作業単位ごとに人間がコミットしてから次へ進む。`git status --short` が空でない状態で次の作業単位を始めない

## 8. 実装後レポート

実装が終わったら、この形式で報告する。

### ✅ 人間が確認すべき点

リスクの高い変更や判断が分かれうる点を箇条書き。なければ「特になし」。
特に以下に該当する変更は必ず挙げる:
不可逆な操作 / 認証・認可 / 外部入力の処理 / 金額・日時などの計算 /
状態変更と並行処理 / 静かに間違った値を返しうる箇所

### 📝 コミット案

```
<type>(<scope>): <日本語での説明>
```

対象ファイル:
- 変更した全ファイルを列挙する(省略・「など」は禁止)

### ➡️ 次の提案(1つだけ)

最も優先度が高いものを1つ。複数出さない。

## 9. ディレクトリ構成の方針

Web 的な MVC は採らない。このアプリは Streamlit 単一プロセスの計算主体アプリで、HTTP ルーティングも ORM も持たず、価値の中心が幾何計算とモデル推論の正しさにあるため、Model/View/Controller より **層 + port/adapter** の方が実態に合う。

| 層 | 役割 | MVC で言えば |
|---|---|---|
| `domain` | 純粋計算と immutable DTO | Model |
| `application` | use case・状態・port | Controller / Service |
| `adapters` | MediaPipe / ONNX / export / IO | Infrastructure |
| `ui` | Streamlit の composition と表示 | View |
| `runtime` | frame scheduling と表示キャッシュ | — |

新しいファイルはこの 5 層のどれかへ置く。層が決まらないものは、`domain` へ置けるところまで設計を分解する。`tests/` はこの層構成をそのまま鏡写しにする。
