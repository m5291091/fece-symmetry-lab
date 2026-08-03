# Model card — age-gender-recognition-retail-0013

| 項目 | 内容 |
|---|---|
| name | `age-gender-recognition-retail-0013` |
| version | source `open-model-zoo/2023.0/FP16` / runtime `FP16-onnx-v2` |
| upstream | Open Model Zoo / Intel |
| asset release | Open Model Zoo 2023.0 / FP16 |
| status | 公式IR・派生ONNXのmanifest登録、変換再現test、OpenCV smoke済み。公式IR固定入力parityは実測済みだが目標gate未達。onnxruntimeとの三者比較により、gapは変換recipeのbugではなくOpenVINO CPU実行の特性である可能性が高いという追加証拠を記録した（下記参照）。gate値の承認、slice評価、3 OS smokeは未完（OD-07 open） |
| runtime | `opencv-contrib-python` DNN / CPU |
| format | source: OpenVINO IR XML + BIN / runtime: ONNX opset 13 |
| license | Apache-2.0。条件に従う商用利用・改変・再配布が可能 |

## 固定asset

| file | size | SHA-256 |
|---|---:|---|
| `age-gender-recognition-retail-0013.xml` | 43,414 bytes | `a625adb569fb40f6b4e392059409088ec4443b465c8ed571e9e8fb71b5192d64` |
| `age-gender-recognition-retail-0013.bin` | 4,276,038 bytes | `7c6e3da30e369fd870f6d208fc6b08f7cc15a137c1c9c1030b911a5fad31da13` |
| `age-gender-recognition-retail-0013.onnx` | 8,556,149 bytes | `a221460e4bc822f3b48138a5e39c384cd14180d1248308277503c3cf99085712` |

- XML: `https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.xml`
- BIN: `https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/age-gender-recognition-retail-0013/FP16/age-gender-recognition-retail-0013.bin`
- source pair identity（XMLのbytesにBINのbytesを連結したもののSHA-256）: `24a93799f39ea23a0f7a02b5ba9b2256f00675893634fdb26d75fcb6ab8867cb`
- ONNXは`scripts/convert_demographic_model.py`の`ir11-to-onnx-opset13-v2` recipeで、検証済みIRから決定的に生成する。変換toolは`onnx 1.22.0`、`numpy 2.5.1`、`ml-dtypes 0.5.4`、`protobuf 7.35.1`へ固定する。
- 公式IRと派生ONNXはrepositoryへ同梱しない。IRを明示的なbootstrap操作で取得し、ONNXも明示操作でローカル生成する。
- **v2での修正（2026-08-02）**: v1のRecipeは`Result`層（例: `prob/sink_port_0`）のONNX出力tensor名を生成する際、そのままnode名にも再利用しており、同名の計算node（例: `SoftMax`層`prob`）と衝突していた。`opencv-contrib-python`のONNX importerはnode名の重複を許容するため症状が出なかったが、`onnxruntime`はこの重複を`FAIL`として拒否し、load不能だった。v2はnode名にIR側の元々一意な層名（例: `prob/sink_port_0`）を使い、ONNX出力tensor名だけを契約どおり`prob`に保つことで解消した。計算内容・数値は変更していない（下記parity実測はv1と同一値）。

- **XML pinの訂正（2026-08-03）**: 固定していたXMLのsize/SHA-256（62,543 bytes / `2de3182a…`）が、URLが実際に配信する内容と一致していなかった。配信元は2023-10-17以降ずっと43,414 bytes / `a625adb5…`であり、clean checkoutでは`bootstrap_models.py`が必ずsize不一致で失敗していた（CIで顕在化）。原因は、pin値をURLの応答ではなく**ローカルで再シリアライズされたXML**（XML宣言なし・タブでなくスペース字下げ）から採取していたこと。両者はElementTree上で属性まで完全に一致する同一グラフであり（82 layers / IR v11 / 1,3,62,62）、モデルが差し替わったわけではない。同一ディレクトリのBINはsize・SHA-256とも一致しており、配信元と版の正当性は裏付けられている。
- 上記に伴い派生ONNXのSHA-256も更新した（`b50a5e26…` → `a221460e…`）。**グラフと重みはbyte単位で不変**で、差分は末尾55バイト、すなわちONNXの`metadata_props`に記録される`source_xml_sha256`だけである（`metadata_props`を除いた直列化のSHA-256は両者とも`d6927278b175f9dc8597bacf3be854ca5a4108c039d267038760b658f515e486`）。recipeとtool版は変更していない。

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

1. 固定評価で参照runtimeとの年齢差0.1年以内、性別確率差`1e-4`以内。 **未達（下記実測を参照、OD-07 open）。**
2. 年齢・性別・肌色・撮影条件slice評価とlimitations。
3. Windows / Ubuntuを含む3 OS runtime確認。
4. 非公開の内部validation datasetについて、学習データのlicense・同意根拠・保持方針を確認できない制約の法務判断。

### OpenVINO参照runtime固定入力parity（OD-07実測）

`scripts/verify_openvino_parity.py`は、manifest固定済みの公式OpenVINO IR（`openvino.Core`、CPU）と、アプリが実行する派生ONNX（`cv2.dnn.readNetFromONNX`、`opencv-contrib-python==5.0.0.93`）へ同一の固定合成tensor（seed `20260801`、飽和端・一様乱数・正規乱数・肌色勾配の14件、`1×3×62×62` BGR float32）を入力し、`fc3_a`（×100年）と`prob`の差を比較する。`openvino`はアプリのruntime lockへ加えないため、この評価は`uv run scripts/verify_openvino_parity.py`が作る独立環境（`openvino==2026.2.1`、`numpy==2.4.6`）でだけ実行し、`pyproject.toml` / `uv.lock`は変更しない。

macOS Apple Silicon、`openvino==2026.2.1`での実測（2026-08-01、manifest固定済みasset、n=14）:

| 指標 | 実測値（最大） | 目標gate | 判定 |
|---|---:|---:|---|
| 年齢差 | 0.205年 | 0.1年以内 | 不合格 |
| 性別確率差 | 5.88×10⁻³ | 1×10⁻⁴以内 | 不合格 |

全14件中、年齢差は4件、性別確率差は14件全てが目標gateを超過した。差は入力ごとに変動し、単一の異常値ではなく変換全体に分布する（詳細は`tests/model/test_openvino_parity.py`実行時の出力を参照）。目標値は現行のIR→ONNX変換では未達であり、次のいずれかをOD-07の完了条件として承認する必要がある。

- 変換精度を追加調査・改善し、目標値0.1年 / `1e-4`を達成する。
- 実測分布に基づく新しいgate値（例: 年齢0.25年以内、性別確率1×10⁻²以内）へ本カードとOD-07を更新し、根拠として本測定を記録する。

### 変換recipeへの疑い切り分け（onnxruntimeとの三者比較）

上記gap自体は変換recipe（v1→v2のnode名修正含む）の前後で数値が一致しており（0.205117年 / 5.877852×10⁻³のまま）、node名修正はこのgapの原因ではなかった。変換recipe自体に数値上のbugが残っている可能性を切り分けるため、`scripts/verify_onnx_runtime_agreement.py`で同じ固定入力14件を、派生ONNX（v2）に対する**第三の独立runtime**である`onnxruntime==1.20.1`（`CPUExecutionProvider`）でも実行し、`openvino`（参照）・`cv2.dnn`（アプリ実行runtime）・`onnxruntime`の三者を比較した。

macOS Apple Silicon実測（2026-08-02、manifest固定済みasset v2、n=14）:

| 比較 | 年齢差（最大） | 性別確率差（最大） |
|---|---:|---:|
| OpenVINO vs `cv2.dnn`（アプリ） | 0.205141年 | 5.878121×10⁻³ |
| OpenVINO vs `onnxruntime` | 0.205141年 | 5.878121×10⁻³ |
| `cv2.dnn` vs `onnxruntime` | 0.000024年 | 1.072884×10⁻⁶ |

`cv2.dnn`と`onnxruntime`という完全に独立した2つのONNX実行系は、同一の派生ONNXに対してほぼ完全に一致する（年齢差0.000024年、確率差1.07×10⁻⁶）。一方、両者はOpenVINOに対してほぼ同じ量・同じ符号傾向でずれる。これは、

1. `scripts/convert_demographic_model.py`が生成するONNX graphはIRの数値的内容を忠実に再現しており、変換recipe自体には数値上のbugがない可能性が高いこと、
2. gapの実体はOpenVINOのCPU pluginが、元のFP16 IRをCPU上でどう実行するか（内部精度・演算順序・活用する演算実装）に起因する、runtime実装固有の特性である可能性が高いこと、

を示す一定の証拠になる。ただし、これは推定であり、OpenVINO側の内部実行経路を直接検証したものではない。この結果は「変換精度を追加調査・改善」という選択肢の見込みが低いことを示す証拠として記録するに留め、gate値の最終決定は別途プロダクト判断を要する。

3 OS runtime確認とslice評価が残る限り、いずれの場合もOD-07とM0は未完のままとする。
