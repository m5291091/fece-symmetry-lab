"""HUD がブラウザへ渡す資産（CSS と実行時 JS）。§5.2 の例外、§14.3e。

**Python の組み立て（`ui.hud`）から分けてある。** ここにあるのは文字列だけで、
変更理由もレビューの読み方も Python 側とは違う——セレクタが Streamlit の DOM に
追随できているか、`z-index` の積み重ねが崩れていないか、5 Hz の tick が骨格を
作り直していないか、といったブラウザ側の関心である。同じファイルに混ぜていた
間、`ui.hud` は 700 行あり、その大半が読み飛ばされる文字列リテラルだった。

§5.2 がこの層へ課す条件は、そのままここのレビュー観点になる。

* 要素の特定は `st.container(key=...)` 由来の `.st-key-<key>` と、`.stSidebar`
  / `.stAppHeader` のような Streamlit の意味的 class 名だけ。emotion のハッシュ、
  `nth-child`、`data-testid` は使わない（`data-testid` は版で変わり、実際に
  擬似全画面が無言で壊れた）。
* stylesheet は bridge が **一度だけ** 設置し、以後は class の切り替えだけで
  状態を変える。5 Hz で `<style>` を送り直すと文書全体のスタイル再計算が毎秒
  5 回走り、映像が実際にカクつく。
* 映像の上に `backdrop-filter` を置かない。更新頻度ではなく *映像フレームごと*
  に再合成されるため、遅延の原因になる。
* 実行時コードは**ページ側の realm** で動かす。`components.v1.html` の iframe は
  再実行のたびに破棄され、そこで定義した timer callback は id だけ残して止まる。
"""

from __future__ import annotations

from typing import Final

#: ページ側の `<head>` へ一度だけ設置する stylesheet。映像は
#: `st.container(key=...)` が生成する `.st-key-camera_preview_frame iframe` から
#: 辿る。Streamlit 内部の `data-testid` に依存しないため、以前に擬似全画面が
#: 黙って壊れた原因を避けられる。
HUD_STYLESHEET: Final[str] = """
/* §5.1: 箱・余白・文字を 1 つの尺度（ビューポート幅）へ結び、画面が小さいほど
   HUD 全体が小さくなるようにする。

   **これは「はみ出したらスクロール」では解けない。** HUD は
   `pointer-events: none` なので、あふれた分を利用者が引き寄せる手段が無く、
   箱の外へ出た数値はそのまま読めなくなる（`overflow-y: auto` は最後の安全網
   であって、設計上あてにしてよい逃げ道ではない）。固定 px の版では、狭い画面
   で箱がビューポートのほぼ全幅を占めたまま中の文字だけが元の大きさで残り、
   「98.42%」の末尾が切れていた。

   折れ線グラフ（`#fsl-hud-chart`）は `width: 100%` と viewBox で自動的に
   同じ比率で縮むため、高さもこの幅に連動する。 */
#fsl-hud {
    position: fixed; z-index: 1000050;
    /* 上から下へわずかに暗くする。数値が集まる上端のコントラストを保ちつつ、
       下端の但し書きが映像から浮きすぎないようにするためだけの装飾である。
       §5.2 が禁じる背景ぼかしは使っていない——あれは映像フレームごとに再合成
       されるが、これは静的なグラデーションなので合成コストは単色と変わらない。 */
    background: linear-gradient(180deg, rgba(0,0,0,.78), rgba(0,0,0,.68));
    box-shadow: 0 10px 30px rgba(0,0,0,.38);
    color: #fff;
    /* **箱の寸法に padding と border を含める。** 既定の `content-box` では下の
       `width`/`max-width` が内容の幅にしか掛からず、左右 padding（13〜22px）と
       2px の border がその外側へ足される。実測（headless Chrome、ビューポート
       800px）で 210px 指定に対して 240px あった。箱の幅は映像を覆う面積その
       ものなので、宣言値と実寸をそろえる（吹き出し `ui.still_overlay` と同じ）。 */
    box-sizing: border-box;
    width: clamp(210px, 26vw, 360px);
    padding: clamp(10px, 1.2vw, 16px) clamp(13px, 1.6vw, 22px);
    border-radius: 16px;
    border: 2px solid #6B6B6B;
    pointer-events: none;
    font-variant-numeric: tabular-nums;
    font-family: inherit;
    overflow-wrap: anywhere;
    max-width: 92vw; max-height: min(90vh, 720px); overflow-y: auto;
}
#fsl-hud[hidden] { display: none; }
/* 全画面の状態帯: 現在地と、START 前なら次に何をすべきかを示す。
   HUD（左上）や終了ボタン（右上）と衝突しないよう中央へ寄せる。 */
#fsl-chrome {
    position: fixed; top: 24px; left: 50%; transform: translateX(-50%);
    z-index: 1000055; pointer-events: none;
    display: flex; flex-direction: column; align-items: center; gap: 6px;
    font-family: inherit; text-align: center;
}
#fsl-chrome[hidden] { display: none; }
#fsl-chrome-badge {
    background: rgba(0,0,0,.72); color: #fff;
    border: 1px solid rgba(255,255,255,.35); border-radius: 999px;
    padding: 6px 16px; font-size: clamp(.68rem, .9vw, .8rem); letter-spacing: .04em;
}
#fsl-chrome-hint {
    background: rgba(0,0,0,.72); color: #fff;
    border-radius: 12px; padding: 10px 18px;
    font-size: clamp(.8rem, 1.3vw, 1rem); max-width: 70vw;
}
#fsl-chrome-hint[hidden] { display: none; }
/* 文字はすべて `clamp(最小, ビューポート連動, 最大)` で書く。最大値は従来の
   固定値そのままなので、広い画面での見た目は変わらない。最小値は、狭い画面で
   数値 6 文字（「98.42%」）が箱の内寸へ確実に収まる大きさである。

   中間の項は `min(vw, vh)` にして、**足りないほうの寸法へ合わせる。** 箱の高さ
   上限は `90vh` なので、横に広く縦に短い窓（分割表示など）で幅だけを見て決めると
   中身が縦にあふれ、`pointer-events: none` の箱では読む手段が無くなる。 */
/* FR-040 / §5.2 / §5.4: 品質ゲートの状態と、いま有効な平滑化時間。数値より上に
   置くのは、どちらも「下に並ぶ数字をどう読むか」を決める前提だからである
   （測っている最中なのか、何秒の平均なのか）。

   **色を持たせない。** 枠色は箱の border が既に担っており、チップまで塗ると
   同じ情報を色で二重に語ることになる。色を落とした環境ではテキストだけが
   残るべきで、その 1 本の経路を確実にしておく（§5.4）。 */
#fsl-hud-chips {
    display: flex; flex-wrap: wrap; gap: 4px 6px; margin-bottom: .45em;
}
#fsl-hud-chips:empty { display: none; }
.fsl-hud-chip {
    border: 1px solid rgba(255,255,255,.28); border-radius: 999px;
    padding: 2px 9px; opacity: .82; white-space: nowrap;
    font-size: clamp(.58rem, min(.78vw, 1.35vh), .7rem); letter-spacing: .02em;
}
.fsl-hud-chip:empty { display: none; }
#fsl-hud-value {
    font-size: clamp(1.5rem, min(3.6vw, 6.4vh), 3.2rem);
    font-weight: 700; line-height: 1;
}
#fsl-hud-label {
    font-size: clamp(.7rem, min(.9vw, 1.6vh), .85rem); opacity: .75; margin-bottom: .2em;
}
/* FR-039: 参照平均の併記。**対応する測定値のすぐ下に置く**（§5.1）。独立した区画へ
   まとめると、どの数字に対する平均なのかが読み取れなくなる。測定値より小さく薄く
   して、比べる相手であって主役ではないことを大きさでも示す（§5.2: 差を達成度として
   演出しない）。

   `:empty` で畳むのは、併記が無いとき——参考年齢を表示していない、参照データが
   読めない、件数がしきい値未満——に空行を残さないためである。空行は「機能が動いて
   いて失敗している」と読まれる（FR-024 と同じ理由）。 */
.fsl-hud-baseline {
    font-size: clamp(.6rem, min(.8vw, 1.4vh), .74rem);
    opacity: .66; line-height: 1.3;
}
.fsl-hud-baseline:empty { display: none; }
#fsl-hud-baseline-symmetry { margin-bottom: .5em; }
/* H1: 「全体のスコア」行。全体の対称度（#fsl-hud-value）より小さく、部位別の
   行より目立たせる。§8.6.3 は表示名だけの区別であり評価の強調ではない。 */
#fsl-hud-score-row {
    display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;
    font-size: clamp(1rem, min(1.6vw, 2.8vh), 1.35rem); font-weight: 600; margin-top: .5em;
}
#fsl-hud-score-label {
    font-size: clamp(.68rem, min(.85vw, 1.5vh), .8rem); font-weight: 400; opacity: .75;
}
/* H1: 部位別 4 つ。折り返し可能な行にして、狭い HUD 幅でも収める。 */
#fsl-hud-parts {
    display: flex; flex-wrap: wrap; gap: 6px clamp(8px, 1vw, 14px);
    font-size: clamp(.72rem, min(1.05vw, 1.9vh), .95rem); margin-top: 8px;
}
/* FR-039: 部位別は「値」と「その平均」の 2 行で 1 つの塊にする。横並びのままだと、
   4 部位 × 2 値が 1 行に混ざってどの平均がどの部位のものか分からなくなる。 */
.fsl-hud-part { display: flex; flex-direction: column; }
.fsl-hud-part b { font-weight: 400; opacity: .75; margin-right: 4px; }
#fsl-hud-demo { font-size: clamp(.8rem, min(1.2vw, 2.1vh), 1.1rem); margin-top: 8px; }
#fsl-hud-demo:empty { display: none; }
/* FR-039 / §10.12-5: 母集団と但し書き。併記と同じ場所に置くことが要件なので、
   別ページへのリンクで代替しない。数値より一段小さくする。 */
#fsl-hud-baseline-note {
    font-size: clamp(.52rem, min(.7vw, 1.2vh), .64rem);
    opacity: .6; line-height: 1.35; margin-top: 8px;
}
#fsl-hud-baseline-note:empty { display: none; }
/* H1: 折れ線グラフ。骨格（<svg>・<polyline> 5 本・軸ラベル・凡例）は bridge が
   1 度だけ作り、5Hz の tick では `points` 属性だけを書き換える。 */
#fsl-hud-chart { display: block; width: 100%; height: auto; margin-top: 10px; }
.fsl-chart-axis { font-size: 6px; fill: rgba(255,255,255,.6); }
.fsl-chart-line { fill: none; stroke-width: 1.6; stroke-linejoin: round; }
/* H5: 総合値の線の下だけを薄く塗る。5 本のうちどれが総合値かを、線の太さ以外の
   手掛かりでも示すためである。**部位別には塗らない**——4 部位を塗り重ねると
   面が積み上がって見え、部位の合成が総合値であるかのように読める（§8.6:
   全体距離は部位別の加重合成ではない）。 */
.fsl-chart-area { fill: rgba(255,255,255,.13); stroke: none; }
.fsl-chart-overall { stroke: #ffffff; stroke-width: 2.2; }
.fsl-chart-contour { stroke: #4FC3F7; }
.fsl-chart-eyes { stroke: #AED581; }
.fsl-chart-nose { stroke: #FFB74D; }
.fsl-chart-mouth { stroke: #F06292; }
#fsl-hud-legend {
    display: flex; flex-wrap: wrap; gap: 4px clamp(6px, .8vw, 10px);
    font-size: clamp(.58rem, .75vw, .68rem); opacity: .8; margin-top: 2px;
}
.fsl-legend { padding-left: 14px; position: relative; }
.fsl-legend::before {
    content: ''; position: absolute; left: 0; top: 50%;
    width: 10px; height: 3px; margin-top: -1px; background: currentColor;
}
.fsl-legend-overall { color: #ffffff; }
.fsl-legend-contour { color: #4FC3F7; }
.fsl-legend-eyes { color: #AED581; }
.fsl-legend-nose { color: #FFB74D; }
.fsl-legend-mouth { color: #F06292; }
/* pointer events を受け取るのは grip だけにして、HUD の下にある映像側の STOP
   ボタンが他の場所ではそのまま押せる状態を保つ。 */
#fsl-hud-grip {
    pointer-events: auto; cursor: move; touch-action: none;
    margin-top: 10px; font-size: clamp(.65rem, .8vw, .75rem); opacity: .55;
    border-top: 1px solid rgba(255,255,255,.25); padding-top: 6px;
}
.st-key-camera_preview_frame { position: relative; border-width: 3px !important; }
/* `width: auto` は必須である。Streamlit の縦ブロックは width:100% のため、
   `right` で固定した要素はビューポート全体へ広がり、ボタンが左端の外へ
   落ちてしまう。 */
.st-key-fsl_exit_fullscreen {
    position: fixed; top: 24px; right: 24px; left: auto; z-index: 1000060;
    width: auto !important; min-width: 0 !important;
}
.st-key-fsl_exit_fullscreen button {
    background: rgba(0,0,0,.72) !important;
    color: #fff !important;
    border: 2px solid rgba(255,255,255,.65) !important;
    padding: 10px 22px !important;
    font-size: 1rem !important;
}
body.fsl-fullscreen .st-key-camera_preview_frame iframe {
    position: fixed; inset: 0;
    width: 100vw !important; height: 100vh !important;
    z-index: 1000000; background: #000;
}
/* Streamlit 自身の装飾は z-index 999990/999991 にあり、ページ側のオーバーレイ
   が自然に選ぶ値よりはるかに高い。そのため全画面ではその上へ重ね、かつ装飾を
   畳む必要がある。さもないと sidebar が映像の 1/3 を覆い、header が HUD を
   隠す。`.stSidebar`/`.stAppHeader` は emotion のハッシュではなく Streamlit の
   意味的な class 名なので、安定した手がかりとして使える（§5.2）。
   streamlit 1.59.1 で確認済み。 */
body.fsl-fullscreen .stSidebar,
body.fsl-fullscreen .stAppHeader { display: none !important; }
"""

#: WebRTC コンポーネント自身の iframe（same-origin）へ注入する。ページ側の
#: stylesheet は iframe の内側まで届かないためである。`VIDEO_HTML_ATTRS` が
#: ネイティブのコントロールを扱うようになった今、ここに残る役割は 2 つだけ。
#:
#: * 映像を実際に画面いっぱいへ広げる。コンポーネントは video を
#:   `width: 100%` かつ高さ auto で配置するため、全高のフレーム内では上端の
#:   帯状のままになる。フレームのビューポートへ固定して初めて画面を満たす。
#: * START/STOP の行を左端から中央へ移す。フレームが画面幅いっぱいに広がると
#:   左端は実際に押しにくい。`.fsl-controls` はこのアプリが JS から付与する
#:   class である。コンポーネントは emotion/MUI のハッシュしか出力せず、
#:   安定した手がかりを持たないためである（§5.2）。
VIDEO_CHROME_CSS: Final[str] = """
html.fsl-fs, html.fsl-fs body { margin: 0; background: #000; }
html.fsl-fs video {
    position: fixed; inset: 0;
    width: 100vw !important; height: 100vh !important;
    object-fit: contain; background: #000;
}
/* `translateX(-50%)` ではなく auto margin で中央寄せする。transform を持つ
   要素は *fixed* な子孫にとっての包含ブロックになるため、この行が <video> を
   包んでいた場合に映像がビューポートではなくこの pill を基準に配置され、
   隅へ寄ってしまう。 */
html.fsl-fs .fsl-controls {
    position: fixed; left: 0; right: 0; bottom: 44px;
    width: max-content; margin-left: auto; margin-right: auto;
    transform: none;
    z-index: 20; gap: 16px;
    background: rgba(0,0,0,.66); padding: 12px 20px; border-radius: 999px;
    box-shadow: 0 6px 24px rgba(0,0,0,.45);
}
html.fsl-fs .fsl-controls button {
    font-size: 1.05rem !important;
    padding: 10px 26px !important;
    min-height: 48px;
}
"""

#: HUD の実行時コード。コンポーネント iframe ではなく *ページ側* の JS realm へ
#: 注入する。Streamlit は再実行のたびにコンポーネント iframe を破棄するため、
#: その中で定義したタイマーのコールバックは realm ごと消える一方でタイマー ID
#: だけが残る。これが HUD の停止した原因である。ページ側で動かせば、要素も
#: ドラッグハンドラもタイマーも再実行を越えて生き残る。
#:
#: H1: 折れ線グラフの系列（`window.__fslHudSeries`）も同じ理由でページ側の
#: グローバルに置く。Streamlit の完全な再実行（bridge の再設置）を越えて
#: 生き残らせる必要があり、かつ `localStorage` には一切書かない（§3.4:
#: 左右差の派生値のセッション間永続化の禁止に含まれる。HUD のドラッグ位置
#: `fslHudPos` とは異なり、この系列は禁止の対象内である）。
HUD_RUNTIME_JS: Final[str] = """
(function () {
  var CFG = window.__fslHudConfig;
  if (!CFG) return;

  // H1: 折れ線グラフのリングバッファ。ページ realm のグローバルに置き、
  // Streamlit の完全な再実行（bridge の再設置）を越えて生き残らせる。
  // localStorage には一切触れない。
  var series = window.__fslHudSeries;
  if (!series) {
    series = window.__fslHudSeries = {
      gen: null,
      buffers: { overall: [], contour: [], eyes: [], nose: [], mouth: [] }
    };
  }

  var sheet = document.getElementById('fsl-hud-style');
  if (!sheet) {
    sheet = document.createElement('style');
    sheet.id = 'fsl-hud-style';
    document.head.appendChild(sheet);
  }
  if (sheet.textContent !== CFG.css) sheet.textContent = CFG.css;

  var chrome = document.getElementById('fsl-chrome');
  if (!chrome) {
    chrome = document.createElement('div');
    chrome.id = 'fsl-chrome';
    chrome.innerHTML =
      '<span id="fsl-chrome-badge"></span>' +
      '<span id="fsl-chrome-hint"></span>';
    document.body.appendChild(chrome);
  }
  document.getElementById('fsl-chrome-badge').textContent = CFG.badgeText;
  document.getElementById('fsl-chrome-hint').textContent = CFG.hintText;
  chrome.hidden = !CFG.fullscreen;

  var hud = document.getElementById('fsl-hud');
  if (!hud) {
    hud = document.createElement('div');
    hud.id = 'fsl-hud';
    hud.hidden = true;
    // H1: 全体・スコア・部位別 4 つ・参考値の内容（H3 以降、Camera でこれらを
    // 表示するのはこの HUD だけである）と、折れ線グラフの骨格（svg 要素・
    // polyline 要素 5 本・軸ラベル・凡例）をここで 1 度だけ作る。5Hz の
    // tick（sync）はこの骨格を作り直さず、textContent と points 属性だけを
    // 書き換える。
    hud.innerHTML =
      // FR-040/§5.4: 状態と平滑化時間のチップ。数値より前に置く（下に並ぶ数字を
      // どう読むかの前提であるため）。中身は tick が textContent だけ書き換える。
      '<div id="fsl-hud-chips">' +
        '<span class="fsl-hud-chip" id="fsl-hud-state"></span>' +
        '<span class="fsl-hud-chip" id="fsl-hud-smoothing"></span>' +
        '<span class="fsl-hud-chip" id="fsl-hud-hold"></span>' +
      '</div>' +
      '<div id="fsl-hud-value"></div>' +
      '<div id="fsl-hud-label"></div>' +
      '<div class="fsl-hud-baseline" id="fsl-hud-baseline-symmetry"></div>' +
      // FR-043: ラベルは空のまま作り、直後（骨格を作った・作らないに関わらず
      // 常に実行される節）で CFG.* から textContent を書き込む。骨格へ日本語を
      // 直接焼き込むと、5Hz の tick とは別経路である言語切替が反映されない。
      '<div id="fsl-hud-score-row">' +
        '<span id="fsl-hud-score-label"></span>' +
        '<span id="fsl-hud-score-value"></span>' +
        '<span class="fsl-hud-baseline" id="fsl-hud-baseline-score"></span>' +
      '</div>' +
      '<div id="fsl-hud-parts">' +
        '<span class="fsl-hud-part">' +
          '<span><b id="fsl-hud-part-label-contour"></b>' +
            '<span id="fsl-hud-part-contour"></span></span>' +
          '<span class="fsl-hud-baseline" id="fsl-hud-baseline-contour"></span>' +
        '</span>' +
        '<span class="fsl-hud-part">' +
          '<span><b id="fsl-hud-part-label-eyes"></b>' +
            '<span id="fsl-hud-part-eyes"></span></span>' +
          '<span class="fsl-hud-baseline" id="fsl-hud-baseline-eyes"></span>' +
        '</span>' +
        '<span class="fsl-hud-part">' +
          '<span><b id="fsl-hud-part-label-nose"></b>' +
            '<span id="fsl-hud-part-nose"></span></span>' +
          '<span class="fsl-hud-baseline" id="fsl-hud-baseline-nose"></span>' +
        '</span>' +
        '<span class="fsl-hud-part">' +
          '<span><b id="fsl-hud-part-label-mouth"></b>' +
            '<span id="fsl-hud-part-mouth"></span></span>' +
          '<span class="fsl-hud-baseline" id="fsl-hud-baseline-mouth"></span>' +
        '</span>' +
      '</div>' +
      '<div id="fsl-hud-demo"></div>' +
      '<svg id="fsl-hud-chart" viewBox="0 0 ' + CFG.chartWidth + ' ' + CFG.chartHeight +
        '" preserveAspectRatio="none">' +
        '<text x="2" y="9" class="fsl-chart-axis">' + CFG.chartYMax + '%</text>' +
        '<text x="2" y="' + (CFG.chartHeight - 2) + '" class="fsl-chart-axis">' +
          CFG.chartYMin + '%</text>' +
        // H5: 面は線より **前** に置く。SVG は文書順で重なるため、後ろに置くと
        // 面が 5 本の線をすべて覆い隠す。
        '<polygon id="fsl-chart-overall-area" class="fsl-chart-area" points=""></polygon>' +
        '<polyline id="fsl-chart-overall" class="fsl-chart-line fsl-chart-overall" points="">' +
          '</polyline>' +
        '<polyline id="fsl-chart-contour" class="fsl-chart-line fsl-chart-contour" points="">' +
          '</polyline>' +
        '<polyline id="fsl-chart-eyes" class="fsl-chart-line fsl-chart-eyes" points="">' +
          '</polyline>' +
        '<polyline id="fsl-chart-nose" class="fsl-chart-line fsl-chart-nose" points="">' +
          '</polyline>' +
        '<polyline id="fsl-chart-mouth" class="fsl-chart-line fsl-chart-mouth" points="">' +
          '</polyline>' +
      '</svg>' +
      '<div id="fsl-hud-legend">' +
        '<span class="fsl-legend fsl-legend-overall" id="fsl-hud-legend-overall"></span>' +
        '<span class="fsl-legend fsl-legend-contour" id="fsl-hud-legend-contour"></span>' +
        '<span class="fsl-legend fsl-legend-eyes" id="fsl-hud-legend-eyes"></span>' +
        '<span class="fsl-legend fsl-legend-nose" id="fsl-hud-legend-nose"></span>' +
        '<span class="fsl-legend fsl-legend-mouth" id="fsl-hud-legend-mouth"></span>' +
      '</div>' +
      // grip を但し書きより先に置く。HUD は `pointer-events: none` かつ
      // `max-height: min(90vh, 720px)` なので、あふれた分は読むことも掴むことも
      // できない。長い但し書きを grip の前に置くと、縦に短い窓で grip が箱の外へ
      // 押し出され、HUD を退ける手段そのものが失われる。
      '<div id="fsl-hud-grip">⠿ <span id="fsl-hud-grip-label"></span></div>' +
      '<div id="fsl-hud-baseline-note"></div>';
    document.body.appendChild(hud);

    // ドラッグは grip だけで受ける。HUD の残りはクリックが透過し、下にある
    // 映像側の STOP ボタンが押せる状態を保つ。
    var grip = hud.querySelector('#fsl-hud-grip');
    var dragging = false, offsetX = 0, offsetY = 0;
    grip.addEventListener('pointerdown', function (event) {
      var box = hud.getBoundingClientRect();
      dragging = true;
      offsetX = event.clientX - box.left;
      offsetY = event.clientY - box.top;
      try { grip.setPointerCapture(event.pointerId); } catch (err) {}
      event.preventDefault();
    });
    grip.addEventListener('pointermove', function (event) {
      if (!dragging) return;
      // H1: HUD の内容量が増えた分だけ、画面外へ完全にドラッグし出せてしまう
      // 影響も大きい。左上だけでなく、HUD 自身の幅・高さぶんだけ右端・下端も
      // 制限する。
      var maxX = Math.max(0, window.innerWidth - hud.offsetWidth);
      var maxY = Math.max(0, window.innerHeight - hud.offsetHeight);
      var x = Math.min(maxX, Math.max(0, event.clientX - offsetX));
      var y = Math.min(maxY, Math.max(0, event.clientY - offsetY));
      hud.style.left = x + 'px';
      hud.style.top = y + 'px';
      hud.style.right = 'auto';
      hud.style.bottom = 'auto';
      try {
        localStorage.setItem('fslHudPos', JSON.stringify({x: x, y: y}));
      } catch (err) { /* プライベートモード: 位置を記憶しないだけで支障はない */ }
    });
    var stop = function (event) {
      if (!dragging) return;
      dragging = false;
      try { grip.releasePointerCapture(event.pointerId); } catch (err) {}
    };
    grip.addEventListener('pointerup', stop);
    grip.addEventListener('pointercancel', stop);
  }

  // FR-043: 骨格は 1 度しか作らないが、表示言語はいつでも切り替わりうる。
  // 骨格の有無に関わらず、bridge が呼ばれるたびに CFG の現在値で書き換える。
  document.getElementById('fsl-hud-score-label').textContent = CFG.scoreLabel;
  document.getElementById('fsl-hud-part-label-contour').textContent = CFG.partLabels.contour;
  document.getElementById('fsl-hud-part-label-eyes').textContent = CFG.partLabels.eyes;
  document.getElementById('fsl-hud-part-label-nose').textContent = CFG.partLabels.nose;
  document.getElementById('fsl-hud-part-label-mouth').textContent = CFG.partLabels.mouth;
  document.getElementById('fsl-hud-legend-overall').textContent = CFG.legendOverallLabel;
  document.getElementById('fsl-hud-legend-contour').textContent = CFG.partLabels.contour;
  document.getElementById('fsl-hud-legend-eyes').textContent = CFG.partLabels.eyes;
  document.getElementById('fsl-hud-legend-nose').textContent = CFG.partLabels.nose;
  document.getElementById('fsl-hud-legend-mouth').textContent = CFG.partLabels.mouth;
  document.getElementById('fsl-hud-grip-label').textContent = CFG.gripText;

  // プリセットを選び直すとドラッグ位置は破棄する。ラジオが既知の角へ戻る
  // 手段であり続けるようにするためである。
  var applyAnchor = function () { hud.setAttribute('style', CFG.anchor); };
  if (window.__fslHudPreset !== CFG.position) {
    window.__fslHudPreset = CFG.position;
    applyAnchor();
    try { localStorage.removeItem('fslHudPos'); } catch (err) {}
  } else {
    var saved = null;
    try { saved = JSON.parse(localStorage.getItem('fslHudPos')); } catch (err) {}
    if (saved && typeof saved.x === 'number') {
      hud.setAttribute('style', 'left:' + saved.x + 'px; top:' + saved.y + 'px;');
    } else {
      applyAnchor();
    }
  }

  document.body.classList.toggle('fsl-fullscreen', !!CFG.fullscreen);

  // タイマーは 1 本だけ。bridge が再実行されたときは積み増さず置き換える。
  if (window.__fslHudTimer) clearInterval(window.__fslHudTimer);
  // コンポーネント iframe は same-origin なので、その <video> と START/STOP の
  // 行へ手が届く。冪等なので毎 tick 実行しても安全である。
  var tameVideoFrame = function () {
    var host = document.querySelector('.st-key-camera_preview_frame iframe');
    var vdoc = null;
    try { vdoc = host && host.contentDocument; } catch (err) { return; }
    if (!vdoc || !vdoc.head) return;
    var vstyle = vdoc.getElementById('fsl-video-style');
    if (!vstyle) {
      vstyle = vdoc.createElement('style');
      vstyle.id = 'fsl-video-style';
      vdoc.head.appendChild(vstyle);
    }
    if (vstyle.textContent !== CFG.videoCss) vstyle.textContent = CFG.videoCss;
    vdoc.documentElement.classList.toggle('fsl-fs', !!CFG.fullscreen);
    // コントロール行が <video> の祖先になってはならない。そこへスタイルを
    // 当てると映像まで再配置されてしまう。映像を含まない最後のラッパーまで
    // しか遡らず、再生開始で木構造が変わって映像を含むようになった要素からは
    // class を外す。
    var stale = vdoc.querySelectorAll('.fsl-controls');
    for (var s = 0; s < stale.length; s++) {
      if (stale[s].querySelector('video')) stale[s].classList.remove('fsl-controls');
    }
    var anyButton = vdoc.querySelector('button');
    if (anyButton) {
      var row = null;
      for (var n = anyButton.parentElement; n && n !== vdoc.body; n = n.parentElement) {
        if (n.querySelector('video')) break;
        row = n;
      }
      if (row && !row.classList.contains('fsl-controls')) row.classList.add('fsl-controls');
    }
  };

  var lastGate = '';
  var PART_KEYS = ['contour', 'eyes', 'nose', 'mouth'];
  // 骨格は 1 度しか作らないので、tick で行うのは textContent の書き換えだけである。
  // 同じ文字列なら触らない（DOM を毎 tick 汚すと style 再計算が走る）。
  var setText = function (id, text) {
    var el = document.getElementById(id);
    if (el && el.textContent !== text) el.textContent = text;
  };
  // H1: 現在値 1 つをリングバッファの右端へ積み、時間軸を右詰めで描く座標へ
  // 変換する。窓（`CFG.chartYMin`〜`CFG.chartYMax`）の外は端へ clamp する。
  var pointsFor = function (buffer) {
    var n = buffer.length;
    if (n === 0) return '';
    var step = CFG.chartWidth / (CFG.chartCapacity - 1);
    var range = CFG.chartYMax - CFG.chartYMin;
    var coords = [];
    for (var i = 0; i < n; i++) {
      var x = CFG.chartWidth - (n - 1 - i) * step;
      var clamped = Math.min(CFG.chartYMax, Math.max(CFG.chartYMin, buffer[i]));
      var y = CFG.chartHeight - ((clamped - CFG.chartYMin) / range) * CFG.chartHeight;
      coords.push(x.toFixed(1) + ',' + y.toFixed(1));
    }
    return coords.join(' ');
  };
  // H5: 総合値の線の下を塗るための閉じた多角形。**線と同じ `pointsFor` の結果を
  // 使い、両端を下端へ落として閉じるだけ**にする。面だけ別の変換で描くと、線と面が
  // ずれて「どちらが値なのか」が読めなくなる。
  var areaPointsFor = function (buffer) {
    var line = pointsFor(buffer);
    if (!line) return '';
    var step = CFG.chartWidth / (CFG.chartCapacity - 1);
    var left = CFG.chartWidth - (buffer.length - 1) * step;
    return left.toFixed(1) + ',' + CFG.chartHeight + ' ' + line + ' ' +
      CFG.chartWidth.toFixed(1) + ',' + CFG.chartHeight;
  };

  var sync = function () {
    tameVideoFrame();
    var holder = document.getElementById(CFG.dataId);
    var hudEl = document.getElementById('fsl-hud');
    if (!hudEl) return;
    // payload が無いのはカメラが動いていないという意味である。payload を出す
    // パネルは再生中しか描画されないため、HUD も一緒に消す。直前のフレームの
    // 数値を残し続けてはならない。
    var hint = document.getElementById('fsl-chrome-hint');
    if (!holder) {
      hudEl.hidden = true;
      if (hint) hint.hidden = false;
      return;
    }
    if (hint) hint.hidden = true;
    var data;
    try { data = JSON.parse(holder.textContent); } catch (err) { return; }

    // H1/§3.4/§8.8: 世代識別子（segment_id 由来）が変わったら、次の値を積む前に
    // リングバッファを空にする。セッションの Reset・入力ソースの切替・指標に
    // 影響する設定変更のいずれでも segment_id は更新されるため、これらを
    // 跨いだ古い系列が画面に残らない。localStorage には一切触れない。
    if (series.gen !== data.gen) {
      series.gen = data.gen;
      series.buffers = { overall: [], contour: [], eyes: [], nose: [], mouth: [] };
    }

    hudEl.hidden = !data.visible;
    if (!data.visible) return;
    var valueEl = document.getElementById('fsl-hud-value');
    var labelEl = document.getElementById('fsl-hud-label');
    var demoEl = document.getElementById('fsl-hud-demo');
    var scoreEl = document.getElementById('fsl-hud-score-value');
    if (valueEl && valueEl.textContent !== data.symmetry) valueEl.textContent = data.symmetry;
    if (labelEl && labelEl.textContent !== data.label) labelEl.textContent = data.label;
    if (demoEl && demoEl.textContent !== data.demo) demoEl.textContent = data.demo;
    if (scoreEl && scoreEl.textContent !== data.score) scoreEl.textContent = data.score;
    for (var p = 0; p < PART_KEYS.length; p++) {
      var partName = PART_KEYS[p];
      var partEl = document.getElementById('fsl-hud-part-' + partName);
      var partText = data.parts ? data.parts[partName] : undefined;
      if (partEl && partText !== undefined && partEl.textContent !== partText) {
        partEl.textContent = partText;
      }
    }
    // FR-039: 参照平均の併記。payload が持たない・母集団が決まらないときは空文字で
    // あり、CSS の `:empty` が行ごと畳む。ここで `—` を入れない（併記の不在は測定の
    // 欠測ではない）。
    // FR-040/§5.4: 状態と平滑化時間。どちらも空文字なら `:empty` が畳む。
    // `—` を入れない（設定や状態の不在は測定の欠測ではない）。
    setText('fsl-hud-state', data.state || '');
    setText('fsl-hud-smoothing', data.smoothing || '');
    // FR-041: 数値が止まって見えるのが設定によるものか、測定が止まったのかを
    // 利用者が区別できるようにする。0 秒（毎回更新）のときは空文字で畳まれる。
    setText('fsl-hud-hold', data.hold || '');

    var baseline = data.baseline || {};
    var baselineParts = baseline.parts || {};
    setText('fsl-hud-baseline-symmetry', baseline.symmetry || '');
    setText('fsl-hud-baseline-score', baseline.score || '');
    setText('fsl-hud-baseline-note', baseline.note || '');
    for (var b = 0; b < PART_KEYS.length; b++) {
      setText('fsl-hud-baseline-' + PART_KEYS[b], baselineParts[PART_KEYS[b]] || '');
    }

    if (data.gate !== lastGate) {
      lastGate = data.gate;
      hudEl.style.borderColor = data.gate;
      var frame = document.querySelector('.st-key-camera_preview_frame');
      if (frame) frame.style.borderColor = data.gate;
    }

    // H1: 折れ線グラフ。5 系列とも「現在値を push → points 属性だけ書き換え」
    // で済ませ、骨格は作り直さない。欠測（null/undefined）の tick はその系列
    // へ追加しない（0 で埋めない、不変条件 11）。
    var chart = data.chart || {};
    var chartKeys = PART_KEYS.concat(['overall']);
    for (var c = 0; c < chartKeys.length; c++) {
      var key = chartKeys[c];
      var buffer = series.buffers[key];
      var value = chart[key];
      if (typeof value === 'number' && isFinite(value)) {
        buffer.push(value);
        if (buffer.length > CFG.chartCapacity) buffer.shift();
      }
      var poly = document.getElementById('fsl-chart-' + key);
      if (poly) poly.setAttribute('points', pointsFor(buffer));
      if (key === 'overall') {
        var area = document.getElementById('fsl-chart-overall-area');
        if (area) area.setAttribute('points', areaPointsFor(buffer));
      }
    }
  };
  sync();
  window.__fslHudTimer = setInterval(sync, CFG.intervalMs);
})();
"""
