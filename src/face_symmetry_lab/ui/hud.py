"""映像へ重ねる HUD（§14.3e）の資産と組み立て。

`ui.camera` から分離した理由は、HUD がブラウザ側の資産（CSS・JS）と、その
資産へ値を渡すだけの薄い Python 関数で構成されており、カメラ画面の合成処理
とは変更理由が異なるためである。ここには Streamlit への描画と文字列生成しか
なく、幾何計算も推論も持たない。

値の整形は必ず `ui.scores` の共通フォーマッタを通す。HUD がパネルとは別の
表示規則へ分岐すると、同じ測定値が画面上で二通りに見えてしまう。
"""

from __future__ import annotations

import json
from typing import Final

import streamlit as st
import streamlit.components.v1 as components

from face_symmetry_lab.application.realtime import PipelineSnapshot
from face_symmetry_lab.domain.models import DemographicEstimate
from face_symmetry_lab.ui.scores import GENDER_LABELS, format_symmetry, gate_color

#: §14.3e/FR-032: HUD はパネル配置と同じくプリセットで移動し、ドラッグ位置は
#: 補助的に保持するだけである。"非表示" は何も描画しない。
HUD_HIDDEN: Final[str] = "非表示"
HUD_POSITION_OPTIONS: Final[tuple[str, ...]] = ("左上", "右上", "左下", "右下", HUD_HIDDEN)
HUD_POSITION_STATE_KEY = "camera_hud_position"

#: 各プリセットが HUD を位置指定コンテキスト内のどこへ固定するか。
HUD_PLACEMENTS: Final[dict[str, str]] = {
    "左上": "top:24px; left:24px;",
    "右上": "top:24px; right:24px;",
    "左下": "bottom:24px; left:24px;",
    "右下": "bottom:24px; right:24px;",
}

#: §14.3e: 全画面帯の文言。ページの装飾を畳んでいる間に表示される唯一の文章で
#: あり、現在の状態と戻り方の両方を担う。テストで文言を検証できるよう定数に
#: 切り出している。
FULLSCREEN_BADGE: Final[str] = "全画面表示中 — 右上の「終了」で戻ります"
FULLSCREEN_HINT: Final[str] = (
    "画面下の「START」を押すとカメラが始まります。"
    "ブラウザの F11 と併用すると、タブやアドレスバーも隠れた完全な全画面になります。"
)

#: 5 Hz の fragment が書き込み、bridge script が読み取る隠し要素。値は JSON
#: テキストとして渡す。Streamlit は markdown 内の HTML をサニタイズするため、
#: そのまま生き残ることが保証されるのはテキスト内容だけである。
HUD_DATA_ELEMENT_ID: Final[str] = "fsl-hud-data"

#: bridge 側の DOM だけで完結する更新間隔。パネルと同じ周期だが、サーバーへの
#: 往復が無く、隠し要素を読み直すだけで済む。
HUD_REFRESH_MS: Final[int] = 200

#: §5.2 の例外（§14.3e）: stylesheet は bridge が **一度だけ** 設置し、以後は
#: class の切り替えだけで制御する。5 Hz で `<style>` を送り直すと毎秒 5 回の
#: 全文書スタイル再計算が発生し、それが映像のカクつきの原因だった。
#: `backdrop-filter` も同様で、HUD 背後の領域を *映像フレームごとに* 再ぼかし
#: するため、どちらも現在は使用していない。
#:
#: 映像は `st.container(key=...)` が生成する `.st-key-camera_preview_frame
#: iframe` から辿る。Streamlit 内部の `data-testid` に依存しないため、
#: 以前に全画面表示が黙って壊れた原因を避けられる。
HUD_STYLESHEET: Final[str] = """
#fsl-hud {
    position: fixed; z-index: 1000050;
    background: rgba(0,0,0,.72); color: #fff;
    padding: 16px 22px; border-radius: 16px;
    border: 2px solid #6B6B6B;
    pointer-events: none;
    font-variant-numeric: tabular-nums;
    font-family: inherit;
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
    padding: 6px 16px; font-size: .8rem; letter-spacing: .04em;
}
#fsl-chrome-hint {
    background: rgba(0,0,0,.72); color: #fff;
    border-radius: 12px; padding: 10px 18px; font-size: 1rem; max-width: 70vw;
}
#fsl-chrome-hint[hidden] { display: none; }
#fsl-hud-value { font-size: 3.2rem; font-weight: 700; line-height: 1; }
#fsl-hud-label { font-size: .85rem; opacity: .75; margin-bottom: 10px; }
#fsl-hud-demo { font-size: 1.1rem; }
#fsl-hud-demo:empty { display: none; }
/* pointer events を受け取るのは grip だけにして、HUD の下にある映像側の STOP
   ボタンが他の場所ではそのまま押せる状態を保つ。 */
#fsl-hud-grip {
    pointer-events: auto; cursor: move; touch-action: none;
    margin-top: 10px; font-size: .75rem; opacity: .55;
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
HUD_RUNTIME_JS: Final[str] = """
(function () {
  var CFG = window.__fslHudConfig;
  if (!CFG) return;

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
    hud.innerHTML =
      '<div id="fsl-hud-value"></div>' +
      '<div id="fsl-hud-label"></div>' +
      '<div id="fsl-hud-demo"></div>' +
      '<div id="fsl-hud-grip">⠿ ドラッグで移動</div>';
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
      var x = Math.max(0, event.clientX - offsetX);
      var y = Math.max(0, event.clientY - offsetY);
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
    hudEl.hidden = !data.visible;
    if (!data.visible) return;
    var valueEl = document.getElementById('fsl-hud-value');
    var labelEl = document.getElementById('fsl-hud-label');
    var demoEl = document.getElementById('fsl-hud-demo');
    if (valueEl && valueEl.textContent !== data.symmetry) valueEl.textContent = data.symmetry;
    if (labelEl && labelEl.textContent !== data.label) labelEl.textContent = data.label;
    if (demoEl && demoEl.textContent !== data.demo) demoEl.textContent = data.demo;
    if (data.gate !== lastGate) {
      lastGate = data.gate;
      hudEl.style.borderColor = data.gate;
      var frame = document.querySelector('.st-key-camera_preview_frame');
      if (frame) frame.style.borderColor = data.gate;
    }
  };
  sync();
  window.__fslHudTimer = setInterval(sync, CFG.intervalMs);
})();
"""


def build_hud_payload(
    *,
    overall: float | None,
    estimate: DemographicEstimate | None,
    show_demographics: bool,
    gate: str,
    position: str,
) -> dict[str, object]:
    """§14.3e の HUD が表示する値。純粋関数で、Streamlit も markup も副作用も持たない。

    数値はすべてパネルと同じフォーマッタ（`format_symmetry`、`GENDER_LABELS`）
    を通す。HUD が同じ測定値に対する 2 つ目の表示規則へ分岐しないようにする
    ためである。欠測は古い値や 0 ではなく `—` と表示する（NFR-013）。

    ``demo`` は推定が無いとき、または利用者が年齢・性別パネルを切ったときに
    `—` ではなく空文字にする。空行は「機能が動いていて失敗している」と
    受け取られてしまうためである（FR-024）。
    """
    visible = position != HUD_HIDDEN and position in HUD_PLACEMENTS
    demo = ""
    if show_demographics and estimate is not None:
        age = f"{estimate.estimated_age_years:.0f} 歳"
        # `DemographicEstimate` は `gender` が `gender_probabilities` の argmax
        # であることを検証済みなので、ここで導出し直さずラベルへ変換する。
        gender = GENDER_LABELS.get(estimate.gender, "—")
        demo = f"参考年齢 {age} · 参考性別 {gender}"
    return {
        "visible": visible,
        "symmetry": format_symmetry(overall),
        "label": "対称度（実験値）",
        "demo": demo,
        "gate": gate,
    }


def hud_html(
    *,
    overall: float | None,
    estimate: DemographicEstimate | None,
    show_demographics: bool,
    gate: str,
    position: str,
) -> str:
    """payload を bridge が読む隠し要素へ直列化する。

    意図的に小さく、装飾も持たせない。5 Hz で通信を越えるのはこれだけであり、
    以前の版——装飾済みのブロックと `<style>` タグ——が映像のカクつきの原因
    だった。再描画のたびにブラウザが全文書のスタイル再計算を強いられていた。
    """
    payload = build_hud_payload(
        overall=overall,
        estimate=estimate,
        show_demographics=show_demographics,
        gate=gate,
        position=position,
    )
    encoded = json.dumps(payload, ensure_ascii=False)
    return f'<div id="{HUD_DATA_ELEMENT_ID}" style="display:none">{encoded}</div>'


def render_hud_payload(snapshot: PipelineSnapshot, *, show_demographics: bool) -> None:
    """1 つの snapshot 分の HUD 値を publish する（5 Hz の見出し fragment から呼ぶ）。

    隠し要素であり id で参照されるため、DOM 上のどこへ出ても構わない。これが
    3 つ目の fragment を新設せずに既存の fragment へ相乗りできる理由である。
    """
    analysis = snapshot.analysis
    overall = analysis.overall_distance_smooth if analysis is not None else None
    if overall is None and analysis is not None and analysis.symmetry is not None:
        overall = analysis.symmetry.overall

    st.markdown(
        hud_html(
            overall=overall,
            estimate=snapshot.demographic_estimate,
            show_demographics=show_demographics,
            gate=gate_color(analysis.state if analysis is not None else None),
            position=st.session_state.get(HUD_POSITION_STATE_KEY, HUD_POSITION_OPTIONS[0]),
        ),
        unsafe_allow_html=True,
    )


def hud_bridge_script(*, fullscreen: bool, position: str) -> str:
    """`HUD_RUNTIME_JS` をページへ設置するローダー。

    `components.v1.html` の iframe 内で動く。この iframe はページと same-origin
    なので `window.parent` へ到達できる。そこで行う仕事は最小限に留める。設定を
    publish し、実行時コード本体は `<script>` として親の realm へ渡すだけである。
    iframe 内で定義したものは、次の再実行で Streamlit がそれを差し替えた瞬間に
    動かなくなるためである。

    HUD をページの `<body>` へ付けることが、全画面表示で生き残る条件でもある。
    `<body>` 直下なら Streamlit のどのコンテナにもクリップされず、包含ブロック
    にもされないため、`position: fixed` が期待どおりに振る舞う。
    """
    config = json.dumps(
        {
            "fullscreen": fullscreen,
            "position": position,
            "anchor": HUD_PLACEMENTS.get(position, ""),
            "dataId": HUD_DATA_ELEMENT_ID,
            "css": HUD_STYLESHEET,
            "videoCss": VIDEO_CHROME_CSS,
            "intervalMs": HUD_REFRESH_MS,
            "badgeText": FULLSCREEN_BADGE,
            "hintText": FULLSCREEN_HINT,
        },
        ensure_ascii=False,
    )
    runtime = json.dumps(HUD_RUNTIME_JS)
    return f"""
<script>
(function () {{
  var win = window.parent;
  var doc = win && win.document;
  if (!doc || !doc.body) return;
  win.__fslHudConfig = {config};
  var previous = doc.getElementById('fsl-hud-runtime');
  if (previous) previous.remove();
  var runtime = doc.createElement('script');
  runtime.id = 'fsl-hud-runtime';
  runtime.textContent = {runtime};
  doc.head.appendChild(runtime);
}})();
</script>
"""


def render_hud_bridge(*, fullscreen: bool, position: str) -> None:
    """HUD bridge を 1 回の実行につき 1 度だけ設置する（5 Hz の fragment 内では呼ばない）。"""
    components.html(hud_bridge_script(fullscreen=fullscreen, position=position), height=0)
