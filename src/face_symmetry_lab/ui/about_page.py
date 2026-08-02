"""About page: methodology, constraints, models, and privacy (§4.1, FR-014).

Purely informational — no camera or image access happens here, so it is safe
to show regardless of consent state. Model asset lookups only read the
pinned manifest; they never download or touch runtime state (§10.6).
"""

from __future__ import annotations

import streamlit as st

from face_symmetry_lab.adapters.io.model_store import ModelStoreError
from face_symmetry_lab.ui.model_setup import (
    demographic_asset,
    landmark_asset,
    segmenter_asset,
)


def _render_model_versions() -> None:
    st.subheader("モデルとバージョン")
    try:
        landmark = landmark_asset()
        st.markdown(f"- **顔ランドマーク**: {landmark.version} · sha256 `{landmark.sha256[:16]}…`")
    except ModelStoreError:
        st.caption("顔ランドマークモデルのmanifestを読み込めません。")
    try:
        demographic = demographic_asset()
        st.markdown(
            f"- **参考年齢・参考性別**: {demographic.version} · sha256 `{demographic.sha256[:16]}…`"
        )
    except ModelStoreError:
        st.caption("年齢・性別モデルのmanifestを読み込めません。")
    try:
        segmenter = segmenter_asset()
        st.markdown(
            f"- **背景置換（表示専用）**: {segmenter.version} · sha256 `{segmenter.sha256[:16]}…`"
        )
    except ModelStoreError:
        st.caption("背景置換モデルのmanifestを読み込めません。")
    st.caption(
        "版とsha256は結果のエクスポートにも記録されます。"
        "詳細は docs/model-cards/ を参照してください。"
    )


def render_about_page() -> None:
    st.title("使い方とスコアの算出方法")
    st.caption("LOCAL PROCESSING · NO IMAGE STORAGE · RESEARCH USE")
    st.write(
        "Face Symmetry Lab は、Web カメラまたは静止画から顔のランドマークを端末内で解析し、"
        "左右差をわかりやすく可視化する研究・セルフチェック用ツールです。"
        "画像は外部へ送信・保存されず、同一人物・同一環境内での左右差と肌表面参考値の相対的な傾向、"
        "およびモデル推定の参考年齢・参考性別を確認できます。"
    )
    st.info(
        "試験中の指標です。感度係数は固定評価セットで未凍結のため、"
        "数値の絶対値ではなく同一条件での相対変化だけを参照してください。"
    )

    st.header("スコアの算出方法")

    st.subheader("幾何左右差スコア（4部位+全体）")
    st.markdown(
        "1. MediaPipe Face Landmarker が検出する 478 点のランドマークのうち、"
        "版管理された左右対応点だけを輪郭・目・鼻・口の 4 部位で使用します。\n"
        "2. 両眼の目頭・目尻の中点を結ぶ線が水平になるよう回転補正し、"
        "両眼の中点間距離を 1.0 として拡大縮小を正規化し、両眼中点を原点へ移動します"
        "（虹彩中心は視線で動くため基準にしません）。\n"
        "3. 眉間・鼻根・鼻尖・人中・上下唇中央・顎先から、Total Least Squares"
        "（直交距離を最小化する回帰）で顔の中心線を推定します。\n"
        "4. 右側の点を中心線に対して鏡映し、対応する左側の点とのユークリッド距離を"
        "部位ごとに RMS（二乗平均平方根）でまとめます。\n"
        "5. 各部位のRMS距離 `d` を `score = clamp(100 × exp(-k × d), 0, 100)` で 0〜100 へ変換し、"
        "全体は部位ごとの重み付き平均（初期値は各 0.25）です。\n"
        "6. 感度係数 `k` は固定評価セットでの校正が済むまで未凍結（試験中）です。"
        "数学上、対応点の距離がすべて 0 なら 100 になりますが、これは望ましさや優劣を意味せず、"
        "実写で 100 になることも保証しません。"
    )

    st.subheader("肌表面参考値（実験値）")
    st.markdown(
        "- 顔crop に対して、channel 平均をそろえる Gray-world ホワイトバランスと "
        "5〜95 percentile を用いた明度正規化を端末内で適用します"
        "（過補正を避ける範囲へ補正量をclamp）。\n"
        "- 補正後、版管理された左右の頬 landmark 周辺を抽出し、輝度・色差の分散から色調均一性を、"
        "局所勾配からテクスチャ均一性を算出します。\n"
        "- ソフトウェア補正はカメラ固有の露出・HDR・レンズ・ISP処理までは復元できないため、"
        "同一人物・同一端末・近い照明での相対比較の補助に限られます。"
        "肌質、疾患、清潔さ、健康、美的品質は表しません。幾何の左右差スコアには混合しません。"
    )

    st.subheader("参考年齢・参考性別")
    st.markdown(
        "- Open Model Zoo 2023.0 FP16 `age-gender-recognition-retail-0013` を、"
        "検証済みの公式 XML/BIN から再現可能な開発時変換で生成した固定 ONNX として "
        "`opencv-contrib-python` の DNN 機能で端末内推論します。\n"
        "- 入力は検出した単一顔の landmark bbox から作る BGR 62×62 の crop で、"
        "顔埋め込みは生成・保持しません。\n"
        "- 出力は連続値の年齢と `[female, male]` の 2 クラス確率で、対象は 18〜75 歳です。"
        "公開ベンチマークでは年齢平均誤差 6.99 年、性別accuracy 95.80% ですが、"
        "内部validation datasetの値であり、実利用時の精度を保証しません。\n"
        "- 年齢は「モデル推定の参考年齢」、性別は「外見からのモデル推定の参考性別」であり、"
        "本人の自己認識・法的性別・実年齢を表しません。左右差・肌表面参考値とは分離して扱います。"
    )

    st.subheader("主な誤差要因")
    st.markdown(
        "- 顔の向き、照明、影、レンズ歪み、ランドマーク検出誤差\n"
        "- 化粧、髭、眼鏡、頭部被覆\n"
        "- カメラの自動露出・ホワイトバランス・圧縮処理\n"
        "- 年齢・性別推定は、肌色、年齢層、照明条件によって精度が変わり得ます"
    )

    st.subheader("精度向上のために必要なこと")
    st.markdown(
        "- 固定評価セットでの感度係数 `k` の校正\n"
        "- 実写での姿勢角符号の確認\n"
        "- 3D 正規化との比較評価\n"
        "- 照明・ぼけを検知するゲートの追加\n"
        "- 同一端末での再撮影評価"
    )

    st.header("使用しないでください")
    st.markdown(
        "- 医療、本人確認、採用、保険、教育、監視、人物比較・人物評価への利用\n"
        "- 本人以外、または明示的な権限・同意のない人物の画像の使用\n"
        "- 専用の同意・保護設計なしの未成年者への利用\n"
        "- スコアの高低による感情的な祝福・警告、人物間比較、平均顔との比較、ランキング"
    )

    st.header("プライバシーとデータの扱い")
    st.markdown(
        "- 画像・映像・ランドマーク・年齢帯・左右差スコアは外部へ送信せず、暗黙に保存もしません。\n"
        "- カメラは `127.0.0.1` のみで待ち受け、STUN/TURN や利用統計は既定で無効です。\n"
        "- 画像アップロードやカメラ利用は、事前の同意画面を完了するまで開始しません。\n"
        "- モデルは自動取得・自動更新せず、版と SHA-256 を追跡してエクスポート結果に記録します。"
    )

    st.divider()
    _render_model_versions()
