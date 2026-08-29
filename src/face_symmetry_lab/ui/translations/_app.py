"""英語の訳表: `ui/app.py`（FR-043）。"""

from __future__ import annotations

from typing import Final

EN: Final[dict[str, str]] = {
    "参照統計": "Reference statistics",
    "使い方・算出方法": "How it works",
    "入力方法": "Input method",
    "利用前に確認してください": "Please confirm before use",
    "映像と画像は端末内のメモリだけで処理され、アプリは保存・外部送信しません。": (
        "Video and images are processed only in the device's memory; "
        "the app does not save or send them externally."
    ),
    "私は18〜75歳です。結果は参考値であり、実年齢、自己認識・法的性別、"
    "肌質・医学的状態、人物の価値を示さないことに同意します。": (
        "I am between 18 and 75 years old. I agree that the results are reference "
        "values only and do not indicate actual age, self-identified or legal "
        "gender, skin condition or medical status, or a person's worth."
    ),
    "本人の映像・画像、または分析する明示的な権限・同意があるものだけを使用します。": (
        "I will only use video or images of myself, or material I have explicit "
        "authorisation or consent to analyse."
    ),
    "3 項目を確認すると、カメラまたは画像を選択できます。": (
        "Once you check all three items, you can choose the camera or an image."
    ),
    "Webカメラ": "Webcam",
    "画像アップロード": "Image upload",
    "映像を見ながら、顔の向きや距離をその場で調整できます。"
    "数値は最大 8 Hz で更新され、映像は端末内のメモリだけで処理されます。": (
        "Watch the live video and adjust your face's angle and distance on the "
        "spot. Values update at up to 8 Hz, and video is processed only in the "
        "device's memory."
    ),
    "JPEG / PNG を 1 枚アップロードして分析します。"
    "10MB・12MP 以下、EXIF の向きを適用し、平滑化のない 1 回きりの結果を表示します。": (
        "Upload a single JPEG or PNG image for analysis. Up to 10MB and 12MP, "
        "EXIF orientation is applied, and a one-off result without smoothing is shown."
    ),
}
