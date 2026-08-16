# data/ — ローカル専用のデータ置き場

**このディレクトリの中身は追跡しない**（`.gitignore` の `data/*`、例外はこの README だけ）。
不変条件 1（`requirement.md` §1.3.1）により、顔の画像・映像・ランドマーク座標・個票を
repository へ入れないため。

## 構成

```text
data/
├── takes/     診断用の実写テイク（*.mov）と、その測定出力（take*_*.json）
└── utkface/   UTKFace notebook（research/utkface-symmetry/）の出力
```

### data/takes/

`scripts/measure_metric_noise.py` / `measure_pose_response.py` /
`measure_session_variation.py` / `measure_zoom_depth_scale.py` の `--video` 入力。

```bash
uv run python scripts/measure_metric_noise.py --video data/takes/take01.mov --max-frames 600
```

撮影条件は `docs/capture-protocol.md`、測定結果は `docs/symmetry-metric-measurements.md`。

### data/utkface/

| ファイル | 内容 | 扱い |
|---|---|---|
| `utkface_symmetry3d_records.csv` | 個票 14,754 行（ファイル名を含む） | **絶対にコミットしない** |
| `utkface_symmetry3d_group_means.csv` | 1 軸ごとの集計 | 検証用 |
| `utkface_symmetry3d_measurement_stats.csv` | 全体統計 | 検証用 |
| `utkface_symmetry3d_summary.json` | 実行条件・件数・警告文 | 検証用 |

UTKFace は**非商用研究限定**（https://susanqq.github.io/UTKFace/）。

配布アセットは `scripts/build_reference_dataset.py` がここから生成し、
`configs/reference/` へ出力する。**アプリが読むのは `configs/reference/` だけで、
`data/` を実行時に参照しない。**
