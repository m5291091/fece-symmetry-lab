#!/usr/bin/env python3
"""Export the seminar MobileNetV2 age classifier from PyTorch to ONNX.

Run this script in a separate training environment containing torch and
torchvision. Those packages intentionally do not belong to the application lock.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="best_age_model_mobilenetv2.pth")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/age_mobilenetv2_utkface_v1.onnx"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        import torch
        from torch import nn
        from torchvision import models
    except ImportError as error:
        raise SystemExit("torch と torchvision を含む学習環境で実行してください。") from error

    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint が見つかりません: {args.checkpoint}")

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
    state_dict: Any = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    torch.onnx.export(
        model,
        sample,
        args.output,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    print(f"exported: {args.output}")
    print(f"sha256: {_sha256(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
