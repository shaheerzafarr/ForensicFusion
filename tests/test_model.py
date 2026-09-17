from __future__ import annotations

import torch

from forensic_fusion.model import ForensicFusionModel


def test_model_output_shapes() -> None:
    config = {
        "model": {
            "backbone": "convnext_tiny",
            "pretrained": False,
            "residual_features": 32,
            "fusion_features": 64,
            "dropout": 0.1,
        }
    }
    model = ForensicFusionModel(config, num_sources=7, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64))
    assert output["logit"].shape == (1,)
    assert output["source_logits"].shape == (1, 7)
