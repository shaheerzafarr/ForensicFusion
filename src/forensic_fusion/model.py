from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class FixedResidualBank(nn.Module):
    """Three fixed high-pass filters followed by a small learned encoder."""

    def __init__(self, output_features: int) -> None:
        super().__init__()
        kernels = torch.tensor(
            [
                [[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, -1, 4, -1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        kernels[2] /= 12.0
        self.register_buffer("kernels", kernels)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, output_features, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_features),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

    def forward(self, normalised_rgb: torch.Tensor) -> torch.Tensor:
        rgb = normalised_rgb * self.std + self.mean
        gray = 0.2989 * rgb[:, :1] + 0.5870 * rgb[:, 1:2] + 0.1140 * rgb[:, 2:3]
        residuals = F.conv2d(F.pad(gray, (2, 2, 2, 2), mode="reflect"), self.kernels)
        residuals = torch.tanh(residuals * 2.0)
        return self.encoder(residuals)


class ForensicFusionModel(nn.Module):
    def __init__(self, config: dict[str, Any], num_sources: int, pretrained: bool | None = None) -> None:
        super().__init__()
        model_config = config["model"]
        if model_config["backbone"] != "convnext_tiny":
            raise ValueError("This clean pipeline currently supports model.backbone=convnext_tiny.")
        use_pretrained = bool(model_config["pretrained"] if pretrained is None else pretrained)
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if use_pretrained else None
        backbone = convnext_tiny(weights=weights)
        self.backbone_features = backbone.features
        self.backbone_pool = backbone.avgpool
        self.backbone_norm = backbone.classifier[0]
        semantic_features = int(backbone.classifier[2].in_features)

        residual_features = int(model_config["residual_features"])
        fusion_features = int(model_config["fusion_features"])
        dropout = float(model_config["dropout"])
        self.residual_stream = FixedResidualBank(residual_features)
        self.fusion = nn.Sequential(
            nn.Linear(semantic_features + residual_features, fusion_features),
            nn.LayerNorm(fusion_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(fusion_features, 1)
        self.source_head = nn.Linear(fusion_features, num_sources)
        self._initialise_heads()

    def _initialise_heads(self) -> None:
        for module in (*self.residual_stream.encoder.modules(), *self.fusion.modules(), self.binary_head, self.source_head):
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def semantic_features(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone_features(image)
        features = self.backbone_norm(self.backbone_pool(features))
        return torch.flatten(features, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        semantic = self.semantic_features(image)
        residual = self.residual_stream(image)
        fused = self.fusion(torch.cat((semantic, residual), dim=1))
        return {
            "logit": self.binary_head(fused).squeeze(1),
            "source_logits": self.source_head(fused),
        }

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone_features.parameters():
            parameter.requires_grad = trainable
        for parameter in self.backbone_norm.parameters():
            parameter.requires_grad = trainable

    def backbone_parameters(self):
        yield from self.backbone_features.parameters()
        yield from self.backbone_norm.parameters()

    def head_parameters(self):
        backbone_ids = {id(parameter) for parameter in self.backbone_parameters()}
        yield from (parameter for parameter in self.parameters() if id(parameter) not in backbone_ids)


def build_optimizer(model: ForensicFusionModel, config: dict[str, Any]) -> torch.optim.Optimizer:
    training = config["training"]
    return torch.optim.AdamW(
        [
            {"params": list(model.backbone_parameters()), "lr": float(training["backbone_learning_rate"])},
            {"params": list(model.head_parameters()), "lr": float(training["head_learning_rate"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
