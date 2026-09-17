from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .model import ForensicFusionModel


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    epochs = int(config["training"]["epochs"])
    warmup_steps = int(config["training"]["warmup_epochs"]) * steps_per_epoch
    total_steps = max(1, epochs * steps_per_epoch)

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    source_ids: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    smoothing = float(config["training"]["label_smoothing"])
    smoothed = targets * (1.0 - smoothing) + 0.5 * smoothing
    binary_loss = F.binary_cross_entropy_with_logits(outputs["logit"], smoothed)
    fake_mask = targets >= 0.5
    if fake_mask.any():
        source_loss = F.cross_entropy(outputs["source_logits"][fake_mask], source_ids[fake_mask])
    else:
        source_loss = outputs["source_logits"].sum() * 0.0
    total = binary_loss + float(config["training"]["generator_loss_weight"]) * source_loss
    return total, binary_loss.detach(), source_loss.detach()


def train_one_epoch(
    model: ForensicFusionModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    model.train()
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    clip = float(config["training"]["gradient_clip_norm"])
    amp = bool(config["training"]["mixed_precision"] and device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)
    totals = {"loss": 0.0, "binary_loss": 0.0, "source_loss": 0.0, "examples": 0}

    progress = tqdm(loader, desc=f"Train {epoch:02d}")
    for step, batch in enumerate(progress, start=1):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        source_ids = batch["source_id"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            outputs = model(images)
            loss, binary_loss, source_loss = _loss(outputs, targets, source_ids, config)
            scaled_loss = loss / accumulation
        scaler.scale(scaled_loss).backward()

        update = step % accumulation == 0 or step == len(loader)
        if update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        batch_size = images.shape[0]
        totals["loss"] += float(loss.detach()) * batch_size
        totals["binary_loss"] += float(binary_loss) * batch_size
        totals["source_loss"] += float(source_loss) * batch_size
        totals["examples"] += batch_size
        progress.set_postfix(loss=f"{totals['loss'] / totals['examples']:.4f}")

    return {key: value / max(1, totals["examples"]) for key, value in totals.items() if key != "examples"}


@torch.inference_mode()
def predict(
    model: ForensicFusionModel,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
    horizontal_flip_tta: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    sample_ids: list[str] = []
    paths: list[str] = []
    amp = bool(mixed_precision and device.type == "cuda")
    for batch in tqdm(loader, desc="Inference"):
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(images)["logit"]
            if horizontal_flip_tta:
                flipped_logits = model(torch.flip(images, dims=(-1,)))["logit"]
                batch_probabilities = (torch.sigmoid(logits) + torch.sigmoid(flipped_logits)) * 0.5
            else:
                batch_probabilities = torch.sigmoid(logits)
        probabilities.append(batch_probabilities.float().cpu().numpy())
        targets.append(batch["target"].numpy())
        sample_ids.extend(batch["sample_id"])
        paths.extend(batch["image_path"])
    return np.concatenate(targets), np.concatenate(probabilities), sample_ids, paths


def load_checkpoint(
    path: str | Path,
    model: ForensicFusionModel,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint
