from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import pandas as pd
import torch

from forensic_fusion.config import ensure_output_directories, load_config
from forensic_fusion.data import make_loader, prepare_manifest
from forensic_fusion.engine import load_checkpoint, make_scheduler, predict, train_one_epoch
from forensic_fusion.metrics import binary_metrics, choose_threshold
from forensic_fusion.model import ForensicFusionModel, build_optimizer
from forensic_fusion.utils import atomic_torch_save, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ForensicFusion on the prepared 200k manifest.")
    parser.add_argument("--config", default="config_kaggle.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints.latest.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download/load ImageNet weights.")
    return parser.parse_args()


def append_history(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([row])
    if path.is_file():
        history = pd.read_csv(path)
        history = history.loc[history["epoch"] != row["epoch"]]
        new_row = pd.concat((history, new_row), ignore_index=True)
    new_row.sort_values("epoch").to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_directories(config)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest = prepare_manifest(config)
    train_frame = manifest.loc[manifest["split"] == "train"].reset_index(drop=True)
    validation_frame = manifest.loc[manifest["split"] == "validation"].reset_index(drop=True)
    print(f"Train: {len(train_frame):,} | Validation: {len(validation_frame):,} | Test held out: {(manifest['split'] == 'test').sum():,}")

    num_sources = int(manifest["source_id"].max()) + 1
    # A resume immediately overwrites all weights, so avoid a needless download.
    pretrained_override = False if (args.no_pretrained or args.resume) else None
    model = ForensicFusionModel(config, num_sources, pretrained=pretrained_override).to(device)
    optimizer = build_optimizer(model, config)
    initial_loader = make_loader(train_frame, config, training=True, epoch=1)
    scheduler = make_scheduler(optimizer, config, len(initial_loader))
    amp_enabled = bool(config["training"]["mixed_precision"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_score = float("-inf")
    epochs_without_improvement = 0
    latest_path = Path(config["checkpoints"]["latest"])
    if args.resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Cannot resume; checkpoint not found: {latest_path}")
        checkpoint = load_checkpoint(latest_path, model, optimizer, scheduler, scaler, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint.get("best_score", best_score))
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        print(f"Resuming at epoch {start_epoch}; best score={best_score:.6f}")

    validation_loader = make_loader(validation_frame, config, training=False)
    epochs = int(config["training"]["epochs"])
    freeze_epochs = int(config["training"]["freeze_backbone_epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    minimum_improvement = float(config["training"]["minimum_improvement"])
    selection_metric = str(config["training"]["selection_metric"])

    for epoch in range(start_epoch, epochs + 1):
        model.set_backbone_trainable(epoch > freeze_epochs)
        train_loader = initial_loader if epoch == 1 and start_epoch == 1 else make_loader(train_frame, config, training=True, epoch=epoch)
        losses = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, config, epoch)
        targets, probabilities, _, _ = predict(
            model,
            validation_loader,
            device,
            mixed_precision=bool(config["training"]["mixed_precision"]),
        )
        threshold = choose_threshold(targets, probabilities, str(config["evaluation"]["threshold_metric"]))
        metrics = binary_metrics(targets, probabilities, threshold)
        score = float(metrics[selection_metric])
        improved = score > best_score + minimum_improvement
        if improved:
            best_score = score
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d} | loss={losses['loss']:.5f} | accuracy={metrics['accuracy']:.5f} | "
            f"precision={metrics['precision']:.5f} | recall={metrics['recall']:.5f} | "
            f"F1={metrics['f1']:.5f} | PR-AUC={metrics['pr_auc']:.5f} | threshold={threshold:.4f}"
        )
        append_history(
            Path(config["logs"]["history_file"]),
            {"epoch": epoch, **losses, **metrics, "selection_score": score},
        )
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "epochs_without_improvement": epochs_without_improvement,
            "threshold": threshold,
            "validation_metrics": metrics,
            "num_sources": num_sources,
            "config": config,
        }
        atomic_torch_save(payload, latest_path)
        if improved:
            atomic_torch_save(payload, config["checkpoints"]["best"])
            print(f"Saved new best checkpoint ({selection_metric}={score:.6f}).")
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {patience} epochs without material improvement.")
            break

    print("Training complete. Run evaluate.py once for the untouched test result.")


if __name__ == "__main__":
    main()
