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
from forensic_fusion.engine import load_checkpoint, predict
from forensic_fusion.metrics import binary_metrics
from forensic_fusion.model import ForensicFusionModel
from forensic_fusion.utils import seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the frozen best checkpoint on the untouched test split.")
    parser.add_argument("--config", default="config_kaggle.yaml")
    parser.add_argument("--checkpoint", default=None, help="Defaults to checkpoints.best from the config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_directories(config)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = prepare_manifest(config)
    test_frame = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    num_sources = int(manifest["source_id"].max()) + 1
    checkpoint_path = Path(args.checkpoint or config["checkpoints"]["best"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")

    model = ForensicFusionModel(config, num_sources, pretrained=False).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, device=device)
    if int(checkpoint.get("num_sources", num_sources)) != num_sources:
        raise ValueError("The checkpoint source mapping does not match the current manifest.")
    threshold = float(checkpoint["threshold"])
    loader = make_loader(test_frame, config, training=False)
    targets, probabilities, sample_ids, paths = predict(
        model,
        loader,
        device,
        mixed_precision=bool(config["training"]["mixed_precision"]),
        horizontal_flip_tta=bool(config["evaluation"]["test_time_horizontal_flip"]),
    )
    metrics = binary_metrics(targets, probabilities, threshold)
    predictions = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "image_path": paths,
            "target": targets.astype(int),
            "fake_probability": probabilities,
            "prediction": (probabilities >= threshold).astype(int),
        }
    ).merge(test_frame[["sample_id", "source"]], on="sample_id", how="left", validate="one_to_one")
    predictions.to_csv(config["logs"]["test_predictions_file"], index=False)

    per_source = {}
    for source, group in predictions.groupby("source", sort=True):
        per_source[str(source)] = binary_metrics(
            group["target"].to_numpy(), group["fake_probability"].to_numpy(), threshold
        )
    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "test_examples": len(predictions),
        "overall": metrics,
        "per_source": per_source,
    }
    write_json(report, config["logs"]["test_metrics_file"])
    print("Final untouched-test metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
