from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_SECTIONS = {
    "dataset",
    "loader",
    "model",
    "training",
    "augmentation",
    "evaluation",
    "checkpoints",
    "logs",
}
SPLIT_NAMES = ("train", "validation", "test")


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("The configuration must be a YAML mapping.")

    missing = REQUIRED_SECTIONS.difference(config)
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    _validate_dataset(config["dataset"])
    return config


def _validate_dataset(dataset: dict[str, Any]) -> None:
    samples = dataset.get("samples")
    if not isinstance(samples, dict):
        raise ValueError("dataset.samples must define train, validation, and test counts.")
    for split in SPLIT_NAMES:
        if split not in samples:
            raise ValueError(f"dataset.samples.{split} is required.")
        real = int(samples[split]["real"])
        fake = int(samples[split]["fake"])
        if real <= 0 or fake <= 0:
            raise ValueError(f"dataset.samples.{split} counts must be positive.")
        if real != fake:
            raise ValueError(f"dataset.samples.{split} must remain class-balanced.")


def configured_sample_counts(dataset: dict[str, Any]) -> dict[str, dict[int, int]]:
    """Return split counts keyed by numeric target (0=real, 1=fake)."""
    return {
        split: {0: int(dataset["samples"][split]["real"]), 1: int(dataset["samples"][split]["fake"])}
        for split in SPLIT_NAMES
    }


def ensure_output_directories(config: dict[str, Any]) -> None:
    paths = [
        config["dataset"]["output_directory"],
        config["checkpoints"]["directory"],
        config["logs"]["directory"],
    ]
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
