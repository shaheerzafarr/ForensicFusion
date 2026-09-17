from __future__ import annotations

import numpy as np
import pandas as pd

from forensic_fusion.data import balanced_sample, make_exact_splits, validate_manifest


def make_frame(per_class: int = 100) -> pd.DataFrame:
    rows = []
    for target in (0, 1):
        for index in range(per_class):
            source = f"source_{index % 4}"
            rows.append(
                {
                    "sample_id": f"{target}_{index}",
                    "image_path": f"/images/{target}_{index}.jpg",
                    "target": target,
                    "source": source,
                }
            )
    return pd.DataFrame(rows)


def test_balanced_sample_is_exact_and_covers_sources() -> None:
    frame = pd.DataFrame(
        {
            "source": ["small"] * 5 + ["medium"] * 40 + ["large"] * 100,
            "value": np.arange(145),
        }
    )
    sampled = balanced_sample(frame, requested=75, seed=42)
    counts = sampled.groupby("source").size().to_dict()
    assert len(sampled) == 75
    assert counts == {"large": 35, "medium": 35, "small": 5}
    assert sampled["value"].is_unique


def test_exact_split_counts_and_generator_coverage() -> None:
    result = make_exact_splits(make_frame(), (0.7, 0.2, 0.1), seed=42)
    by_class_and_split = result.groupby(["target", "split"]).size().to_dict()
    assert by_class_and_split == {
        (0, "test"): 10,
        (0, "train"): 70,
        (0, "validation"): 20,
        (1, "test"): 10,
        (1, "train"): 70,
        (1, "validation"): 20,
    }
    fake_coverage = result.loc[result["target"] == 1].groupby("source")["split"].nunique()
    assert (fake_coverage == 3).all()


def test_explicit_counts_keep_200k_training_separate() -> None:
    result = make_exact_splits(
        make_frame(per_class=143),
        (100 / 143, 29 / 143, 14 / 143),
        seed=42,
        exact_totals=(100, 29, 14),
    )
    assert result.groupby("split").size().to_dict() == {
        "test": 28,
        "train": 200,
        "validation": 58,
    }


def test_manifest_validation() -> None:
    manifest = make_exact_splits(make_frame(), (0.7, 0.2, 0.1), seed=42)
    source_map = {source: index for index, source in enumerate(sorted(manifest["source"].unique()))}
    manifest["source_id"] = manifest["source"].map(source_map)
    config = {
        "dataset": {
            "samples": {
                "train": {"real": 70, "fake": 70},
                "validation": {"real": 20, "fake": 20},
                "test": {"real": 10, "fake": 10},
            }
        }
    }
    validate_manifest(manifest, config)
