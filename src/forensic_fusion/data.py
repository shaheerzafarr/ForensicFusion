from __future__ import annotations

import hashlib
import json
import math
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from .config import configured_sample_counts
from .utils import seed_worker, write_json


IMAGE_COLUMNS = ("image_path", "image", "path", "filepath", "file_path", "filename", "file_name", "img", "img_path")
LABEL_COLUMNS = ("target", "label", "class", "is_fake", "fake", "real")
SOURCE_COLUMNS = ("generator", "generator_name", "model", "model_name", "source", "method", "architecture")
REAL_LABELS = {"real", "original", "authentic", "natural", "human", "false", "0"}
FAKE_LABELS = {"fake", "synthetic", "generated", "ai", "ai-generated", "ai_generated", "deepfake", "true", "1", "artifact"}


def find_dataset_root(configured_root: str | Path, pattern: str) -> Path:
    configured = Path(configured_root)
    candidates = [configured]
    if str(configured).startswith("/kaggle/input"):
        candidates.extend([Path("/kaggle/input") / configured.name, Path("/kaggle/input")])
    candidates.extend([Path("dataset/data"), Path("dataset"), Path("data")])

    checked: set[str] = set()
    for candidate in candidates:
        key = str(candidate.absolute())
        if key in checked or not candidate.is_dir():
            continue
        checked.add(key)
        if next(candidate.glob(pattern), None) is not None:
            return candidate
    raise FileNotFoundError(
        f"No metadata files matching {pattern!r} were found. Checked: {sorted(checked)}"
    )


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    lookup = {str(column).strip().casefold(): str(column) for column in columns}
    return next((lookup[name.casefold()] for name in candidates if name.casefold() in lookup), None)


def _normalise_labels(values: pd.Series, paths: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    numeric = pd.to_numeric(values, errors="coerce")
    result.loc[numeric == 0] = 0
    result.loc[numeric > 0] = 1
    text = values.astype(str).str.strip().str.casefold()
    result.loc[text.isin(REAL_LABELS)] = 0
    result.loc[text.isin(FAKE_LABELS)] = 1

    unresolved = result.isna()
    if unresolved.any():
        lower_paths = paths.loc[unresolved].astype(str).str.casefold()
        fake = lower_paths.str.contains(
            r"fake|synthetic|generated|\bgan\b|diffusion|stylegan|deepfake|inpaint|glide|ddpm",
            regex=True,
        )
        real = lower_paths.str.contains(r"real|original|authentic|natural", regex=True)
        result.loc[lower_paths.index[fake]] = 1
        result.loc[lower_paths.index[~fake & real]] = 0
    return result


def _normalise_raw_path(value: Any) -> str:
    text = str(value).strip()
    if not text or text.casefold() == "nan":
        return ""
    return text.replace("\\", os.sep).replace("/", os.sep)


def _choose_path_builder(raw_paths: pd.Series, metadata_file: Path, root: Path) -> Callable[[str], str]:
    parent = metadata_file.parent
    folder = parent.name

    def direct(raw: str) -> Path:
        return Path(raw) if os.path.isabs(raw) else parent / raw

    def from_root(raw: str) -> Path:
        return Path(raw) if os.path.isabs(raw) else root / raw

    def strip_folder(raw: str) -> Path:
        parts = Path(raw).parts
        if parts and parts[0].casefold() == folder.casefold():
            return parent.joinpath(*parts[1:])
        return parent / raw

    builders = [direct, strip_folder, from_root]
    examples = [_normalise_raw_path(value) for value in raw_paths.dropna().head(64)]
    examples = [value for value in examples if value]
    scores = [sum(builder(value).is_file() for value in examples) for builder in builders]
    selected = builders[int(np.argmax(scores))]
    return lambda raw: str(selected(raw))


def read_metadata(root: Path, pattern: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    metadata_files = sorted(root.glob(pattern))
    if not metadata_files:
        raise RuntimeError(f"No metadata files found below {root}")

    frames: list[pd.DataFrame] = []
    reports: list[dict[str, Any]] = []
    print(f"Found {len(metadata_files)} metadata files below {root}")

    for metadata_file in metadata_files:
        header = pd.read_csv(metadata_file, nrows=0)
        image_column = _find_column(header.columns, IMAGE_COLUMNS)
        label_column = _find_column(header.columns, LABEL_COLUMNS)
        source_column = _find_column(header.columns, SOURCE_COLUMNS)
        if image_column is None:
            reports.append({"metadata_file": str(metadata_file), "error": "missing image column"})
            continue

        use_columns = list(dict.fromkeys(column for column in (image_column, label_column, source_column) if column))
        row_offset = 0
        valid_rows = 0
        unresolved_rows = 0
        for chunk in pd.read_csv(metadata_file, usecols=use_columns, chunksize=250_000, low_memory=False):
            raw = chunk[image_column].map(_normalise_raw_path)
            builder = _choose_path_builder(raw, metadata_file, root)
            paths = raw.map(builder)
            labels = _normalise_labels(chunk[label_column], paths) if label_column else _normalise_labels(pd.Series("", index=chunk.index), paths)
            valid = raw.ne("") & labels.notna()
            unresolved_rows += int((raw.ne("") & labels.isna()).sum())

            if source_column:
                sources = chunk[source_column].fillna(metadata_file.parent.name).astype(str).str.strip()
                sources = sources.mask(sources.str.casefold().isin({"", "nan", "none"}), metadata_file.parent.name)
            else:
                sources = pd.Series(metadata_file.parent.name, index=chunk.index)

            kept_indices = chunk.index[valid]
            sample_ids = [
                hashlib.sha1(f"{metadata_file}:{row_offset + int(index)}".encode()).hexdigest()[:20]
                for index in kept_indices
            ]
            frame = pd.DataFrame(
                {
                    "sample_id": sample_ids,
                    "image_path": paths.loc[valid].values,
                    "target": labels.loc[valid].astype("int8").values,
                    "source": sources.loc[valid].astype(str).values,
                }
            )
            frames.append(frame)
            valid_rows += len(frame)
            row_offset += len(chunk)

        reports.append(
            {
                "metadata_file": str(metadata_file),
                "rows": row_offset,
                "valid_rows": valid_rows,
                "unresolved_labels": unresolved_rows,
            }
        )
        print(f"  {metadata_file.parent.name}: {valid_rows:,} usable rows")

    if not frames:
        raise RuntimeError("Metadata files were found, but none contained usable labeled rows.")
    index = pd.concat(frames, ignore_index=True)
    canonical = index["image_path"].str.replace("\\", "/", regex=False).str.casefold()
    index = index.loc[~canonical.duplicated()].reset_index(drop=True)
    return index, reports


def _balanced_quotas(counts: pd.Series, requested: int) -> dict[str, int]:
    counts = counts.sort_index().astype(int)
    if int(counts.sum()) < requested:
        raise ValueError(f"Requested {requested:,} rows, but only {int(counts.sum()):,} are available.")
    quotas = {str(name): 0 for name in counts.index}
    remaining = requested
    while remaining:
        active = [str(name) for name in counts.index if quotas[str(name)] < int(counts.loc[name])]
        if not active:
            raise RuntimeError("Unable to allocate the requested balanced sample.")
        share = max(1, remaining // len(active))
        progressed = 0
        for name in active:
            available = int(counts.loc[name]) - quotas[name]
            addition = min(share, available, remaining - progressed)
            quotas[name] += addition
            progressed += addition
            if progressed == remaining:
                break
        remaining -= progressed
    return quotas


def balanced_sample(frame: pd.DataFrame, requested: int, seed: int) -> pd.DataFrame:
    counts = frame.groupby("source", observed=True).size()
    quotas = _balanced_quotas(counts, requested)
    pieces = []
    for source, quota in quotas.items():
        source_seed = (seed + zlib.crc32(source.encode("utf-8"))) % (2**32 - 1)
        group = frame.loc[frame["source"].astype(str) == source]
        pieces.append(group.sample(n=quota, replace=False, random_state=source_seed))
    return pd.concat(pieces, ignore_index=True)


def _digest_file(task: tuple[str, bool]) -> tuple[str | None, str | None]:
    path, verify = task
    try:
        digest = hashlib.blake2b(digest_size=20)
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if verify:
            with Image.open(path) as image:
                image.verify()
        return digest.hexdigest(), None
    except (OSError, ValueError) as error:
        return None, f"{path}: {error}"


def add_content_hashes(frame: pd.DataFrame, workers: int, verify: bool) -> tuple[pd.DataFrame, list[str]]:
    tasks = [(path, verify) for path in frame["image_path"].astype(str)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        results = list(tqdm(executor.map(_digest_file, tasks), total=len(tasks), desc="Hashing candidates"))
    hashes, errors = zip(*results) if results else ([], [])
    output = frame.copy()
    output["content_hash"] = hashes
    error_messages = [error for error in errors if error]
    output = output.loc[output["content_hash"].notna()].copy()

    conflict_hashes = output.groupby("content_hash")["target"].nunique()
    conflict_hashes = set(conflict_hashes[conflict_hashes > 1].index)
    output = output.loc[~output["content_hash"].isin(conflict_hashes)]
    output = output.drop_duplicates("content_hash", keep="first").reset_index(drop=True)
    return output, error_messages


def _integer_split_counts(group_sizes: pd.Series, ratios: np.ndarray, totals: np.ndarray) -> dict[str, np.ndarray]:
    names = [str(name) for name in group_sizes.index]
    raw = np.outer(group_sizes.to_numpy(dtype=int), ratios)
    allocation = np.floor(raw).astype(int)
    for row, size in enumerate(group_sizes.to_numpy(dtype=int)):
        remainder = size - int(allocation[row].sum())
        order = np.argsort(-(raw[row] - allocation[row]))
        allocation[row, order[:remainder]] += 1
        if size >= len(ratios):
            for column in np.flatnonzero(allocation[row] == 0):
                donor = int(np.argmax(allocation[row]))
                if allocation[row, donor] > 1:
                    allocation[row, donor] -= 1
                    allocation[row, column] += 1

    delta = totals - allocation.sum(axis=0)
    guard = 0
    while np.any(delta != 0):
        guard += 1
        if guard > int(group_sizes.sum()) * 2:
            raise RuntimeError("Could not reconcile exact split sizes.")
        destinations = np.flatnonzero(delta > 0)
        donors = np.flatnonzero(delta < 0)
        if not len(destinations) or not len(donors):
            raise RuntimeError("Invalid split allocation state.")
        destination, donor = int(destinations[0]), int(donors[0])
        candidates = []
        for row, size in enumerate(group_sizes.to_numpy(dtype=int)):
            minimum = 1 if size >= len(ratios) else 0
            if allocation[row, donor] > minimum:
                old_error = abs(allocation[row, donor] - raw[row, donor]) + abs(allocation[row, destination] - raw[row, destination])
                new_error = abs(allocation[row, donor] - 1 - raw[row, donor]) + abs(allocation[row, destination] + 1 - raw[row, destination])
                candidates.append((new_error - old_error, row))
        if not candidates:
            raise RuntimeError("Minimum per-source coverage prevents exact split allocation.")
        _, row = min(candidates)
        allocation[row, donor] -= 1
        allocation[row, destination] += 1
        delta[destination] -= 1
        delta[donor] += 1
    return {name: allocation[row] for row, name in enumerate(names)}


def make_exact_splits(
    frame: pd.DataFrame,
    ratios: tuple[float, float, float],
    seed: int,
    exact_totals: tuple[int, int, int] | None = None,
) -> pd.DataFrame:
    split_names = np.array(["train", "validation", "test"])
    ratios_array = np.asarray(ratios, dtype=float)
    pieces = []
    for target, class_frame in frame.groupby("target", sort=True):
        class_size = len(class_frame)
        if exact_totals is not None:
            totals = np.asarray(exact_totals, dtype=int)
            if int(totals.sum()) != class_size:
                raise ValueError(
                    f"Exact split totals sum to {int(totals.sum()):,}, but target {target} has {class_size:,} rows."
                )
        else:
            raw_totals = ratios_array * class_size
            totals = np.floor(raw_totals).astype(int)
            remainder = class_size - int(totals.sum())
            totals[np.argsort(-(raw_totals - totals))[:remainder]] += 1
        sizes = class_frame.groupby("source", observed=True).size().sort_index()
        allocations = _integer_split_counts(sizes, ratios_array, totals)
        for source, counts in allocations.items():
            source_seed = (seed + int(target) * 1_000_003 + zlib.crc32(source.encode())) % (2**32 - 1)
            shuffled = class_frame.loc[class_frame["source"].astype(str) == source].sample(frac=1, random_state=source_seed)
            start = 0
            for split, count in zip(split_names, counts):
                part = shuffled.iloc[start : start + int(count)].copy()
                part["split"] = split
                pieces.append(part)
                start += int(count)
    result = pd.concat(pieces, ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def _manifest_signature(config: dict[str, Any]) -> dict[str, Any]:
    dataset = config["dataset"]
    return {
        "seed": int(config["seed"]),
        "root": str(dataset["root"]),
        "samples": dataset["samples"],
        "deduplicate_by_content": bool(dataset.get("deduplicate_by_content", True)),
        "candidate_multiplier": float(dataset.get("candidate_multiplier", 1.2)),
    }


def validate_manifest(frame: pd.DataFrame, config: dict[str, Any]) -> None:
    required = {"sample_id", "image_path", "target", "source", "source_id", "split"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    dataset = config["dataset"]
    split_counts = configured_sample_counts(dataset)
    expected_class = {
        target: sum(split_counts[split][target] for split in ("train", "validation", "test"))
        for target in (0, 1)
    }
    actual_class = frame.groupby("target").size().to_dict()
    if actual_class != expected_class:
        raise ValueError(f"Incorrect class counts: expected {expected_class}, got {actual_class}")
    expected_split = {
        split: sum(split_counts[split].values()) for split in ("train", "validation", "test")
    }
    actual_split = frame.groupby("split").size().to_dict()
    if actual_split != expected_split:
        raise ValueError(f"Incorrect split counts: expected {expected_split}, got {actual_split}")
    for target in expected_class:
        for split in ("train", "validation", "test"):
            expected = split_counts[split][target]
            actual = int(((frame["target"] == target) & (frame["split"] == split)).sum())
            if actual != expected:
                raise ValueError(
                    f"Incorrect target={target}, split={split} count: expected {expected}, got {actual}"
                )
    if "content_hash" in frame and frame["content_hash"].dropna().duplicated().any():
        raise ValueError("Content hashes overlap between rows.")
    fake = frame.loc[frame["target"] == 1]
    fake_source_sizes = fake.groupby("source", observed=True).size()
    fake_coverage = fake.groupby("source", observed=True)["split"].nunique()
    missing_coverage = [
        str(source)
        for source, size in fake_source_sizes.items()
        if size >= 3 and int(fake_coverage.loc[source]) != 3
    ]
    if missing_coverage:
        raise ValueError(f"Fake generators missing from one or more splits: {missing_coverage}")


def prepare_manifest(config: dict[str, Any], force: bool = False) -> pd.DataFrame:
    dataset = config["dataset"]
    manifest_path = Path(dataset["manifest_file"])
    report_path = Path(dataset["report_file"])
    signature = _manifest_signature(config)
    if not force and manifest_path.is_file() and report_path.is_file():
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("signature") == signature:
            manifest = pd.read_csv(manifest_path)
            validate_manifest(manifest, config)
            print(f"Reusing validated manifest: {manifest_path}")
            return manifest

    root = find_dataset_root(dataset["root"], dataset.get("metadata_pattern", "**/metadata.csv"))
    index, metadata_reports = read_metadata(root, dataset.get("metadata_pattern", "**/metadata.csv"))
    seed = int(config["seed"])
    multiplier = float(dataset.get("candidate_multiplier", 1.2))
    split_counts = configured_sample_counts(dataset)
    requested = {
        target: sum(split_counts[split][target] for split in ("train", "validation", "test"))
        for target in (0, 1)
    }

    candidates = []
    for target, count in requested.items():
        available = index.loc[index["target"] == target]
        candidate_count = min(len(available), max(count, int(math.ceil(count * multiplier))))
        candidates.append(balanced_sample(available, candidate_count, seed + target))
    candidates_frame = pd.concat(candidates, ignore_index=True)

    hash_errors: list[str] = []
    if dataset.get("deduplicate_by_content", True):
        candidates_frame, hash_errors = add_content_hashes(
            candidates_frame,
            workers=int(dataset.get("hash_workers", 8)),
            verify=bool(dataset.get("verify_selected_images", False)),
        )

    indexed_fake_sources = set(index.loc[index["target"] == 1, "source"].astype(str))
    candidate_fake_sources = set(candidates_frame.loc[candidates_frame["target"] == 1, "source"].astype(str))
    missing_fake_sources = sorted(indexed_fake_sources - candidate_fake_sources)
    if missing_fake_sources:
        raise RuntimeError(
            "No readable unique candidates remain for these fake generators: "
            f"{missing_fake_sources}. Fix their image paths/data before training."
        )
    too_small = candidates_frame.loc[candidates_frame["target"] == 1].groupby("source", observed=True).size()
    too_small = too_small[too_small < 3]
    if not too_small.empty:
        raise RuntimeError(
            "Every fake generator needs at least three unique images to appear in train/validation/test; "
            f"insufficient generators: {too_small.to_dict()}"
        )

    selected = []
    for target, count in requested.items():
        available = candidates_frame.loc[candidates_frame["target"] == target]
        if len(available) < count:
            raise RuntimeError(
                f"Only {len(available):,} unique readable class-{target} candidates remain; {count:,} are required. "
                "Increase dataset.candidate_multiplier and rebuild with --force."
            )
        selected.append(balanced_sample(available, count, seed + 10 + target))
    manifest = pd.concat(selected, ignore_index=True)
    real_total = requested[0]
    ratios = tuple(split_counts[split][0] / real_total for split in ("train", "validation", "test"))
    exact_totals = tuple(split_counts[split][0] for split in ("train", "validation", "test"))
    manifest = make_exact_splits(manifest, ratios, seed, exact_totals=exact_totals)
    source_names = sorted(manifest["source"].astype(str).unique())
    source_map = {source: source_id for source_id, source in enumerate(source_names)}
    manifest["source_id"] = manifest["source"].astype(str).map(source_map).astype("int16")
    validate_manifest(manifest, config)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    write_json(source_map, dataset["source_map_file"])
    split_table = pd.crosstab([manifest["target"], manifest["source"]], manifest["split"]).reset_index()
    report = {
        "signature": signature,
        "resolved_dataset_root": str(root),
        "indexed_images": len(index),
        "manifest_images": len(manifest),
        "class_counts": {str(k): int(v) for k, v in manifest.groupby("target").size().items()},
        "split_counts": {str(k): int(v) for k, v in manifest.groupby("split").size().items()},
        "content_hash_errors": hash_errors[:100],
        "metadata": metadata_reports,
        "source_split_counts": split_table.to_dict(orient="records"),
    }
    write_json(report, report_path)
    print(f"Saved {len(manifest):,}-image manifest to {manifest_path}")
    return manifest


class RandomJpegCompression:
    def __init__(self, quality_min: int, quality_max: int) -> None:
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, image: Image.Image) -> Image.Image:
        quality = int(torch.randint(self.quality_min, self.quality_max + 1, (1,)).item())
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as compressed:
            return compressed.convert("RGB")


def build_transform(config: dict[str, Any], training: bool) -> transforms.Compose:
    size = int(config["dataset"]["image_size"])
    augmentation = config["augmentation"]
    normalise = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    if not training:
        return transforms.Compose(
            [transforms.Resize(int(size * 1.14), antialias=True), transforms.CenterCrop(size), transforms.ToTensor(), normalise]
        )
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.85, 1.0), ratio=(0.90, 1.10), antialias=True),
            transforms.RandomHorizontalFlip(float(augmentation["horizontal_flip_probability"])),
            transforms.RandomApply([transforms.ColorJitter(0.12, 0.12, 0.08, 0.02)], p=float(augmentation["color_jitter_probability"])),
            transforms.RandomApply(
                [RandomJpegCompression(int(augmentation["jpeg_quality_min"]), int(augmentation["jpeg_quality_max"]))],
                p=float(augmentation["jpeg_probability"]),
            ),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=float(augmentation["blur_probability"])),
            transforms.ToTensor(),
            normalise,
        ]
    )


class ForensicImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.paths = frame["image_path"].astype(str).tolist()
        self.targets = frame["target"].astype(int).to_numpy()
        self.source_ids = frame["source_id"].astype(int).to_numpy()
        self.sample_ids = frame["sample_id"].astype(str).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                tensor = self.transform(image.convert("RGB"))
        except (OSError, ValueError) as error:
            raise RuntimeError(f"Could not decode manifest image: {path}") from error
        return {
            "image": tensor,
            "target": torch.tensor(self.targets[index], dtype=torch.float32),
            "source_id": torch.tensor(self.source_ids[index], dtype=torch.long),
            "sample_id": self.sample_ids[index],
            "image_path": path,
        }


def make_loader(
    frame: pd.DataFrame,
    config: dict[str, Any],
    training: bool,
    epoch: int = 0,
) -> DataLoader:
    loader_config = config["loader"]
    workers = int(loader_config["num_workers"])
    generator = torch.Generator().manual_seed(int(config["seed"]) + epoch)
    kwargs: dict[str, Any] = {
        "dataset": ForensicImageDataset(frame, build_transform(config, training)),
        "batch_size": int(loader_config["batch_size"]),
        "shuffle": training,
        "num_workers": workers,
        "pin_memory": bool(loader_config["pin_memory"] and torch.cuda.is_available()),
        # Keep all 200,000 training examples; the final batch is still large
        # enough for the residual stream's batch-normalisation layers.
        "drop_last": False,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if workers > 0:
        kwargs["persistent_workers"] = bool(loader_config.get("persistent_workers", True))
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)
