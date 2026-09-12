import os
import sys
import gc
import json
import random
import hashlib
from pathlib import Path

import yaml
import numpy as np
import pandas as pd

from PIL import Image, ImageFilter

from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

try:
    import torch_directml
except ImportError:
    torch_directml = None

from torchvision import transforms

from transformers import (
    AutoModel,
    ConvNextV2Model,
)


# ============================================================
# CONFIG
# ============================================================

config_path = "config.yaml"
resume_checkpoint_path = None
root_override = None

for i, arg in enumerate(sys.argv):
    if arg == "--config" and i + 1 < len(sys.argv):
        config_path = sys.argv[i + 1]
    elif arg.startswith("--config="):
        config_path = arg.split("=", 1)[1]
    elif arg == "--resume" and i + 1 < len(sys.argv):
        resume_checkpoint_path = sys.argv[i + 1]
    elif arg.startswith("--resume="):
        resume_checkpoint_path = arg.split("=", 1)[1]
    elif arg == "--root" and i + 1 < len(sys.argv):
        root_override = sys.argv[i + 1]
    elif arg.startswith("--root="):
        root_override = arg.split("=", 1)[1]

if os.environ.get("CONFIG_PATH"):
    config_path = os.environ["CONFIG_PATH"]

if os.environ.get("RESUME_PATH"):
    resume_checkpoint_path = os.environ["RESUME_PATH"]

if os.environ.get("DATASET_ROOT"):
    root_override = os.environ["DATASET_ROOT"]

print(f"Loading configuration from: {config_path}")
with open(config_path, "r") as f:
    CFG = yaml.safe_load(f)

if root_override:
    CFG["dataset"]["root"] = root_override
    print(f"Dataset root overridden to: {root_override}")


def resolve_drive_paths(cfg):
    shared_prefix = "/content/drive/MyDrive/shared-with-me/ForensicFusion"
    direct_prefix = "/content/drive/MyDrive/ForensicFusion"

    target_prefix = None
    replacement_prefix = None

    if Path(direct_prefix).exists() and not Path(shared_prefix).exists():
        target_prefix = shared_prefix
        replacement_prefix = direct_prefix
    elif Path(shared_prefix).exists() and not Path(direct_prefix).exists():
        target_prefix = direct_prefix
        replacement_prefix = shared_prefix

    if target_prefix and replacement_prefix:
        print(f"[Drive Auto-Detect] Remapping path prefix {target_prefix} -> {replacement_prefix}")
        def _remap(obj):
            if isinstance(obj, dict):
                return {k: _remap(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_remap(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith(target_prefix):
                return obj.replace(target_prefix, replacement_prefix, 1)
            return obj
        return _remap(cfg)
    return cfg


CFG = resolve_drive_paths(CFG)


SEED = CFG["seed"]


# ============================================================
# REPRODUCIBILITY
# ============================================================

def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)


seed_everything(SEED)


# ============================================================
# DEVICE
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch_directml is not None and torch_directml.is_available():
        return torch_directml.device()
    return torch.device("cpu")

DEVICE = get_device()
if str(DEVICE).startswith("cuda"):
    torch.backends.cudnn.benchmark = True


# ============================================================
# PATHS
# ============================================================

DATASET_ROOT = Path(
    CFG["dataset"]["root"]
)

INDEX_FILE = Path(
    CFG["dataset"]["index_file"]
)

SPLIT_FILE = Path(
    CFG["dataset"]["split_file"]
)

GENERATOR_FILE = Path(
    CFG["dataset"]["generator_file"]
)

REPORT_FILE = Path(
    CFG["dataset"]["report_file"]
)

CHECKPOINT_DIR = Path(
    CFG["checkpoints"]["directory"]
)

LATEST_CHECKPOINT = Path(
    CFG["checkpoints"]["latest"]
)

BEST_CHECKPOINT = Path(
    CFG["checkpoints"]["best"]
)

LOG_FILE = Path(
    CFG["logs"]["training_file"]
)


for directory in [
    INDEX_FILE.parent,
    CHECKPOINT_DIR,
    LOG_FILE.parent,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# DATASET DISCOVERY
# ============================================================

def discover_metadata_files():

    print("Searching for metadata.csv files...")

    # Fast shallow search first
    files = sorted(DATASET_ROOT.glob("*/metadata.csv"))
    if not files:
        files = sorted(DATASET_ROOT.glob("*/metadata*.csv"))
    if not files:
        files = sorted(DATASET_ROOT.glob("*/*/metadata.csv"))

    # If configured root has few files (< 30) while dataset is unzipped on local disk:
    if len(files) < 30:
        disk_candidates = [
            Path("/content/dataset/data"),
            Path("/content/dataset"),
            Path("./dataset/data"),
            Path("./dataset"),
            Path("/content/ForensicFusion/dataset/data"),
            Path("/content/ForensicFusion/dataset"),
            Path("/content/data"),
        ]
        for candidate in disk_candidates:
            try:
                if candidate.exists() and candidate.resolve() != DATASET_ROOT.resolve():
                    cand_files = sorted(candidate.glob("*/metadata.csv"))
                    if not cand_files:
                        cand_files = sorted(candidate.glob("*/*/metadata.csv"))
                    if len(cand_files) > len(files):
                        print(f"Auto-detected full unzipped dataset with {len(cand_files)} folders on local disk: {candidate}")
                        files = cand_files
                        break
            except Exception:
                pass

    # Fallback to rglob if still not found
    if not files:
        files = sorted(DATASET_ROOT.rglob("metadata.csv"))

    print(
        f"Found {len(files)} metadata.csv files."
    )

    for file in files:

        print(
            "  ",
            file
        )

    if not files:

        raise RuntimeError(
            f"No metadata.csv found inside:\n"
            f"{DATASET_ROOT}"
        )

    return files


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(
    dataframe,
    candidates
):

    columns = {
        str(column).strip().lower():
            column

        for column in dataframe.columns
    }

    for candidate in candidates:

        if candidate.lower() in columns:

            return columns[
                candidate.lower()
            ]

    return None


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(value):

    if pd.isna(value):

        return None

    # Numeric labels (0 = Real, >0 = Synthetic/Fake)
    try:
        val_float = float(value)
        if val_float == 0:
            return 0
        elif val_float > 0:
            return 1
    except (ValueError, TypeError):
        pass

    text = str(
        value
    ).strip().lower()

    # REAL
    real_values = {
        "real",
        "original",
        "authentic",
        "natural",
        "0",
        "false",
        "false image",
        "non-fake",
        "non_fake",
        "human",
    }

    # FAKE
    fake_values = {
        "fake",
        "synthetic",
        "generated",
        "ai",
        "ai-generated",
        "ai_generated",
        "deepfake",
        "1",
        "true",
        "artifact",
    }

    if text in real_values:

        return 0

    if text in fake_values:

        return 1

    return None


# ============================================================
# LABEL INFERENCE FROM PATH
# ============================================================

def infer_label_from_path(path):

    parts = [
        part.lower()
        for part in Path(path).parts
    ]

    fake_keywords = [
        "fake",
        "synthetic",
        "generated",
        "generation",
        "gan",
        "diffusion",
        "stylegan",
        "biggan",
        "progan",
        "stable",
        "sdxl",
        "dalle",
        "midjourney",
        "imagen",
        "artificial",
        "deepfake",
        "cips",
        "ddpm",
        "inpaint",
        "gcinpaint",
        "glide",
        "lama",
        "mat",
        "palette",
        "facesyn",
        "sfhq",
        "taming",
        "vqdiff",
    ]

    real_keywords = [
        "real",
        "original",
        "authentic",
        "natural",
        "human",
        "coco",
        "imagenet",
        "celebahq",
        "ffhq",
        "lsun",
        "metfaces",
    ]

    for part in parts:

        if any(
            keyword in part
            for keyword in fake_keywords
        ):

            return 1

    for part in parts:

        if any(
            keyword in part
            for keyword in real_keywords
        ):

            return 0

    return None


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================

def resolve_image_path(
    raw_path,
    parent_str,
    parent_name,
    existing_files_set=None
):

    if not raw_path:
        return None

    raw_str = str(raw_path).strip()
    if not raw_str or raw_str.lower() == "nan":
        return None

    raw_norm = raw_str.replace("/", os.sep) if "/" in raw_str else raw_str

    if existing_files_set is not None:
        raw_lower = raw_norm.lower()
        if raw_lower in existing_files_set:
            return os.path.join(parent_str, raw_norm)

        # Redundant parent folder prefix (e.g. afhq/afhq/afhq/train/dog/...)
        pfx = parent_name.lower() + os.sep
        if raw_lower.startswith(pfx):
            sub = raw_norm[len(parent_name) + 1:]
            if sub.lower() in existing_files_set:
                return os.path.join(parent_str, sub)

        # Stripped leading ./
        stripped = raw_norm.lstrip("." + os.sep)
        if stripped.lower() in existing_files_set:
            return os.path.join(parent_str, stripped)

        # Relative to dataset root
        p3 = os.path.join(str(DATASET_ROOT), raw_norm)
        if os.path.isfile(p3):
            return p3

        # Absolute path
        if os.path.isabs(raw_norm) and os.path.isfile(raw_norm):
            return raw_norm

        return None

    # Common case 1: relative to CSV parent directory
    p1 = os.path.join(parent_str, raw_norm)
    if os.path.isfile(p1):
        return p1

    # Redundant parent folder prefix (e.g. afhq/afhq/afhq/train/dog/...)
    if raw_norm.lower().startswith(parent_name.lower() + os.sep):
        p2 = os.path.join(parent_str, raw_norm[len(parent_name) + 1:])
        if os.path.isfile(p2):
            return p2

    # Relative to dataset root
    p3 = os.path.join(str(DATASET_ROOT), raw_norm)
    if os.path.isfile(p3):
        return p3

    # Stripped leading ./
    p4 = os.path.join(parent_str, raw_norm.lstrip("." + os.sep))
    if os.path.isfile(p4):
        return p4

    # Absolute path
    if os.path.isabs(raw_norm) and os.path.isfile(raw_norm):
        return raw_norm

    return None


# ============================================================
# READ ONE CSV
# ============================================================

def process_metadata_file(
    metadata_file,
    generator_map
):

    print()
    print(
        "Processing:",
        metadata_file
    )

    try:
        dataframe = pd.read_csv(
            metadata_file,
            low_memory=False
        )
    except Exception as error:
        print(
            "FAILED:",
            error
        )
        return pd.DataFrame(), {
            "metadata_file": str(metadata_file),
            "error": str(error)
        }

    if dataframe.empty:
        return pd.DataFrame(), {
            "metadata_file": str(metadata_file),
            "rows": 0,
            "valid_rows": 0
        }

    # Detect column names
    image_column = find_column(
        dataframe,
        [
            "image_path",
            "image",
            "path",
            "filepath",
            "file_path",
            "filename",
            "file_name",
            "img",
            "img_path",
        ]
    )

    label_column = find_column(
        dataframe,
        [
            "label",
            "target",
            "class",
            "is_fake",
            "fake",
            "real",
        ]
    )

    generator_column = find_column(
        dataframe,
        [
            "generator",
            "generator_name",
            "model",
            "model_name",
            "source",
            "method",
            "architecture",
        ]
    )

    category_column = find_column(
        dataframe,
        [
            "category",
            "type",
            "dataset",
        ]
    )

    if image_column is None:
        print(
            "WARNING: No obvious image column."
        )
        return pd.DataFrame(), {
            "metadata_file": str(metadata_file),
            "error": "No image column"
        }

    parent_str = str(metadata_file.parent)
    parent_name = metadata_file.parent.name

    if parent_name not in generator_map:
        generator_map[parent_name] = len(generator_map)

    # Fast path resolution prefix check on first valid row
    valid_series = dataframe[image_column].dropna().astype(str)
    if valid_series.empty:
        return pd.DataFrame(), {
            "metadata_file": str(metadata_file),
            "rows": len(dataframe),
            "valid_rows": 0
        }

    first_raw = valid_series.iloc[0].strip().replace("/", os.sep)
    p1 = os.path.join(parent_str, first_raw)

    strip_len = 0
    base_dir = parent_str
    if os.path.isfile(p1):
        base_dir = parent_str
    elif first_raw.lower().startswith(parent_name.lower() + os.sep):
        sub = first_raw[len(parent_name) + 1:]
        if os.path.isfile(os.path.join(parent_str, sub)):
            strip_len = len(parent_name) + 1
            base_dir = parent_str
    elif os.path.isfile(os.path.join(str(DATASET_ROOT), first_raw)):
        base_dir = str(DATASET_ROOT)
    elif os.path.isabs(first_raw) and os.path.isfile(first_raw):
        base_dir = ""

    # Vectorized path string construction
    raw_paths = dataframe[image_column].fillna("").astype(str).str.strip().str.replace("/", os.sep)
    if strip_len > 0:
        raw_paths = raw_paths.str[strip_len:]

    if base_dir:
        image_paths = base_dir + os.sep + raw_paths
    else:
        image_paths = raw_paths

    # Vectorized Labels
    if label_column is not None:
        num_target = pd.to_numeric(dataframe[label_column], errors="coerce")
        if num_target.notna().sum() > 0:
            target = (num_target.fillna(1) > 0).astype(int)
        else:
            target = dataframe[label_column].astype(str).str.strip().str.lower().map({
                "real": 0, "original": 0, "authentic": 0, "natural": 0, "0": 0, "false": 0,
                "fake": 1, "synthetic": 1, "generated": 1, "ai": 1, "1": 1, "true": 1
            }).fillna(1).astype(int)
    else:
        target = image_paths.str.lower().apply(infer_label_from_path).fillna(1).astype(int)

    # Vectorized Generator & Category
    if generator_column is not None:
        generator = dataframe[generator_column].fillna(parent_name).astype(str).str.strip()
        generator = generator.replace("", parent_name).replace("nan", parent_name)
    else:
        generator = pd.Series(parent_name, index=dataframe.index)

    for g in generator.unique():
        if g not in generator_map:
            generator_map[g] = len(generator_map)
    generator_id = generator.map(generator_map).astype(int)

    if category_column is not None:
        category = dataframe[category_column].fillna("unknown").astype(str)
    else:
        category = pd.Series("unknown", index=dataframe.index)

    sample_id = parent_name + "_" + dataframe.index.astype(str)

    res_df = pd.DataFrame({
        "sample_id": sample_id,
        "image_path": image_paths,
        "target": target,
        "generator": generator,
        "generator_id": generator_id,
        "category": category,
        "metadata_file": str(metadata_file),
        "dataset_folder": parent_str,
    })

    # Drop any entries with empty image_path
    min_len = len(parent_str) + 1 if base_dir else 1
    res_df = res_df[res_df["image_path"].str.len() > min_len].copy()

    report = {
        "metadata_file": str(metadata_file),
        "rows": len(dataframe),
        "valid_rows": len(res_df),
        "missing_images": len(dataframe) - len(res_df),
        "unresolved_labels": 0,
        "columns": list(dataframe.columns),
    }

    print(
        f"Rows: {len(dataframe)} | Valid: {len(res_df)}"
    )

    return res_df, report


# ============================================================
# BUILD GLOBAL INDEX
# ============================================================

def build_index():

    metadata_files = (
        discover_metadata_files()
    )

    generator_map = {}

    dfs = []

    reports = []

    for metadata_file in metadata_files:

        df, report = (
            process_metadata_file(
                metadata_file,
                generator_map
            )
        )

        if not df.empty:
            dfs.append(
                df
            )

        reports.append(
            report
        )

    if not dfs:

        raise RuntimeError(
            "No valid images were discovered."
        )

    dataframe = pd.concat(
        dfs,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Remove duplicate image paths
    # --------------------------------------------------------

    before = len(
        dataframe
    )

    dataframe = (
        dataframe.drop_duplicates(
            subset=[
                "image_path"
            ]
        )
    )

    duplicates = (
        before - len(dataframe)
    )

    print()
    print(
        "Duplicate images removed:",
        duplicates
    )

    # --------------------------------------------------------
    # Verify image readability (optional)
    # --------------------------------------------------------

    verify_images = CFG.get("dataset", {}).get("verify_images", False)
    bad_images = []

    if verify_images:
        print()
        print(
            "Checking image integrity..."
        )

        for path in tqdm(
            dataframe[
                "image_path"
            ],
            desc="Checking images"
        ):

            try:

                with Image.open(
                    path
                ) as image:

                    image.verify()

            except Exception:

                bad_images.append(
                    path
                )

        if bad_images:

            print(
                "Corrupted images:",
                len(bad_images)
            )

            dataframe = dataframe[
                ~dataframe[
                    "image_path"
                ].isin(
                    bad_images
                )
            ]

    # --------------------------------------------------------
    # Save index
    # --------------------------------------------------------

    dataframe = dataframe.reset_index(
        drop=True
    )

    dataframe.to_csv(
        INDEX_FILE,
        index=False
    )

    with open(
        GENERATOR_FILE,
        "w"
    ) as file:

        json.dump(
            generator_map,
            file,
            indent=2
        )

    report = {

        "dataset_root":
            str(DATASET_ROOT),

        "metadata_files":
            len(metadata_files),

        "total_images":
            len(dataframe),

        "real_images":
            int(
                (
                    dataframe.target
                    == 0
                ).sum()
            ),

        "fake_images":
            int(
                (
                    dataframe.target
                    == 1
                ).sum()
            ),

        "generators":
            len(generator_map),

        "duplicates_removed":
            duplicates,

        "corrupted_images":
            len(bad_images),

        "files":
            reports
    }

    with open(
        REPORT_FILE,
        "w"
    ) as file:

        json.dump(
            report,
            file,
            indent=2
        )

    print()
    print("=" * 80)
    print("DATASET INDEX COMPLETE")
    print("=" * 80)

    print(
        "Images:",
        len(dataframe)
    )

    print(
        "Real:",
        int(
            (
                dataframe.target
                == 0
            ).sum()
        )
    )

    print(
        "Fake:",
        int(
            (
                dataframe.target
                == 1
            ).sum()
        )
    )

    print(
        "Generators:",
        len(generator_map)
    )

    print(
        "Index:",
        INDEX_FILE
    )

    return dataframe


# ============================================================
# LOAD INDEX
# ============================================================

def get_index():

    if INDEX_FILE.exists():

        try:
            df = pd.read_csv(
                INDEX_FILE,
                low_memory=False
            )
            required_cols = {"sample_id", "image_path", "target", "generator_id"}
            if not df.empty and required_cols.issubset(df.columns):
                sample_path = str(df["image_path"].iloc[0])
                if os.path.exists(sample_path):
                    print(
                        "Existing dataset index found."
                    )
                    return df
                else:
                    print(
                        f"Existing index points to paths that do not exist ({sample_path}). Rebuilding..."
                    )
            else:
                print(
                    "Existing dataset index file is empty or missing required columns. Rebuilding..."
                )
        except Exception as error:
            print(
                f"Error reading existing index file ({error}). Rebuilding..."
            )

    return build_index()


# ============================================================
# CREATE SPLITS
# ============================================================

def create_splits(
    dataframe
):

    if SPLIT_FILE.exists():

        try:
            split_df = pd.read_csv(
                SPLIT_FILE,
                low_memory=False
            )
            if (
                not split_df.empty
                and "split" in split_df.columns
                and len(split_df) == len(dataframe)
                and set(split_df["split"].dropna().unique()).issubset({"train", "validation", "test"})
            ):
                print(
                    "Existing dataset split found."
                )
                return split_df
            else:
                print(
                    "Existing dataset split file is empty, invalid, or mismatched. Rebuilding splits..."
                )
        except Exception as error:
            print(
                f"Error reading existing split file ({error}). Rebuilding splits..."
            )

    dataframe = dataframe.copy()

    # --------------------------------------------------------
    # Stratify by target + generator
    # --------------------------------------------------------

    dataframe[
        "stratum"
    ] = (
        dataframe[
            "target"
        ].astype(str)
        + "_"
        +
        dataframe[
            "generator_id"
        ].astype(str)
    )

    dataframe[
        "split"
    ] = "train"

    for _, group in dataframe.groupby(
        "stratum"
    ):

        shuffled = group.sample(
            frac=1,
            random_state=SEED
        )

        n = len(
            shuffled
        )

        validation_count = max(
            1,
            int(
                n
                *
                CFG["dataset"][
                    "validation_ratio"
                ]
            )
        )

        test_count = max(
            1,
            int(
                n
                *
                CFG["dataset"][
                    "test_ratio"
                ]
            )
        )

        validation_indices = (
            shuffled.iloc[
                :validation_count
            ].index
        )

        test_indices = (
            shuffled.iloc[
                validation_count:
                validation_count
                +
                test_count
            ].index
        )

        dataframe.loc[
            validation_indices,
            "split"
        ] = "validation"

        dataframe.loc[
            test_indices,
            "split"
        ] = "test"

    dataframe.drop(
        columns=[
            "stratum"
        ],
        inplace=True
    )

    dataframe.to_csv(
        SPLIT_FILE,
        index=False
    )

    print()
    print(
        dataframe[
            "split"
        ].value_counts()
    )

    return dataframe


# ============================================================
# FORENSIC HIGH PASS
# ============================================================

class HighPass:

    def __call__(
        self,
        image
    ):

        blurred = image.filter(
            ImageFilter.GaussianBlur(
                radius=2
            )
        )

        original = np.asarray(
            image
        ).astype(
            np.float32
        )

        blurred = np.asarray(
            blurred
        ).astype(
            np.float32
        )

        residual = (
            original
            -
            blurred
        )

        residual = np.abs(
            residual
        )

        residual /= (
            residual.max()
            +
            1e-8
        )

        residual *= 255

        return Image.fromarray(
            residual.astype(
                np.uint8
            )
        )


# ============================================================
# FFT
# ============================================================

class FrequencyTransform:

    def __call__(
        self,
        image
    ):

        array = np.asarray(
            image
        ).astype(
            np.float32
        )

        gray = (
            0.299 * array[:, :, 0]
            +
            0.587 * array[:, :, 1]
            +
            0.114 * array[:, :, 2]
        )

        fft = np.fft.fft2(
            gray
        )

        fft = np.fft.fftshift(
            fft
        )

        magnitude = np.log1p(
            np.abs(fft)
        )

        magnitude -= (
            magnitude.min()
        )

        magnitude /= (
            magnitude.max()
            +
            1e-8
        )

        magnitude *= 255

        return Image.fromarray(
            magnitude.astype(
                np.uint8
            )
        ).convert(
            "RGB"
        )


# ============================================================
# DATASET
# ============================================================

class ArtifactDataset(
    Dataset
):

    def __init__(
        self,
        dataframe,
        image_size,
        training
    ):

        self.dataframe = (
            dataframe.reset_index(
                drop=True
            )
        )

        self.image_size = image_size
        self.training = training

        self.highpass = (
            HighPass()
        )

        self.frequency = (
            FrequencyTransform()
        )

        if training:

            self.rgb_transform = (
                transforms.Compose([

                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(
                            0.80,
                            1.0
                        )
                    ),

                    transforms.RandomHorizontalFlip(
                        p=0.5
                    ),

                    transforms.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.10,
                        hue=0.02
                    ),

                    transforms.ToTensor(),

                    transforms.Normalize(
                        [
                            0.485,
                            0.456,
                            0.406
                        ],
                        [
                            0.229,
                            0.224,
                            0.225
                        ]
                    )
                ])

            )

        else:

            self.rgb_transform = (
                transforms.Compose([

                    transforms.Resize(
                        (
                            image_size,
                            image_size
                        )
                    ),

                    transforms.ToTensor(),

                    transforms.Normalize(
                        [
                            0.485,
                            0.456,
                            0.406
                        ],
                        [
                            0.229,
                            0.224,
                            0.225
                        ]
                    )
                ])
            )

        self.forensic_transform = (
            transforms.Compose([

                transforms.ToTensor(),

                transforms.Normalize(
                    [
                        0.5,
                        0.5,
                        0.5
                    ],
                    [
                        0.5,
                        0.5,
                        0.5
                    ]
                )
            ])
        )

    def __len__(
        self
    ):

        return len(
            self.dataframe
        )

    def __getitem__(
        self,
        index
    ):

        row = (
            self.dataframe.iloc[
                index
            ]
        )

        image_path = row[
            "image_path"
        ]

        image = None
        for attempt in range(3):
            try:
                image = Image.open(
                    image_path
                ).convert(
                    "RGB"
                )
                break
            except Exception:
                if attempt == 2:
                    # Fallback to another random sample to never crash training on I/O error
                    fallback_idx = random.randint(0, len(self.dataframe) - 1)
                    return self.__getitem__(fallback_idx)
                time.sleep(0.05)

        # Pre-resize to target dimensions for 15x faster FFT and Gaussian filtering
        if image.size != (self.image_size, self.image_size):
            image_small = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        else:
            image_small = image

        rgb = (
            self.rgb_transform(
                image if self.training else image_small
            )
        )

        highpass_image = (
            self.highpass(
                image_small
            )
        )

        highpass = (
            self.forensic_transform(
                highpass_image
            )
        )

        frequency_image = (
            self.frequency(
                image_small
            )
        )

        frequency = (
            self.forensic_transform(
                frequency_image
            )
        )

        return {

            "rgb":
                rgb,

            "highpass":
                highpass,

            "frequency":
                frequency,

            "label":
                torch.tensor(
                    int(
                        row["target"]
                    ),
                    dtype=torch.long
                ),

            "generator":
                torch.tensor(
                    int(
                        row["generator_id"]
                    ),
                    dtype=torch.long
                ),

            "sample_id":
                row["sample_id"]
        }


# ============================================================
# DINO
# ============================================================

class DINOEncoder(
    nn.Module
):

    def __init__(
        self,
        name
    ):

        super().__init__()

        print()
        print(
            "Loading DINO (DINOv2)..."
        )

        self.model = (
            AutoModel.from_pretrained(
                name
            )
        )

        self.output_dim = (
            self.model.config.hidden_size
        )

        print(
            "DINO dimension:",
            self.output_dim
        )

    def forward(
        self,
        x
    ):

        output = self.model(
            pixel_values=x
        )

        if (
            hasattr(
                output,
                "pooler_output"
            )
            and
            output.pooler_output
            is not None
        ):

            feature = (
                output.pooler_output
            )

        else:

            feature = (
                output.last_hidden_state[
                    :, 0
                ]
            )

        return feature


# ============================================================
# CONVNEXT
# ============================================================

class ConvNextEncoder(
    nn.Module
):

    def __init__(
        self,
        name
    ):

        super().__init__()

        print()
        print(
            "Loading ConvNeXtV2..."
        )

        self.model = (
            ConvNextV2Model.from_pretrained(
                name
            )
        )

        self.output_dim = (
            self.model.config.hidden_sizes[
                -1
            ]
        )

        print(
            "ConvNeXt dimension:",
            self.output_dim
        )

    def forward(
        self,
        x
    ):

        output = self.model(
            pixel_values=x
        )

        return (
            output.pooler_output
        )


# ============================================================
# FORENSIC CNN
# ============================================================

class ForensicCNN(
    nn.Module
):

    def __init__(
        self,
        output_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.GELU(),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.GELU(),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.GELU(),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                128,
                256,
                3,
                padding=1
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.GELU(),

            nn.AdaptiveAvgPool2d(
                1
            )
        )

        self.projection = nn.Linear(
            256,
            output_dim
        )

    def forward(
        self,
        x
    ):

        x = self.network(
            x
        )

        x = x.flatten(
            1
        )

        return self.projection(
            x
        )


# ============================================================
# PROJECTION
# ============================================================

class Projection(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        output_dim
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                output_dim
            ),

            nn.LayerNorm(
                output_dim
            ),

            nn.GELU(),

            nn.Dropout(
                0.10
            )
        )

    def forward(
        self,
        x
    ):

        return self.network(
            x
        )


# ============================================================
# FUSION TRANSFORMER
# ============================================================

class FusionTransformer(
    nn.Module
):

    def __init__(
        self,
        dimension
    ):

        super().__init__()

        self.cls_token = nn.Parameter(
            torch.randn(
                1,
                1,
                dimension
            ) * 0.02
        )

        encoder_layer = (
            nn.TransformerEncoderLayer(
                d_model=dimension,
                nhead=4,
                dim_feedforward=
                    dimension * 4,
                dropout=0.10,
                activation="gelu",
                batch_first=True,
                norm_first=True
            )
        )

        self.encoder = (
            nn.TransformerEncoder(
                encoder_layer,
                num_layers=2
            )
        )

    def forward(
        self,
        tokens
    ):

        batch_size = (
            tokens.shape[0]
        )

        cls = (
            self.cls_token
            .expand(
                batch_size,
                -1,
                -1
            )
        )

        tokens = torch.cat(
            [
                cls,
                tokens
            ],
            dim=1
        )

        output = (
            self.encoder(
                tokens
            )
        )

        return output[:, 0]


# ============================================================
# COMPLETE MODEL
# ============================================================

class ArtifactDetector(
    nn.Module
):

    def __init__(
        self,
        config,
        num_generators
    ):

        super().__init__()

        dimension = (
            config["model"][
                "fusion_dim"
            ]
        )

        # ----------------------------------------------------
        # PRETRAINED EXPERT 1
        # ----------------------------------------------------

        self.dino = (
            DINOEncoder(
                config[
                    "model"
                ][
                    "dino"
                ][
                    "name"
                ]
            )
        )

        # ----------------------------------------------------
        # PRETRAINED EXPERT 2
        # ----------------------------------------------------

        self.convnext = (
            ConvNextEncoder(
                config[
                    "model"
                ][
                    "convnext"
                ][
                    "name"
                ]
            )
        )

        # ----------------------------------------------------
        # FORENSIC EXPERT
        # ----------------------------------------------------

        self.highpass_network = (
            ForensicCNN(
                dimension
            )
        )

        self.frequency_network = (
            ForensicCNN(
                dimension
            )
        )

        # ----------------------------------------------------
        # PROJECT PRETRAINED FEATURES
        # ----------------------------------------------------

        self.dino_projection = (
            Projection(
                self.dino.output_dim,
                dimension
            )
        )

        self.conv_projection = (
            Projection(
                self.convnext.output_dim,
                dimension
            )
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        self.fusion = (
            FusionTransformer(
                dimension
            )
        )

        # ----------------------------------------------------
        # REAL / FAKE HEAD
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                dimension,
                256
            ),

            nn.LayerNorm(
                256
            ),

            nn.GELU(),

            nn.Dropout(
                config["model"][
                    "dropout"
                ]
            ),

            nn.Linear(
                256,
                2
            )
        )

        # ----------------------------------------------------
        # GENERATOR HEAD
        #
        # Auxiliary task.
        #
        # It forces the representation to learn
        # generator-specific characteristics.
        # ----------------------------------------------------

        self.generator_classifier = (
            nn.Linear(
                dimension,
                num_generators
            )
        )

    def forward(
        self,
        rgb,
        highpass,
        frequency
    ):

        # ----------------------------------------------------
        # DINO
        # ----------------------------------------------------

        if any(p.requires_grad for p in self.dino.parameters()):
            dino_features = self.dino(rgb)
        else:
            with torch.no_grad():
                dino_features = self.dino(rgb)

        dino_features = (
            self.dino_projection(
                dino_features
            )
        )

        # ----------------------------------------------------
        # CONVNEXT
        # ----------------------------------------------------

        if any(p.requires_grad for p in self.convnext.parameters()):
            conv_features = self.convnext(rgb)
        else:
            with torch.no_grad():
                conv_features = self.convnext(rgb)

        conv_features = (
            self.conv_projection(
                conv_features
            )
        )

        # ----------------------------------------------------
        # HIGH-PASS
        # ----------------------------------------------------

        highpass_features = (
            self.highpass_network(
                highpass
            )
        )

        # ----------------------------------------------------
        # FREQUENCY
        # ----------------------------------------------------

        frequency_features = (
            self.frequency_network(
                frequency
            )
        )

        # ----------------------------------------------------
        # FOUR EXPERTS
        # ----------------------------------------------------

        tokens = torch.stack(
            [
                dino_features,
                conv_features,
                highpass_features,
                frequency_features
            ],
            dim=1
        )

        # ----------------------------------------------------
        # ATTENTION FUSION
        # ----------------------------------------------------

        fused = (
            self.fusion(
                tokens
            )
        )

        # ----------------------------------------------------
        # OUTPUT
        # ----------------------------------------------------

        binary_logits = (
            self.classifier(
                fused
            )
        )

        generator_logits = (
            self.generator_classifier(
                fused
            )
        )

        return {
            "binary":
                binary_logits,

            "generator":
                generator_logits
        }


# ============================================================
# FREEZE
# ============================================================

def freeze_backbones(
    model
):

    for parameter in (
        model.dino.parameters()
    ):

        parameter.requires_grad = False

    for parameter in (
        model.convnext.parameters()
    ):

        parameter.requires_grad = False


# ============================================================
# UNFREEZE LAST PART
# ============================================================

def unfreeze_last_layers(
    model
):

    print(
        "\nUnfreezing final pretrained layers..."
    )

    # --------------------------------------------------------
    # DINO
    # --------------------------------------------------------

    dino = model.dino.model

    if hasattr(
        dino,
        "encoder"
    ):

        layers = (
            dino.encoder.layer
        )

        for parameter in (
            layers[-1].parameters()
        ):

            parameter.requires_grad = True

    # --------------------------------------------------------
    # ConvNeXt
    # --------------------------------------------------------

    convnext = (
        model.convnext.model
    )

    if hasattr(
        convnext,
        "encoder"
    ):

        stages = (
            convnext.encoder.stages
        )

        for parameter in (
            stages[-1].parameters()
        ):

            parameter.requires_grad = True


# ============================================================
# OPTIMIZER
# ============================================================

def create_optimizer(
    model
):

    backbone_parameters = []

    other_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:

            continue

        if (
            name.startswith(
                "dino."
            )
            or
            name.startswith(
                "convnext."
            )
        ):

            backbone_parameters.append(
                parameter
            )

        else:

            other_parameters.append(
                parameter
            )

    optimizer = (
        torch.optim.AdamW(
            [
                {
                    "params":
                        other_parameters,

                    "lr":
                        CFG[
                            "training"
                        ][
                            "head_learning_rate"
                        ]
                },

                {
                    "params":
                        backbone_parameters,

                    "lr":
                        CFG[
                            "training"
                        ][
                            "backbone_learning_rate"
                        ]
                }
            ],

            weight_decay=
                CFG[
                    "training"
                ][
                    "weight_decay"
                ],
            foreach=False
        )
    )

    return optimizer


# ============================================================
# LOSS
# ============================================================

def calculate_loss(
    outputs,
    labels,
    generators
):

    binary_loss = F.cross_entropy(
        outputs["binary"],
        labels
    )

    generator_loss = F.cross_entropy(
        outputs["generator"],
        generators
    )

    total_loss = (
        binary_loss
        +
        CFG["training"][
            "generator_loss_weight"
        ]
        *
        generator_loss
    )

    return (
        total_loss,
        binary_loss,
        generator_loss
    )


# ============================================================
# CHECKPOINT SAVE
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    step,
    global_step,
    best_auc
):

    state = {

        "model":
            model.state_dict(),

        "optimizer":
            optimizer.state_dict() if optimizer is not None else None,

        "scheduler":
            scheduler.state_dict() if scheduler is not None else None,

        "epoch":
            epoch,

        "step":
            step,

        "global_step":
            global_step,

        "best_auc":
            best_auc,

        "seed":
            SEED,
    }

    temporary_file = (
        str(
            LATEST_CHECKPOINT
        )
        +
        ".tmp"
    )

    torch.save(
        state,
        temporary_file
    )

    # Safety: preserve existing checkpoint if it has higher progress or create backup
    if LATEST_CHECKPOINT.exists():
        try:
            import shutil
            disk_meta = torch.load(LATEST_CHECKPOINT, map_location="cpu")
            disk_step = disk_meta.get("global_step", disk_meta.get("step", 0))
            if disk_step > global_step:
                safety_file = LATEST_CHECKPOINT.parent / f"latest_step_{disk_step}.pt"
                print(f"[Safety Backup] Checkpoint on disk has higher step ({disk_step} > {global_step}). Preserving to: {safety_file}")
                shutil.copy2(LATEST_CHECKPOINT, safety_file)
            shutil.copy2(LATEST_CHECKPOINT, str(LATEST_CHECKPOINT) + ".bak")
        except Exception:
            pass

    os.replace(
        temporary_file,
        LATEST_CHECKPOINT
    )


# ============================================================
# CHECKPOINT LOAD
# ============================================================

def load_checkpoint(
    model,
    optimizer,
    scheduler,
    resume_path=None
):

    target_ckpt = None
    if resume_path:
        p = Path(resume_path)
        if p.exists():
            target_ckpt = p
        else:
            print(f"Warning: Specified --resume checkpoint not found: {resume_path}")

    if target_ckpt is None and LATEST_CHECKPOINT.exists():
        target_ckpt = LATEST_CHECKPOINT

    # Google Drive auto-discovery: check common Drive locations if LATEST_CHECKPOINT is not found
    if target_ckpt is None and Path("/content/drive/MyDrive").exists():
        drive_candidates = []
        search_dirs = [
            Path("/content/drive/MyDrive/ForensicFusion/checkpoints"),
            Path("/content/drive/MyDrive/ForensicFusion"),
            Path("/content/drive/MyDrive/ForensicFusion_Models"),
            Path("/content/drive/MyDrive/shared-with-me/ForensicFusion/checkpoints"),
            Path("/content/drive/MyDrive/shared-with-me/ForensicFusion"),
            Path("./checkpoints"),
        ]
        for s_dir in search_dirs:
            if s_dir.exists():
                for pattern in ["latest.pt", "best.pt", "checkpoints/latest.pt", "checkpoints/best.pt"]:
                    found = list(s_dir.glob(pattern))
                    drive_candidates.extend(found)
        unique_candidates = []
        for c in drive_candidates:
            if c not in unique_candidates and c.exists() and c.is_file():
                unique_candidates.append(c)
        if unique_candidates:
            target_ckpt = unique_candidates[0]
            print(f"Auto-detected existing checkpoint from Google Drive: {target_ckpt}")

    # Kaggle auto-discovery: check /kaggle/input for previous checkpoint runs attached as inputs
    if target_ckpt is None and Path("/kaggle/input").exists():
        candidates = []
        for p in Path("/kaggle/input").glob("*"):
            if "artifact-dataset" in p.name.lower():
                continue
            for pattern in ["latest.pt", "best.pt", "checkpoints/latest.pt", "checkpoints/best.pt", "*/latest.pt", "*/best.pt"]:
                candidates.extend(p.glob(pattern))
        if candidates:
            target_ckpt = sorted(candidates)[0]
            print(f"Auto-detected existing checkpoint from Kaggle input: {target_ckpt}")

    if target_ckpt is None or not target_ckpt.exists():
        print()
        print("=" * 80)
        print("NOTICE: NO EXISTING CHECKPOINT FOUND - STARTING FRESH FROM EPOCH 1")
        print(f"Expected checkpoint path: {LATEST_CHECKPOINT}")
        if Path("/content/drive/MyDrive").exists():
            print("\nIf you shared a Google Drive folder from another account, note that Colab")
            print("cannot see folders in 'Shared with me' until you add a shortcut:")
            print("  1. In Google Drive (new account), go to 'Shared with me'")
            print("  2. Right-click the folder -> 'Organize' -> 'Add shortcut' -> 'My Drive'")
        print("=" * 80)
        print()
        return (
            1,
            0,
            0,
            0.0
        )

    print()
    print("=" * 80)
    print(
        f"RESUMING FROM CHECKPOINT: {target_ckpt}"
    )
    print("=" * 80)

    checkpoint = torch.load(
        target_ckpt,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )

    if optimizer is not None and checkpoint.get("optimizer") is not None:
        try:
            optimizer.load_state_dict(
                checkpoint[
                    "optimizer"
                ]
            )
        except Exception as e:
            print("Notice: Could not load optimizer state:", e)

    if scheduler is not None and checkpoint.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(
                checkpoint[
                    "scheduler"
                ]
            )
        except Exception as e:
            print("Notice: Could not load scheduler state:", e)

    epoch = (
        checkpoint.get(
            "epoch",
            1
        )
    )

    step = (
        checkpoint.get(
            "step",
            0
        )
    )

    global_step = (
        checkpoint.get(
            "global_step",
            0
        )
    )

    best_auc = (
        checkpoint.get(
            "best_auc",
            0.0
        )
    )

    print(
        "Resumed epoch:",
        epoch
    )

    print(
        "Resumed step in epoch:",
        step
    )

    print(
        "Global step:",
        global_step
    )

    print(
        "Best AUC:",
        best_auc
    )

    print("=" * 80)

    return (
        epoch,
        step,
        global_step,
        best_auc
    )


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    epoch,
    global_step,
    best_auc,
    start_step=0,
    total_steps=None,
    scaler=None,
    scheduler=None
):

    model.train()

    is_cuda = torch.cuda.is_available() and str(DEVICE).startswith("cuda")
    if scaler is None and is_cuda:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        except Exception:
            scaler = None

    accumulation = (
        CFG[
            "hardware"
        ][
            "gradient_accumulation_steps"
        ]
    )

    optimizer.zero_grad()

    running_loss = 0.0

    if total_steps is None:
        total_steps = len(loader) + start_step

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}",
        initial=start_step,
        total=total_steps
    )

    last_step = start_step

    try:
        for i, batch in enumerate(
            progress
        ):
            step = start_step + i
            last_step = step

            rgb = batch[
                "rgb"
            ].to(DEVICE)

            highpass = batch[
                "highpass"
            ].to(DEVICE)

            frequency = batch[
                "frequency"
            ].to(DEVICE)

            labels = batch[
                "label"
            ].to(DEVICE)

            generators = batch[
                "generator"
            ].to(DEVICE)

            # ----------------------------------------------------
            # Forward
            # ----------------------------------------------------

            if is_cuda:
                with torch.amp.autocast("cuda", enabled=True):
                    outputs = model(
                        rgb,
                        highpass,
                        frequency
                    )

                    loss, binary_loss, generator_loss = (
                        calculate_loss(
                            outputs,
                            labels,
                            generators
                        )
                    )

                    scaled_loss = (
                        loss
                        /
                        accumulation
                    )

                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            else:
                outputs = model(
                    rgb,
                    highpass,
                    frequency
                )

                loss, binary_loss, generator_loss = (
                    calculate_loss(
                        outputs,
                        labels,
                        generators
                    )
                )

                scaled_loss = (
                    loss
                    /
                    accumulation
                )

                scaled_loss.backward()

            # ----------------------------------------------------
            # Backward & Optimization
            # ----------------------------------------------------

            if (step + 1) % accumulation == 0:

                if is_cuda and scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        CFG[
                            "training"
                        ][
                            "gradient_clip"
                        ]
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        CFG[
                            "training"
                        ][
                            "gradient_clip"
                        ]
                    )
                    optimizer.step()

                optimizer.zero_grad()

            global_step += 1

            running_loss += loss.item()

            progress.set_postfix({
                "loss":
                    f"{loss.item():.4f}"
            })

            # ----------------------------------------------------
            # Checkpoint
            # ----------------------------------------------------

            if (
                global_step
                %
                CFG[
                    "training"
                ][
                    "checkpoint_every_steps"
                ]
                == 0
            ):

                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    step + 1,
                    global_step,
                    best_auc
                )

                print(
                    f"\nCheckpoint saved at Step {step + 1} / Global Step {global_step}."
                )

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! Saving checkpoint at Step {last_step + 1}...")
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            last_step + 1,
            global_step,
            best_auc
        )
        print(f"Checkpoint saved to {LATEST_CHECKPOINT}.")
        raise

    # --------------------------------------------------------
    # Handle leftover gradients
    # --------------------------------------------------------

    if (
        (len(loader) + start_step)
        %
        accumulation
        != 0
    ):

        if is_cuda and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                CFG[
                    "training"
                ][
                    "gradient_clip"
                ]
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                CFG[
                    "training"
                ][
                    "gradient_clip"
                ]
            )
            optimizer.step()

        optimizer.zero_grad()

    return (
        running_loss
        /
        max(1, len(loader)),

        global_step
    )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader
):

    model.eval()

    labels = []

    probabilities = []

    for batch in tqdm(
        loader,
        desc="Validation"
    ):

        rgb = batch[
            "rgb"
        ].to(DEVICE)

        highpass = batch[
            "highpass"
        ].to(DEVICE)

        frequency = batch[
            "frequency"
        ].to(DEVICE)

        outputs = model(
            rgb,
            highpass,
            frequency
        )

        probability = (
            torch.softmax(
                outputs["binary"],
                dim=1
            )[
                :,
                1
            ]
            .detach()
            .cpu()
            .numpy()
        )

        probabilities.extend(
            probability.tolist()
        )

        labels.extend(
            batch[
                "label"
            ].numpy().tolist()
        )

    labels = np.asarray(
        labels
    )

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(
        np.int64
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0
        )
    )

    try:

        auc = roc_auc_score(
            labels,
            probabilities
        )

    except Exception:

        auc = 0.0

    return {

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "auc":
            float(auc)
    }


# ============================================================
# LOGGING
# ============================================================

def log_metrics(
    epoch,
    train_loss,
    metrics
):

    row = {

        "epoch":
            epoch,

        "train_loss":
            train_loss,

        **metrics
    }

    dataframe = pd.DataFrame(
        [row]
    )

    if LOG_FILE.exists():

        dataframe.to_csv(
            LOG_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        dataframe.to_csv(
            LOG_FILE,
            index=False
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("ARTIFACT PRODUCTION FORENSIC DETECTOR")
    print("=" * 80)
    print()
    print("Device:", DEVICE)
    print()

    # ========================================================
    # STEP 1
    # DATA DISCOVERY
    # ========================================================

    dataframe = get_index()

    # ========================================================
    # STEP 2
    # SPLITS
    # ========================================================

    dataframe = create_splits(
        dataframe
    )

    train_dataframe = dataframe[
        dataframe["split"]
        ==
        "train"
    ].reset_index(
        drop=True
    )

    validation_dataframe = dataframe[
        dataframe["split"]
        ==
        "validation"
    ].reset_index(
        drop=True
    )

    test_dataframe = dataframe[
        dataframe["split"]
        ==
        "test"
    ].reset_index(
        drop=True
    )

    print()
    print("=" * 80)
    print("DATASET")
    print("=" * 80)

    print(
        "Train:",
        len(train_dataframe)
    )

    print(
        "Validation:",
        len(validation_dataframe)
    )

    print(
        "Test:",
        len(test_dataframe)
    )

    print(
        "Total:",
        len(dataframe)
    )

    print(
        "Generators:",
        dataframe[
            "generator_id"
        ].nunique()
    )

    # ========================================================
    # DATASETS
    # ========================================================

    image_size = (
        CFG[
            "dataset"
        ][
            "image_size"
        ]
    )

    train_dataset = ArtifactDataset(
        train_dataframe,
        image_size,
        training=True
    )

    validation_dataset = ArtifactDataset(
        validation_dataframe,
        image_size,
        training=False
    )

    # ========================================================
    # DATALOADERS
    # ========================================================

    batch_size = (
        CFG[
            "hardware"
        ][
            "batch_size"
        ]
    )

    workers = (
        CFG[
            "hardware"
        ][
            "num_workers"
        ]
    )

    pin_mem = str(DEVICE).startswith("cuda")
    persistent = (workers > 0)
    prefetch = 2 if workers > 0 else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        prefetch_factor=prefetch
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_mem,
        persistent_workers=persistent,
        prefetch_factor=prefetch
    )

    # ========================================================
    # MODEL
    # ========================================================

    num_generators = int(
        dataframe[
            "generator_id"
        ].nunique()
    )

    model = ArtifactDetector(
        CFG,
        num_generators
    )

    model.to(
        DEVICE
    )

    # ========================================================
    # INITIAL FREEZE
    # ========================================================

    freeze_backbones(
        model
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = create_optimizer(
        model
    )

    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CFG[
                "training"
            ][
                "epochs"
            ]
        )
    )

    # ========================================================
    # RESUME
    # ========================================================

    start_epoch, start_step, global_step, best_auc = (
        load_checkpoint(
            model,
            optimizer,
            scheduler,
            resume_checkpoint_path
        )
    )

    current_start_step = start_step

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        start_epoch,
        CFG[
            "training"
        ][
            "epochs"
        ] + 1
    ):

        print()
        print("=" * 80)
        print(
            f"EPOCH {epoch}"
        )
        print("=" * 80)

        # ----------------------------------------------------
        # Progressive fine tuning
        # ----------------------------------------------------

        if (
            epoch
            >
            CFG[
                "training"
            ][
                "freeze_backbones_epochs"
            ]
        ):

            unfreeze_last_layers(
                model
            )

            optimizer = (
                create_optimizer(
                    model
                )
            )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        epoch_train_loader = train_loader
        samples_per_epoch = CFG.get("training", {}).get("samples_per_epoch", None)
        if (
            samples_per_epoch is not None
            and isinstance(samples_per_epoch, int)
            and 0 < samples_per_epoch < len(train_dataframe)
        ):
            epoch_train_df = train_dataframe.sample(
                n=samples_per_epoch,
                random_state=SEED + epoch
            ).reset_index(drop=True)
            epoch_train_dataset = ArtifactDataset(
                epoch_train_df,
                image_size,
                training=True
            )
            epoch_train_loader = DataLoader(
                epoch_train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=workers,
                pin_memory=pin_mem,
                persistent_workers=persistent,
                prefetch_factor=prefetch
            )

        total_epoch_steps = len(epoch_train_loader)

        if current_start_step > 0:
            if current_start_step < total_epoch_steps:
                print(f"\nResuming Epoch {epoch} at batch {current_start_step} / {total_epoch_steps}...")
                dataset = epoch_train_loader.dataset
                g = torch.Generator()
                g.manual_seed(SEED + epoch)
                perm = torch.randperm(len(dataset), generator=g).tolist()
                skip_samples = current_start_step * batch_size
                remaining_indices = perm[skip_samples:]
                from torch.utils.data import Subset
                remaining_dataset = Subset(dataset, remaining_indices)
                epoch_train_loader = DataLoader(
                    remaining_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=workers,
                    pin_memory=pin_mem,
                    persistent_workers=persistent,
                    prefetch_factor=prefetch
                )
            else:
                current_start_step = 0

        try:
            train_loss, global_step = (
                train_one_epoch(
                    model,
                    epoch_train_loader,
                    optimizer,
                    epoch,
                    global_step,
                    best_auc,
                    start_step=current_start_step,
                    total_steps=total_epoch_steps,
                    scheduler=scheduler
                )
            )
            current_start_step = 0
        except KeyboardInterrupt:
            print("\nTraining session paused. Your progress is saved in checkpoints/latest.pt.")
            return

        print()
        print(
            "Training loss:",
            train_loss
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        epoch_val_loader = validation_loader
        val_samples = CFG.get("training", {}).get("validation_samples", None)
        if (
            val_samples is not None
            and isinstance(val_samples, int)
            and 0 < val_samples < len(validation_dataframe)
        ):
            epoch_val_df = validation_dataframe.sample(
                n=val_samples,
                random_state=SEED
            ).reset_index(drop=True)
            epoch_val_dataset = ArtifactDataset(
                epoch_val_df,
                image_size,
                training=False
            )
            epoch_val_loader = DataLoader(
                epoch_val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=workers,
                pin_memory=False
            )

        metrics = validate(
            model,
            epoch_val_loader
        )

        print()
        print(
            "Validation results:"
        )

        for key, value in (
            metrics.items()
        ):

            print(
                f"{key:12s}: "
                f"{value:.5f}"
            )

        # ----------------------------------------------------
        # Log
        # ----------------------------------------------------

        log_metrics(
            epoch,
            train_loss,
            metrics
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if (
            metrics["auc"]
            >
            best_auc
        ):

            best_auc = (
                metrics["auc"]
            )

            torch.save(
                {
                    "model":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "auc":
                        best_auc,

                    "generators":
                        num_generators,

                    "config":
                        CFG
                },

                BEST_CHECKPOINT
            )

            print()
            print(
                "*** NEW BEST MODEL ***"
            )

            print(
                "AUC:",
                best_auc
            )

        # ----------------------------------------------------
        # Latest checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch + 1,
            0,
            global_step,
            best_auc
        )

        scheduler.step()

        gc.collect()

        print()
        print(
            "Checkpoint saved."
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 80)
    print("TRAINING FINISHED")
    print("=" * 80)

    print(
        "Best validation AUC:",
        best_auc
    )

    print(
        "Best model:",
        BEST_CHECKPOINT
    )


if __name__ == "__main__":

    import multiprocessing
    multiprocessing.freeze_support()

    main()