import os
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

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import torch_directml

from torchvision import transforms

from transformers import (
    AutoModel,
    ConvNextV2Model,
)


# ============================================================
# CONFIG
# ============================================================

with open("config.yaml", "r") as f:
    CFG = yaml.safe_load(f)


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

DEVICE = torch_directml.device()

print()
print("=" * 80)
print("ARTIFACT PRODUCTION FORENSIC DETECTOR")
print("=" * 80)
print()
print("Device:", DEVICE)
print()


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

    files = sorted(
        DATASET_ROOT.rglob(
            "metadata.csv"
        )
    )

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

    # Numeric labels
    if isinstance(
        value,
        (int, np.integer)
    ):

        if int(value) in [0, 1]:

            return int(value)

    if isinstance(
        value,
        (float, np.floating)
    ):

        if int(value) in [0, 1]:

            return int(value)

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
    ]

    real_keywords = [
        "real",
        "original",
        "authentic",
        "natural",
        "human",
        "coco",
        "imagenet",
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
    metadata_file
):

    raw_path = str(
        raw_path
    ).strip()

    if not raw_path:

        return None

    path = Path(
        raw_path
    )

    candidates = []

    # Absolute path
    if path.is_absolute():

        candidates.append(
            path
        )

    # Relative to CSV
    candidates.append(
        metadata_file.parent / path
    )

    # Relative to dataset root
    candidates.append(
        DATASET_ROOT / path
    )

    # Sometimes metadata stores "./..."
    candidates.append(
        metadata_file.parent
        / raw_path.lstrip("./")
    )

    for candidate in candidates:

        try:

            candidate = (
                candidate
                .resolve()
            )

        except Exception:

            continue

        if candidate.is_file():

            return candidate

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
            metadata_file
        )

    except Exception as error:

        print(
            "FAILED:",
            error
        )

        return [], {
            "metadata_file":
                str(metadata_file),

            "error":
                str(error)
        }

    if dataframe.empty:

        return [], {
            "metadata_file":
                str(metadata_file),

            "rows":
                0
        }

    print(
        "Columns:",
        list(dataframe.columns)
    )

    # --------------------------------------------------------
    # IMAGE COLUMN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # LABEL COLUMN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # GENERATOR/SOURCE COLUMN
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CATEGORY
    # --------------------------------------------------------

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

        return [], {
            "metadata_file":
                str(metadata_file),

            "error":
                "No image column"
        }

    # --------------------------------------------------------
    # Generator fallback
    # --------------------------------------------------------

    fallback_generator = (
        metadata_file.parent.name
    )

    if (
        fallback_generator
        not in generator_map
    ):

        generator_map[
            fallback_generator
        ] = len(
            generator_map
        )

    rows = []

    missing = 0

    unresolved_labels = 0

    for row_number, row in (
        dataframe.iterrows()
    ):

        image_path = (
            resolve_image_path(
                row[
                    image_column
                ],
                metadata_file
            )
        )

        if image_path is None:

            missing += 1

            continue

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = None

        if label_column is not None:

            label = normalize_label(
                row[
                    label_column
                ]
            )

        # Fallback: infer from path
        if label is None:

            label = (
                infer_label_from_path(
                    image_path
                )
            )

        if label is None:

            unresolved_labels += 1

            continue

        # ----------------------------------------------------
        # Generator
        # ----------------------------------------------------

        if generator_column is not None:

            generator = str(
                row[
                    generator_column
                ]
            ).strip()

            if (
                not generator
                or generator.lower()
                == "nan"
            ):

                generator = (
                    fallback_generator
                )

        else:

            generator = (
                fallback_generator
            )

        if (
            generator
            not in generator_map
        ):

            generator_map[
                generator
            ] = len(
                generator_map
            )

        generator_id = (
            generator_map[
                generator
            ]
        )

        # ----------------------------------------------------
        # Category
        # ----------------------------------------------------

        if category_column:

            category = str(
                row[
                    category_column
                ]
            )

        else:

            category = (
                "unknown"
            )

        # ----------------------------------------------------
        # Stable ID
        # ----------------------------------------------------

        sample_id = hashlib.sha1(
            str(
                image_path
            ).encode(
                "utf-8"
            )
        ).hexdigest()

        rows.append({

            "sample_id":
                sample_id,

            "image_path":
                str(image_path),

            "target":
                int(label),

            "generator":
                generator,

            "generator_id":
                generator_id,

            "category":
                category,

            "metadata_file":
                str(metadata_file),

            "dataset_folder":
                str(
                    metadata_file.parent
                ),
        })

    report = {

        "metadata_file":
            str(metadata_file),

        "rows":
            len(dataframe),

        "valid_rows":
            len(rows),

        "missing_images":
            missing,

        "unresolved_labels":
            unresolved_labels,

        "columns":
            list(dataframe.columns),
    }

    print(
        f"Rows: {len(dataframe)}"
    )

    print(
        f"Valid: {len(rows)}"
    )

    print(
        f"Missing images: {missing}"
    )

    print(
        f"Unresolved labels: "
        f"{unresolved_labels}"
    )

    return rows, report


# ============================================================
# BUILD GLOBAL INDEX
# ============================================================

def build_index():

    metadata_files = (
        discover_metadata_files()
    )

    generator_map = {}

    all_rows = []

    reports = []

    for metadata_file in metadata_files:

        rows, report = (
            process_metadata_file(
                metadata_file,
                generator_map
            )
        )

        all_rows.extend(
            rows
        )

        reports.append(
            report
        )

    if not all_rows:

        raise RuntimeError(
            "No valid images were discovered."
        )

    dataframe = pd.DataFrame(
        all_rows
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
    # Verify image readability
    # --------------------------------------------------------

    print()
    print(
        "Checking image integrity..."
    )

    bad_images = []

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

        print(
            "Existing dataset index found."
        )

        return pd.read_csv(
            INDEX_FILE
        )

    return build_index()


# ============================================================
# CREATE SPLITS
# ============================================================

def create_splits(
    dataframe
):

    if SPLIT_FILE.exists():

        print(
            "Existing dataset split found."
        )

        return pd.read_csv(
            SPLIT_FILE
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

                transforms.Resize(
                    (
                        image_size,
                        image_size
                    )
                ),

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

        try:

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

        except Exception as error:

            raise RuntimeError(
                f"Could not load:\n"
                f"{image_path}\n"
                f"{error}"
            )

        rgb = (
            self.rgb_transform(
                image
            )
        )

        highpass_image = (
            self.highpass(
                image
            )
        )

        highpass = (
            self.forensic_transform(
                highpass_image
            )
        )

        frequency_image = (
            self.frequency(
                image
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
            "Loading DINOv3..."
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

        dino_features = (
            self.dino(
                rgb
            )
        )

        dino_features = (
            self.dino_projection(
                dino_features
            )
        )

        # ----------------------------------------------------
        # CONVNEXT
        # ----------------------------------------------------

        conv_features = (
            self.convnext(
                rgb
            )
        )

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
                ]
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
            optimizer.state_dict(),

        "scheduler":
            scheduler.state_dict(),

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
    scheduler
):

    if not LATEST_CHECKPOINT.exists():

        return (
            1,
            0,
            0.0
        )

    print()
    print("=" * 80)
    print(
        "RESUMING FROM CHECKPOINT"
    )
    print("=" * 80)

    checkpoint = torch.load(
        LATEST_CHECKPOINT,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer"
        ]
    )

    scheduler.load_state_dict(
        checkpoint[
            "scheduler"
        ]
    )

    epoch = (
        checkpoint[
            "epoch"
        ]
    )

    global_step = (
        checkpoint[
            "global_step"
        ]
    )

    best_auc = (
        checkpoint[
            "best_auc"
        ]
    )

    print(
        "Last epoch:",
        epoch
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
    best_auc
):

    model.train()

    accumulation = (
        CFG[
            "hardware"
        ][
            "gradient_accumulation_steps"
        ]
    )

    optimizer.zero_grad()

    running_loss = 0.0

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}"
    )

    for step, batch in enumerate(
        progress
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

        labels = batch[
            "label"
        ].to(DEVICE)

        generators = batch[
            "generator"
        ].to(DEVICE)

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

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
        # Gradient accumulation
        # ----------------------------------------------------

        if (
            (step + 1)
            % accumulation
            == 0
        ):

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

        running_loss += (
            loss.item()
        )

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
                step,
                global_step,
                best_auc
            )

            print(
                "\nCheckpoint saved."
            )

    # --------------------------------------------------------
    # Handle leftover gradients
    # --------------------------------------------------------

    if (
        len(loader)
        %
        accumulation
        != 0
    ):

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
        len(loader),

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=False
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=False
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

    start_epoch, global_step, best_auc = (
        load_checkpoint(
            model,
            optimizer,
            scheduler
        )
    )

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

        train_loss, global_step = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                epoch,
                global_step,
                best_auc
            )
        )

        print()
        print(
            "Training loss:",
            train_loss
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        metrics = validate(
            model,
            validation_loader
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

    main()