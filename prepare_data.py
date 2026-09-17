from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from forensic_fusion.config import ensure_output_directories, load_config  # noqa: E402
from forensic_fusion.data import prepare_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the training, validation, and test manifest.")
    parser.add_argument("--config", default="config_kaggle.yaml", help="Path to the YAML configuration.")
    parser.add_argument("--force", action="store_true", help="Discard and rebuild an existing manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_directories(config)
    manifest = prepare_manifest(config, force=args.force)
    summary = manifest.groupby(["split", "target"]).size().unstack(fill_value=0)
    print("\nFinal manifest (target 0=real, 1=fake):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
