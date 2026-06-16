"""Command-line entry point for building the object-crop CLIP FAISS index."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clip_index.build_clip_index import build_clip_faiss_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a CLIP image embedding FAISS index from YOLO object crops."
    )
    parser.add_argument("--crop-dir", default="processed/object_crops")
    parser.add_argument("--detection-csv", default="processed/detections.csv")
    parser.add_argument("--output-dir", default="processed/clip_index")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. 'cuda' or 'cpu'. Defaults to CUDA when available.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    report = build_clip_faiss_index(
        crop_dir=args.crop_dir,
        detection_csv=args.detection_csv,
        output_dir=args.output_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
