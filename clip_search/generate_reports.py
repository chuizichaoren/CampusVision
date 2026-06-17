"""
Generate similarity_report.csv and retrieval_examples.md for Task B validation.

This script:
1. Loads clip_metadata.json produced by Task A to discover available class names.
2. For each class, constructs positive queries (same class) and negative queries
   (different class) and computes cosine similarity via the CLIP FAISS index.
3. Writes similarity_report.csv with one row per (query, image, label) triple.
4. Writes retrieval_examples.md illustrating top-k retrieval for sample queries.

Usage
-----
    python generate_reports.py [--index-dir processed/clip_index]
                               [--output-dir processed/clip_index]
                               [--top-k 5]
                               [--samples-per-class 5]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from transformers import CLIPModel, CLIPProcessor

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Query templates  (English text prompts sent to CLIP)
# ---------------------------------------------------------------------------

# Map each YOLO class name → a natural-language query string.
# We auto-generate these at runtime from the metadata; the ones below are
# manually tuned for campus-surveillance datasets.  If a class name is NOT
# listed here, a generic "a <class_name> in campus scene" template is used.
CLASS_QUERY_TEMPLATES: dict[str, str] = {
    "person":       "a person walking on campus",
    "car":          "a car parked on campus",
    "bicycle":      "a bicycle on campus",
    "motorcycle":   "a motorcycle on campus",
    "bus":          "a bus on campus road",
    "truck":        "a truck on campus",
    "dog":          "a dog on campus",
    "backpack":     "a person carrying a backpack",
    "umbrella":     "a person holding an umbrella",
    "handbag":      "a person carrying a handbag",
    "traffic light":"a traffic light",
    "bench":        "a bench on campus",
    "chair":        "a chair indoors",
    "laptop":       "a laptop computer",
    "cell phone":   "a person using a cell phone",
}


def _query_for_class(class_name: str) -> str:
    return CLASS_QUERY_TEMPLATES.get(class_name, f"a {class_name} in campus scene")


# ---------------------------------------------------------------------------
# Core routines
# ---------------------------------------------------------------------------


def _load_resources(
    index_dir: Path,
    model_name: str,
    device: str,
) -> tuple[faiss.Index, list[dict[str, Any]], CLIPProcessor, CLIPModel]:
    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "clip_metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(f"faiss.index not found at {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"clip_metadata.json not found at {metadata_path}")

    index = faiss.read_index(str(index_path))
    metadata: list[dict[str, Any]] = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )
    processor = CLIPProcessor.from_pretrained(model_name)
    model_obj = CLIPModel.from_pretrained(model_name).to(device)
    model_obj.eval()
    return index, metadata, processor, model_obj


def _encode_texts(
    texts: list[str],
    processor: CLIPProcessor,
    model_obj: CLIPModel,
    device: str,
) -> np.ndarray:
    """Return L2-normalised text embeddings, shape (N, D), float32."""
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model_obj.get_text_features(**inputs)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Similarity report
# ---------------------------------------------------------------------------


def build_similarity_report(
    index_dir: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    samples_per_class: int,
    top_k: int,
) -> pd.DataFrame:
    """
    For every class present in clip_metadata.json:
      - Draw up to *samples_per_class* image embeddings as positive examples.
      - Draw up to *samples_per_class* image embeddings from EACH other class
        as negative examples.
      - Compute cosine similarity between the class text query and all samples.

    Returns the full DataFrame (also saved to ``similarity_report.csv``).
    """
    index, metadata, processor, model_obj = _load_resources(
        index_dir, model_name, device
    )

    # Group metadata by class
    class_to_ids: dict[str, list[int]] = {}
    for entry in metadata:
        class_to_ids.setdefault(entry["class_name"], []).append(entry["id"])

    all_classes = sorted(class_to_ids.keys())
    LOGGER.info("Classes found: %s", all_classes)

    # Load ALL image embeddings once (they are already L2-normalised)
    n_vectors = index.ntotal
    embedding_dim = index.d
    all_embeddings = np.zeros((n_vectors, embedding_dim), dtype=np.float32)
    # Reconstruct via a dummy range search workaround: use IndexFlatIP directly
    # IndexFlatIP supports reconstruct() for exact retrieval.
    for vid in range(n_vectors):
        all_embeddings[vid] = index.reconstruct(vid)

    rows: list[dict[str, Any]] = []

    for class_name in all_classes:
        query_text = _query_for_class(class_name)

        # Encode the text query for this class
        text_vec = _encode_texts([query_text], processor, model_obj, device)  # (1, D)

        # Positive samples: images of the same class
        pos_ids = class_to_ids[class_name][:samples_per_class]

        # Negative samples: images from OTHER classes (up to samples_per_class each)
        neg_ids: list[int] = []
        for other_class in all_classes:
            if other_class == class_name:
                continue
            neg_ids.extend(class_to_ids[other_class][:samples_per_class])

        for vid in pos_ids:
            sim = float(np.dot(text_vec[0], all_embeddings[vid]))
            entry = metadata[vid]
            rows.append(
                {
                    "query_text": query_text,
                    "query_class": class_name,
                    "image_id": vid,
                    "image_class": entry["class_name"],
                    "crop_path": entry["crop_path"],
                    "cosine_similarity": round(sim, 6),
                    "match_type": "positive",  # same class
                    "frame_id": entry["frame_id"],
                    "timestamp": entry["timestamp"],
                }
            )

        for vid in neg_ids:
            sim = float(np.dot(text_vec[0], all_embeddings[vid]))
            entry = metadata[vid]
            rows.append(
                {
                    "query_text": query_text,
                    "query_class": class_name,
                    "image_id": vid,
                    "image_class": entry["class_name"],
                    "crop_path": entry["crop_path"],
                    "cosine_similarity": round(sim, 6),
                    "match_type": "negative",  # different class
                    "frame_id": entry["frame_id"],
                    "timestamp": entry["timestamp"],
                }
            )

    df = pd.DataFrame(rows)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "similarity_report.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Saved similarity report → %s  (%d rows)", csv_path, len(df))

    # Print summary statistics
    summary = (
        df.groupby(["query_class", "match_type"])["cosine_similarity"]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    LOGGER.info("\nSimilarity summary:\n%s", summary.to_string())

    return df


# ---------------------------------------------------------------------------
# Retrieval examples Markdown report
# ---------------------------------------------------------------------------


def build_retrieval_examples(
    index_dir: Path,
    output_dir: Path,
    model_name: str,
    device: str,
    top_k: int,
) -> None:
    """
    Run top-k retrieval for one representative query per detected class and
    write a Markdown report with tables showing retrieved results.
    """
    index, metadata, processor, model_obj = _load_resources(
        index_dir, model_name, device
    )

    class_to_ids: dict[str, list[int]] = {}
    for entry in metadata:
        class_to_ids.setdefault(entry["class_name"], []).append(entry["id"])

    all_classes = sorted(class_to_ids.keys())
    actual_k = min(top_k, index.ntotal)

    md_lines: list[str] = [
        "# CLIP Retrieval Examples",
        "",
        "This document shows top-k retrieval results for one representative text query",
        "per object class detected in the dataset.  Scores are cosine similarities",
        "computed by the inner-product FAISS index over L2-normalised CLIP embeddings.",
        "",
        f"- **Model**: `openai/clip-vit-base-patch32`",
        f"- **Index type**: `IndexFlatIP` (cosine similarity)",
        f"- **Top-k**: {actual_k}",
        f"- **Total images indexed**: {index.ntotal}",
        f"- **Classes**: {', '.join(all_classes)}",
        "",
        "---",
        "",
    ]

    for class_name in all_classes:
        query_text = _query_for_class(class_name)

        # Encode query
        text_vec = _encode_texts([query_text], processor, model_obj, device)

        # Search
        scores, indices = index.search(text_vec, actual_k)

        md_lines += [
            f"## Class: `{class_name}`",
            "",
            f"**Query**: *{query_text}*",
            "",
            f"| Rank | Score | Retrieved Class | Match | Crop Path | Frame | Timestamp |",
            f"|------|-------|-----------------|-------|-----------|-------|-----------|",
        ]

        for rank, (vid, score) in enumerate(
            zip(indices[0].tolist(), scores[0].tolist()), start=1
        ):
            if vid < 0:
                break
            entry = metadata[vid]
            retrieved_class = entry["class_name"]
            match_emoji = "✅" if retrieved_class == class_name else "❌"
            md_lines.append(
                f"| {rank} "
                f"| {score:.4f} "
                f"| `{retrieved_class}` "
                f"| {match_emoji} "
                f"| `{entry['crop_path']}` "
                f"| {entry['frame_id']} "
                f"| {entry['timestamp']} |"
            )

        # Compute precision@k for this query
        n_correct = sum(
            1
            for vid in indices[0].tolist()
            if vid >= 0 and metadata[vid]["class_name"] == class_name
        )
        precision_at_k = n_correct / actual_k if actual_k > 0 else 0.0

        md_lines += [
            "",
            f"> **Precision@{actual_k}** for `{class_name}`: "
            f"**{n_correct}/{actual_k}** = {precision_at_k:.2%}",
            "",
            "---",
            "",
        ]

    # Append cross-class similarity analysis section
    md_lines += [
        "## Cross-Class Similarity Analysis",
        "",
        "The table below summarises mean cosine similarity (positive vs negative pairs)",
        "across all class queries.  See `similarity_report.csv` for full row-level data.",
        "",
        "| Query Class | Positive Mean Sim | Negative Mean Sim | Δ (Gap) |",
        "|-------------|-------------------|-------------------|---------|",
    ]

    # Quick in-memory computation for the summary table
    n_vectors = index.ntotal
    all_embeddings = np.zeros((n_vectors, index.d), dtype=np.float32)
    for vid in range(n_vectors):
        all_embeddings[vid] = index.reconstruct(vid)

    samples = 10
    for class_name in all_classes:
        query_text = _query_for_class(class_name)
        text_vec = _encode_texts([query_text], processor, model_obj, device)

        pos_ids = class_to_ids[class_name][:samples]
        neg_ids = [
            vid
            for cn, ids in class_to_ids.items()
            if cn != class_name
            for vid in ids[:samples]
        ]

        pos_sims = [float(np.dot(text_vec[0], all_embeddings[v])) for v in pos_ids]
        neg_sims = [float(np.dot(text_vec[0], all_embeddings[v])) for v in neg_ids]

        pos_mean = np.mean(pos_sims) if pos_sims else float("nan")
        neg_mean = np.mean(neg_sims) if neg_sims else float("nan")
        gap = pos_mean - neg_mean if pos_sims and neg_sims else float("nan")

        md_lines.append(
            f"| `{class_name}` | {pos_mean:.4f} | {neg_mean:.4f} | {gap:+.4f} |"
        )

    md_lines += [
        "",
        "> A positive Δ (gap) confirms that the CLIP text query is semantically",
        "> closer to images of the correct class than to images of other classes.",
        "",
    ]

    # Write file
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "retrieval_examples.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    LOGGER.info("Saved retrieval examples → %s", md_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate similarity_report.csv and retrieval_examples.md.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index-dir",
        default="processed/clip_index",
        help="Directory containing faiss.index and clip_metadata.json (Task A output).",
    )
    parser.add_argument(
        "--output-dir",
        default="processed/clip_index",
        help="Directory to write similarity_report.csv and retrieval_examples.md.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve per query.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=10,
        help="Max number of image samples per class used in similarity report.",
    )
    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model name (must match Task A).",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
        stream=sys.stderr,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_dir = Path(args.index_dir)
    output_dir = Path(args.output_dir)

    LOGGER.info("Building similarity report …")
    build_similarity_report(
        index_dir=index_dir,
        output_dir=output_dir,
        model_name=args.model,
        device=device,
        samples_per_class=args.samples_per_class,
        top_k=args.top_k,
    )

    LOGGER.info("Building retrieval examples …")
    build_retrieval_examples(
        index_dir=index_dir,
        output_dir=output_dir,
        model_name=args.model,
        device=device,
        top_k=args.top_k,
    )

    LOGGER.info("Done.  Outputs written to %s", output_dir)


if __name__ == "__main__":
    main()