"""CLIP text-query search over a FAISS index built by build_index.py (Task A)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def search_by_text(
    query: str,
    index_dir: str = "processed/clip_index",
    top_k: int = 5,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
) -> list[dict[str, Any]]:
    """
    Encode *query* with CLIP and retrieve the top-k most similar images.

    The FAISS index built by Task-A uses **IndexFlatIP over L2-normalised**
    embeddings, which is equivalent to cosine similarity.  This function
    normalises the text embedding the same way before querying.

    Args:
        query:      Natural-language description, e.g. ``"a person walking"``.
        index_dir:  Directory produced by ``build_index.py`` (Task A).
                    Must contain ``faiss.index`` and ``clip_metadata.json``.
        top_k:      Number of results to return.
        model_name: Must match the model used in Task A
                    (default ``"openai/clip-vit-base-patch32"``).
        device:     Torch device string.  Auto-selects CUDA when available.

    Returns:
        List of result dicts, each containing::

            {
                "rank":        int,          # 1-based rank
                "score":       float,        # cosine similarity in [−1, 1]
                "id":          int,          # vector index in FAISS
                "crop_path":   str,
                "class_name":  str,
                "timestamp":   str,
                "confidence":  float,
                "frame_id":    int,
                "bbox":        [x1, y1, x2, y2],
            }

    Raises:
        FileNotFoundError: If ``faiss.index`` or ``clip_metadata.json`` is missing.
        ValueError:        If ``top_k`` is not a positive integer.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    index_root = Path(index_dir)
    index_path = index_root / "faiss.index"
    metadata_path = index_root / "clip_metadata.json"

    _assert_exists(index_path, "FAISS index")
    _assert_exists(metadata_path, "clip_metadata.json")

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Loading CLIP model %s on %s", model_name, selected_device)

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(selected_device)
    model.eval()

    # --- encode query text ---------------------------------------------------
    text_vector = _encode_text(
        query=query,
        processor=processor,
        model=model,
        device=selected_device,
    )  # shape (1, D), L2-normalised float32

    # --- load index & metadata -----------------------------------------------
    index = faiss.read_index(str(index_path))
    metadata: list[dict[str, Any]] = json.loads(metadata_path.read_text(encoding="utf-8"))

    # --- search --------------------------------------------------------------
    actual_k = min(top_k, index.ntotal)
    scores, indices = index.search(text_vector, actual_k)
    # scores: (1, k)  indices: (1, k)

    results: list[dict[str, Any]] = []
    for rank, (vector_id, score) in enumerate(
        zip(indices[0].tolist(), scores[0].tolist()), start=1
    ):
        if vector_id < 0:          # FAISS returns -1 when fewer results exist
            break
        entry = metadata[vector_id]
        results.append(
            {
                "rank": rank,
                "score": round(float(score), 6),
                "id": entry["id"],
                "crop_path": entry["crop_path"],
                "class_name": entry["class_name"],
                "timestamp": entry["timestamp"],
                "confidence": round(float(entry["confidence"]), 4),
                "frame_id": int(entry["frame_id"]),
                "bbox": entry["bbox"],
            }
        )

    LOGGER.info("Query '%s' → %d results (top score %.4f)", query, len(results), results[0]["score"] if results else 0.0)
    return results


def batch_search(
    queries: list[str],
    index_dir: str = "processed/clip_index",
    top_k: int = 5,
    model_name: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """
    Run :func:`search_by_text` for multiple queries in a single model load.

    Args:
        queries:    List of natural-language query strings.
        index_dir:  Same as in :func:`search_by_text`.
        top_k:      Number of results per query.
        model_name: Must match the model used in Task A.
        device:     Torch device string.

    Returns:
        Mapping ``{query_string: [result_dict, ...]}``.
    """
    if not queries:
        return {}
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    index_root = Path(index_dir)
    index_path = index_root / "faiss.index"
    metadata_path = index_root / "clip_metadata.json"

    _assert_exists(index_path, "FAISS index")
    _assert_exists(metadata_path, "clip_metadata.json")

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(selected_device)
    model.eval()

    index = faiss.read_index(str(index_path))
    metadata: list[dict[str, Any]] = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_k = min(top_k, index.ntotal)

    all_results: dict[str, list[dict[str, Any]]] = {}
    for query in queries:
        text_vector = _encode_text(
            query=query,
            processor=processor,
            model=model,
            device=selected_device,
        )
        scores, indices = index.search(text_vector, actual_k)

        results: list[dict[str, Any]] = []
        for rank, (vector_id, score) in enumerate(
            zip(indices[0].tolist(), scores[0].tolist()), start=1
        ):
            if vector_id < 0:
                break
            entry = metadata[vector_id]
            results.append(
                {
                    "rank": rank,
                    "score": round(float(score), 6),
                    "id": entry["id"],
                    "crop_path": entry["crop_path"],
                    "class_name": entry["class_name"],
                    "timestamp": entry["timestamp"],
                    "confidence": round(float(entry["confidence"]), 4),
                    "frame_id": int(entry["frame_id"]),
                    "bbox": entry["bbox"],
                }
            )
        all_results[query] = results
        LOGGER.info("Query '%s' → %d results", query, len(results))

    return all_results


# ---------------------------------------------------------------------------
# Similarity report helpers (used by generate_similarity_report.py)
# ---------------------------------------------------------------------------


def compute_text_image_similarity(
    query: str,
    crop_paths: list[str],
    model_name: str = "openai/clip-vit-base-patch32",
    device: str | None = None,
) -> list[float]:
    """
    Return cosine similarity between *query* and each image in *crop_paths*.

    Useful for building the similarity validation table.

    Args:
        query:       Text query string.
        crop_paths:  List of image file paths on disk.
        model_name:  CLIP model (must match Task A).
        device:      Torch device string.

    Returns:
        List of float cosine similarities in the same order as *crop_paths*.
    """
    from PIL import Image

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(selected_device)
    model.eval()

    # encode text
    text_inputs = processor(text=[query], return_tensors="pt", padding=True).to(selected_device)
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=1)

    # encode images
    images = [Image.open(p).convert("RGB") for p in crop_paths]
    image_inputs = processor(images=images, return_tensors="pt", padding=True).to(selected_device)
    with torch.no_grad():
        image_features = model.get_image_features(**image_inputs)
        image_features = torch.nn.functional.normalize(image_features, p=2, dim=1)

    # cosine similarity (dot product after normalisation)
    sims = (image_features @ text_features.T).squeeze(1)
    return [round(float(s), 6) for s in sims.cpu().tolist()]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _encode_text(
    query: str,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str,
) -> np.ndarray:
    """Return an L2-normalised text embedding as a (1, D) float32 array."""
    inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
    return features.cpu().numpy().astype(np.float32)


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found at: {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_cli_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Search CLIP FAISS index with a text query.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("query", type=str, help="Natural-language text query.")
    parser.add_argument(
        "--index-dir",
        default="processed/clip_index",
        help="Directory containing faiss.index and clip_metadata.json.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return.")
    parser.add_argument(
        "--model",
        default="openai/clip-vit-base-patch32",
        help="CLIP model name (must match Task A).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save results as JSON.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser


def _cli_main() -> None:
    import sys

    parser = _build_cli_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
        stream=sys.stderr,
    )

    results = search_by_text(
        query=args.query,
        index_dir=args.index_dir,
        top_k=args.top_k,
        model_name=args.model,
    )

    # pretty-print to stdout
    print(f"\n{'='*60}")
    print(f"Query : {args.query}")
    print(f"Top-{args.top_k} results")
    print(f"{'='*60}")
    for r in results:
        print(
            f"  #{r['rank']:>2}  score={r['score']:.4f}"
            f"  class={r['class_name']:<12}"
            f"  frame={r['frame_id']:<6}"
            f"  {r['crop_path']}"
        )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    _cli_main()
