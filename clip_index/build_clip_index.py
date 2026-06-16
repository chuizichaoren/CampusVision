"""Build a CLIP image embedding FAISS index for YOLO object crops."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

LOGGER = logging.getLogger(__name__)

REQUIRED_DETECTION_COLUMNS = {
    "frame_id",
    "timestamp",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "crop_path",
}


def build_clip_faiss_index(
    crop_dir: str = "processed/object_crops",
    detection_csv: str = "processed/detections.csv",
    output_dir: str = "processed/clip_index",
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
    device: str | None = None,
) -> dict:
    """
    Read YOLO crop metadata, extract CLIP image embeddings, and build a FAISS index.

    The saved FAISS index uses inner product over L2-normalized CLIP image
    embeddings, which is equivalent to cosine similarity. The i-th vector in
    the FAISS index, the i-th row in ``image_embeddings.npy``, and the i-th
    object in ``clip_metadata.json`` are kept in the same order.

    Args:
        crop_dir: Directory containing YOLO object crop images.
        detection_csv: CSV file with YOLO detection metadata.
        output_dir: Directory where index, embeddings, metadata, and report are saved.
        model_name: Hugging Face CLIP model name.
        batch_size: Number of images encoded per CLIP forward pass.
        device: Torch device. If None, uses CUDA when available, otherwise CPU.

    Returns:
        The build report dictionary saved to ``build_index_report.json``.

    Raises:
        FileNotFoundError: If ``detection_csv`` or ``crop_dir`` does not exist.
        ValueError: If the CSV is missing required columns, no valid images are
            found, or ``batch_size`` is invalid.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")

    crop_root = Path(crop_dir)
    csv_path = Path(detection_csv)
    index_output_dir = Path(output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"detections.csv not found: {csv_path}")
    if not crop_root.exists():
        raise FileNotFoundError(f"Object crop directory not found: {crop_root}")

    detections = pd.read_csv(csv_path)
    _validate_detection_columns(detections)

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Loading CLIP model %s on %s", model_name, selected_device)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(selected_device)
    model.eval()

    valid_records, warnings = _collect_valid_records(
        detections=detections,
        crop_root=crop_root,
        csv_path=csv_path,
    )
    if not valid_records:
        raise ValueError(
            "No valid crop images were found. Check detections.csv crop_path "
            f"values and files under {crop_root}."
        )

    embeddings = _encode_images(
        records=valid_records,
        processor=processor,
        model=model,
        batch_size=batch_size,
        device=selected_device,
    )

    metadata = [
        _metadata_from_record(record=record, vector_id=idx)
        for idx, record in enumerate(valid_records)
    ]

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_output_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_output_dir / "faiss.index"
    embeddings_path = index_output_dir / "image_embeddings.npy"
    metadata_path = index_output_dir / "clip_metadata.json"
    report_path = index_output_dir / "build_index_report.json"

    faiss.write_index(index, str(index_path))
    np.save(embeddings_path, embeddings)
    _write_json(metadata_path, metadata)

    report = {
        "total_rows_in_detection_csv": int(len(detections)),
        "valid_images": int(len(valid_records)),
        "skipped_images": int(len(detections) - len(valid_records)),
        "embedding_dim": int(embeddings.shape[1]),
        "model_name": model_name,
        "index_type": "IndexFlatIP",
        "output_index_path": _as_posix_string(index_path),
        "output_metadata_path": _as_posix_string(metadata_path),
        "output_embeddings_path": _as_posix_string(embeddings_path),
        "output_report_path": _as_posix_string(report_path),
        "device": selected_device,
        "warnings": warnings,
    }
    _write_json(report_path, report)

    LOGGER.info(
        "Built CLIP FAISS index with %s valid images and %s skipped rows.",
        report["valid_images"],
        report["skipped_images"],
    )
    return report


def _validate_detection_columns(detections: pd.DataFrame) -> None:
    missing_columns = REQUIRED_DETECTION_COLUMNS.difference(detections.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        required = ", ".join(sorted(REQUIRED_DETECTION_COLUMNS))
        raise ValueError(
            f"detections.csv is missing required column(s): {missing}. "
            f"Required columns are: {required}."
        )


def _collect_valid_records(
    detections: pd.DataFrame,
    crop_root: Path,
    csv_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    valid_records: list[dict[str, Any]] = []
    warnings: list[str] = []

    for row_number, row in detections.iterrows():
        try:
            record = _record_from_row(row=row, crop_root=crop_root, csv_path=csv_path)
            _validate_image(record["resolved_crop_path"])
        except Exception as exc:
            warning = f"Skipping CSV row {row_number + 2}: {exc}"
            LOGGER.warning(warning)
            warnings.append(warning)
            continue

        valid_records.append(record)

    return valid_records, warnings


def _record_from_row(
    row: pd.Series,
    crop_root: Path,
    csv_path: Path,
) -> dict[str, Any]:
    raw_crop_path = _required_string(row["crop_path"], "crop_path")
    resolved_crop_path = _resolve_crop_path(raw_crop_path, crop_root, csv_path)
    if resolved_crop_path is None:
        raise FileNotFoundError(f"crop image not found: {raw_crop_path}")

    class_name = _required_string(row["class_name"], "class_name")
    frame_id = int(_required_number(row["frame_id"], "frame_id"))
    confidence = float(_required_number(row["confidence"], "confidence"))
    bbox = [
        int(round(_required_number(row["x1"], "x1"))),
        int(round(_required_number(row["y1"], "y1"))),
        int(round(_required_number(row["x2"], "x2"))),
        int(round(_required_number(row["y2"], "y2"))),
    ]

    return {
        "crop_path": _normalize_path_text(raw_crop_path),
        "resolved_crop_path": resolved_crop_path,
        "class_name": class_name,
        "timestamp": _required_string(row["timestamp"], "timestamp"),
        "confidence": confidence,
        "frame_id": frame_id,
        "bbox": bbox,
        "text_description": f"a {class_name} in campus scene",
        "text_description_zh": "",
    }


def _resolve_crop_path(
    raw_crop_path: str,
    crop_root: Path,
    csv_path: Path,
) -> Path | None:
    raw_path = Path(raw_crop_path)
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                raw_path,
                csv_path.parent / raw_path,
                crop_root / raw_path.name,
            ]
        )

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _validate_image(image_path: Path) -> None:
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"invalid or damaged image {image_path}: {exc}") from exc


def _encode_images(
    records: list[dict[str, Any]],
    processor: CLIPProcessor,
    model: CLIPModel,
    batch_size: int,
    device: str,
) -> np.ndarray:
    embedding_batches: list[np.ndarray] = []

    for start in tqdm(
        range(0, len(records), batch_size),
        desc="Encoding object crops with CLIP",
    ):
        batch_records = records[start : start + batch_size]
        images = [_load_rgb_image(record["resolved_crop_path"]) for record in batch_records]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        embedding_batches.append(features.cpu().numpy().astype(np.float32))

    return np.vstack(embedding_batches).astype(np.float32)


def _load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _metadata_from_record(record: dict[str, Any], vector_id: int) -> dict[str, Any]:
    return {
        "id": vector_id,
        "crop_path": record["crop_path"],
        "class_name": record["class_name"],
        "timestamp": record["timestamp"],
        "confidence": record["confidence"],
        "frame_id": record["frame_id"],
        "bbox": record["bbox"],
        "text_description": record["text_description"],
        "text_description_zh": record["text_description_zh"],
    }


def _required_string(value: Any, field_name: str) -> str:
    if pd.isna(value):
        raise ValueError(f"missing value for {field_name}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"empty value for {field_name}")
    return text


def _required_number(value: Any, field_name: str) -> float:
    if pd.isna(value):
        raise ValueError(f"missing value for {field_name}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid numeric value for {field_name}: {value}") from exc


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _normalize_path_text(path_text: str) -> str:
    return path_text.replace("\\", "/")


def _as_posix_string(path: Path) -> str:
    return path.as_posix()
