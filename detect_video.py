"""Run YOLOv8 video detection and export visualized results.

Outputs:
  - annotated_video.mp4
  - annotated_frames/
  - object_crops/
  - detections.csv
  - detection_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO


LOGGER = logging.getLogger("detect_video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects in a video with YOLOv8 and save annotated outputs."
    )
    parser.add_argument(
        "video",
        help="Input video path, for example data/campus.mp4",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLO model weights path. Defaults to the bundled yolov8n.pt.",
    )
    parser.add_argument(
        "--output-dir",
        default="processed/yolo_detection",
        help="Directory for annotated video, frames, crops, and metadata.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="YOLO confidence threshold.",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="YOLO NMS IoU threshold.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch/YOLO device, such as cpu, mps, or cuda. Auto-selected if omitted.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Run detection every N frames. Use 1 for every frame.",
    )
    parser.add_argument(
        "--save-frame-stride",
        type=int,
        default=1,
        help="Save annotated image every N processed frames.",
    )
    parser.add_argument(
        "--no-crops",
        action="store_true",
        help="Do not save object crop images.",
    )
    parser.add_argument(
        "--show-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw class labels on annotated frames.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    report = detect_video(
        video_path=args.video,
        model_path=args.model,
        output_dir=args.output_dir,
        confidence=args.conf,
        iou=args.iou,
        device=args.device,
        frame_stride=args.frame_stride,
        save_frame_stride=args.save_frame_stride,
        save_crops=not args.no_crops,
        show_labels=args.show_labels,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def detect_video(
    video_path: str,
    model_path: str = "yolov8n.pt",
    output_dir: str = "processed/yolo_detection",
    confidence: float = 0.25,
    iou: float = 0.45,
    device: str | None = None,
    frame_stride: int = 1,
    save_frame_stride: int = 1,
    save_crops: bool = True,
    show_labels: bool = True,
) -> dict[str, Any]:
    """Detect objects in a video and save annotated video, frames, crops, and CSV."""

    if frame_stride <= 0:
        raise ValueError("frame_stride must be a positive integer.")
    if save_frame_stride <= 0:
        raise ValueError("save_frame_stride must be a positive integer.")

    input_video = Path(video_path)
    weights = Path(model_path)
    output_root = Path(output_dir)
    annotated_frames_dir = output_root / "annotated_frames"
    object_crops_dir = output_root / "object_crops"
    annotated_video_path = output_root / "annotated_video.mp4"
    detections_csv_path = output_root / "detections.csv"
    summary_path = output_root / "detection_summary.json"

    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not weights.exists():
        raise FileNotFoundError(f"YOLO model weights not found: {weights}")

    output_root.mkdir(parents=True, exist_ok=True)
    annotated_frames_dir.mkdir(parents=True, exist_ok=True)
    if save_crops:
        object_crops_dir.mkdir(parents=True, exist_ok=True)

    selected_device = device or _select_device()
    LOGGER.info("Loading YOLO model %s on %s", weights, selected_device)
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = _create_video_writer(annotated_video_path, fps / frame_stride, width, height)

    detection_rows: list[dict[str, Any]] = []
    class_counter: Counter[str] = Counter()
    processed_frames = 0
    saved_frames = 0

    LOGGER.info("Processing %s frames from %s", total_frames or "unknown", input_video)
    try:
        frame_id = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_id % frame_stride != 0:
                frame_id += 1
                continue

            timestamp = frame_id / fps
            result = model.predict(
                frame,
                conf=confidence,
                iou=iou,
                device=selected_device,
                verbose=False,
            )[0]

            annotated_frame = result.plot(labels=show_labels)
            writer.write(annotated_frame)

            if processed_frames % save_frame_stride == 0:
                frame_name = f"frame_{frame_id:06d}_{_format_timestamp(timestamp)}.jpg"
                cv2.imwrite(str(annotated_frames_dir / frame_name), annotated_frame)
                saved_frames += 1

            frame_rows = _rows_from_result(
                result=result,
                frame=frame,
                frame_id=frame_id,
                timestamp=timestamp,
                crop_dir=object_crops_dir,
                save_crops=save_crops,
            )
            for row in frame_rows:
                class_counter[row["class_name"]] += 1
            detection_rows.extend(frame_rows)

            processed_frames += 1
            frame_id += 1
    finally:
        cap.release()
        writer.release()

    _write_detections_csv(detections_csv_path, detection_rows)

    summary = {
        "input_video": _path_text(input_video),
        "model": _path_text(weights),
        "device": selected_device,
        "confidence": confidence,
        "iou": iou,
        "frame_stride": frame_stride,
        "save_frame_stride": save_frame_stride,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "saved_annotated_frames": saved_frames,
        "total_detections": len(detection_rows),
        "class_counts": dict(sorted(class_counter.items())),
        "annotated_video": _path_text(annotated_video_path),
        "annotated_frames_dir": _path_text(annotated_frames_dir),
        "object_crops_dir": _path_text(object_crops_dir) if save_crops else None,
        "detections_csv": _path_text(detections_csv_path),
        "summary_json": _path_text(summary_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    LOGGER.info("Annotated video saved to %s", annotated_video_path)
    LOGGER.info("Annotated frames saved to %s", annotated_frames_dir)
    LOGGER.info("Detection metadata saved to %s", detections_csv_path)
    return summary


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _create_video_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, max(fps, 1.0), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create annotated video writer: {output_path}")
    return writer


def _rows_from_result(
    result: Any,
    frame: Any,
    frame_id: int,
    timestamp: float,
    crop_dir: Path,
    save_crops: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    boxes = result.boxes
    if boxes is None:
        return rows

    for detection_id, box in enumerate(boxes):
        class_id = int(box.cls.item())
        class_name = result.names[class_id]
        confidence = float(box.conf.item())
        x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
        x1, y1, x2, y2 = _clip_bbox(x1, y1, x2, y2, frame.shape[1], frame.shape[0])

        crop_path = ""
        if save_crops and x2 > x1 and y2 > y1:
            class_dir = crop_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            crop_name = f"frame_{frame_id:06d}_det_{detection_id:03d}_{class_name}.jpg"
            crop_file = class_dir / crop_name
            cv2.imwrite(str(crop_file), frame[y1:y2, x1:x2])
            crop_path = _path_text(crop_file)

        rows.append(
            {
                "frame_id": frame_id,
                "timestamp": _format_timestamp(timestamp),
                "timestamp_seconds": round(timestamp, 3),
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 6),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop_path": crop_path,
            }
        )

    return rows


def _clip_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, min(x1, width - 1)),
        max(0, min(y1, height - 1)),
        max(0, min(x2, width)),
        max(0, min(y2, height)),
    )


def _write_detections_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "frame_id",
        "timestamp",
        "timestamp_seconds",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "crop_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    total_minutes = total_seconds // 60
    minute = total_minutes % 60
    hour = total_minutes // 60
    return f"{hour:02d}-{minute:02d}-{sec:02d}-{ms:03d}"


def _path_text(path: Path) -> str:
    return path.as_posix()


if __name__ == "__main__":
    main()
