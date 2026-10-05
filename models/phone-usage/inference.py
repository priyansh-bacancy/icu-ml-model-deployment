"""Archival SageMaker handler for the PPE-compatible model.tar package.

The live endpoint uses the custom FastAPI ECR image. This handler keeps the
five-file model archive self-contained and preserves the same frame-level JSON
contract if it is later used with a compatible SageMaker inference toolkit.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import boto3
import cv2
from ultralytics import YOLO

from builder import build_detections
from extractor import extract_result


MODEL_ID = "phone-usage-detection"
EVENT_TYPE = "phone_usage_event"
MODEL_VERSION = "v1.0.0"
BUCKET = os.environ.get(
    "FOOTAGE_BUCKET", "icu-solarcam-storage-bacancy-ap-southeast-2"
)
IMAGE_SIZE = 1280
CONFIDENCE = 0.25
S3 = boto3.client("s3")


def model_fn(model_dir: str) -> YOLO:
    return YOLO(os.path.join(model_dir, "best.pt"))


def input_fn(request_body: str | bytes, content_type: str = "application/json") -> dict:
    if content_type.split(";", 1)[0].lower() != "application/json":
        raise ValueError("Content-Type must be application/json")
    data = json.loads(request_body)
    if set(data) != {"job_id", "footage_s3_key"}:
        raise ValueError("request must contain exactly job_id and footage_s3_key")
    suffix = Path(data["footage_s3_key"]).suffix or ".mp4"
    temporary = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temporary.close()
    S3.download_file(BUCKET, data["footage_s3_key"], temporary.name)
    return {
        "job_id": data["job_id"],
        "footage_s3_key": data["footage_s3_key"],
        "local_path": temporary.name,
    }


def _video_detections(path: str, model: YOLO) -> tuple[list[dict], float]:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError("unable to decode video")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    detections: list[dict] = []
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(
                frame,
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False,
            )[0]
            detections.extend(extract_result(result, model.names, frame_number))
            frame_number += 1
    finally:
        capture.release()
    return detections, fps


def predict_fn(inputs: dict, model: YOLO) -> dict:
    started = time.perf_counter()
    local_path = inputs["local_path"]
    is_image = Path(local_path).suffix.lower() in {
        ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"
    }
    try:
        if is_image:
            result = model.predict(
                local_path,
                conf=CONFIDENCE,
                imgsz=IMAGE_SIZE,
                device="cpu",
                verbose=False,
            )[0]
            flat = extract_result(result, model.names, 0)
            fps = None
        else:
            flat, fps = _video_detections(local_path, model)
        return {
            "model_id": MODEL_ID,
            "event_type": EVENT_TYPE,
            "event_id": inputs["job_id"],
            "model_version": MODEL_VERSION,
            "status": "SUCCESS",
            "processing_time_ms": round((time.perf_counter() - started) * 1000),
            "output_media": {
                "annotated_video_s3_key": None,
                "annotated_image_s3_keys": [],
            },
            "detections": build_detections(flat, is_video=not is_image, fps=fps),
            "error": None,
        }
    except Exception as exc:
        return {
            "model_id": MODEL_ID,
            "event_type": EVENT_TYPE,
            "event_id": inputs["job_id"],
            "model_version": MODEL_VERSION,
            "status": "FAILED",
            "processing_time_ms": round((time.perf_counter() - started) * 1000),
            "output_media": {
                "annotated_video_s3_key": None,
                "annotated_image_s3_keys": [],
            },
            "detections": [],
            "error": {"code": "PROCESSING_ERROR", "message": str(exc)[:1000]},
        }
    finally:
        Path(local_path).unlink(missing_ok=True)


def output_fn(prediction: dict, accept: str = "application/json") -> tuple[str, str]:
    if accept.split(";", 1)[0].lower() != "application/json":
        raise ValueError("Accept must be application/json")
    return json.dumps(prediction), "application/json"
