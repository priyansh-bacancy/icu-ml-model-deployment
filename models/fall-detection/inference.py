import json
import logging
import os
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import boto3
import cv2
import torch
from ultralytics import YOLO


IMAGE_SIZE = 640
DEFAULT_CONFIDENCE = 0.25
MODEL_FILENAME = "best.pt"
MODEL_ID = "fall-detection"
MODEL_VERSION = "v7"
EVENT_TYPE = "fall_event"
EXPECTED_CLASS_NAMES = {0: "falling", 1: "sitting", 2: "standing"}
FRAME_SAMPLE_INTERVAL = 15
MAX_ANNOTATED_SNAPSHOTS = 5
SAMPLING_SCHEMA_VERSION = "1.0"
UPLOADS_PREFIX = "uploads/"
SUPPORTED_VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")

logger = logging.getLogger(__name__)


def _transcode_to_browser_mp4(raw_path: Path, final_path: Path) -> None:
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(raw_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-movflags",
                "+faststart",
                "-an",
                str(final_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "ffmpeg is required to create browser-compatible video"
        ) from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "unknown ffmpeg error").strip()
        raise RuntimeError(f"Could not transcode annotated video: {detail}") from error
    if not final_path.is_file() or final_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg did not create a valid annotated video")
    raw_path.unlink()


def model_fn(model_dir: str) -> dict:
    model_path = Path(model_dir) / MODEL_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    model = YOLO(model_path)
    if model.names != EXPECTED_CLASS_NAMES:
        raise ValueError(
            f"Expected model classes {EXPECTED_CLASS_NAMES}, found {model.names}"
        )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return {"model": model, "device": device}


def input_fn(request_body: bytes | str, content_type: str) -> dict:
    media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    if media_type != "application/json":
        raise ValueError("Fall-detection video inference accepts application/json only")

    body = (
        request_body.decode("utf-8")
        if isinstance(request_body, bytes)
        else request_body
    )
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ValueError("Request body is not valid JSON") from error

    required_fields = {"job_id", "footage_s3_key"}
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ValueError("Request must contain exactly 'job_id' and 'footage_s3_key'")
    job_id = payload["job_id"]
    footage_s3_key = payload["footage_s3_key"]
    if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("'job_id' contains unsupported characters")
    if not isinstance(footage_s3_key, str) or not footage_s3_key.strip("/"):
        raise ValueError("'footage_s3_key' must be a non-empty S3 key")
    if footage_s3_key.startswith("/") or ".." in Path(footage_s3_key).parts:
        raise ValueError("'footage_s3_key' must be a relative S3 key")
    if not footage_s3_key.startswith(UPLOADS_PREFIX):
        raise ValueError("'footage_s3_key' must start with 'uploads/'")
    if Path(footage_s3_key).suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise ValueError("'footage_s3_key' must reference a supported video")
    return {
        "job_id": job_id,
        "footage_s3_key": footage_s3_key,
    }


def _processing_time_ms(started_at: float) -> int:
    return round((time.perf_counter() - started_at) * 1000)


def _output_media(
    annotated_video_s3_key: str | None = None,
    annotated_image_s3_keys: list[str] | None = None,
) -> dict:
    return {
        "annotated_video_s3_key": annotated_video_s3_key,
        "annotated_image_s3_keys": annotated_image_s3_keys or [],
    }


def _result_block(
    job_id: str,
    status: str,
    processing_time_ms: int,
    output_media: dict,
    detections: list[dict],
    error: dict | None,
) -> dict:
    return {
        "model_id": MODEL_ID,
        "event_type": EVENT_TYPE,
        "event_id": job_id,
        "model_version": MODEL_VERSION,
        "status": status,
        "processing_time_ms": processing_time_ms,
        "output_media": output_media,
        "detections": detections,
        "error": error,
    }


def _job_key(job_id: str, filename: str) -> str:
    return f"jobs/{job_id}/{MODEL_ID}/{filename}"


def _persist_json(job_id: str, filename: str, value: dict) -> None:
    _, output_bucket = _bucket_configuration()
    boto3.client("s3").put_object(
        Bucket=output_bucket,
        Key=_job_key(job_id, filename),
        Body=json.dumps(value).encode("utf-8"),
        ContentType="application/json",
    )


def _extract_frame_records(result, frame_number: int) -> list[dict]:
    boxes = result.boxes
    if boxes is None:
        return []

    records = []
    coordinates = boxes.xyxy.detach().cpu().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    class_ids = boxes.cls.detach().cpu().to(torch.int64).tolist()
    for box, confidence, class_id in zip(
        coordinates,
        confidences,
        class_ids,
        strict=True,
    ):
        xmin, ymin, xmax, ymax = box
        records.append(
            {
                "frame": frame_number,
                "class_id": class_id,
                "label": str(result.names[class_id]),
                "confidence": round(float(confidence), 4),
                "bbox": {
                    "x": round(xmin),
                    "y": round(ymin),
                    "w": round(xmax - xmin),
                    "h": round(ymax - ymin),
                },
                "attributes": {},
            }
        )
    return records


def _prelabel_key(
    prelabel_prefix: str,
    frame_number: int,
) -> str:
    sample_name = f"frame_{frame_number:06d}"
    return f"{prelabel_prefix}/labels/{sample_name}.txt"


def _is_sampled_frame(frame_number: int) -> bool:
    return frame_number % FRAME_SAMPLE_INTERVAL == 0


def _yolo_label_line(record: dict, width: int, height: int) -> str:
    bbox = record["bbox"]
    raw_xmin = float(bbox["x"])
    raw_ymin = float(bbox["y"])
    raw_xmax = raw_xmin + float(bbox["w"])
    raw_ymax = raw_ymin + float(bbox["h"])
    xmin = max(0.0, min(float(width), raw_xmin))
    ymin = max(0.0, min(float(height), raw_ymin))
    xmax = max(xmin, min(float(width), raw_xmax))
    ymax = max(ymin, min(float(height), raw_ymax))
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    normalized_width = (xmax - xmin) / width
    normalized_height = (ymax - ymin) / height
    return (
        f"{record['class_id']} {x_center:.6f} {y_center:.6f} "
        f"{normalized_width:.6f} {normalized_height:.6f}"
    )


def _write_prelabel(
    frame_records: list[dict],
    label_path: Path,
    width: int,
    height: int,
) -> None:
    label_lines = [_yolo_label_line(record, width, height) for record in frame_records]
    content = "\n".join(label_lines)
    if content:
        content += "\n"
    label_path.write_text(content, encoding="utf-8")


def _sample_manifest_entry(
    frame_number: int,
    fps: float,
    label_key: str,
    frame_records: list[dict],
) -> dict:
    class_counts: dict[str, int] = defaultdict(int)
    for record in frame_records:
        class_counts[record["label"]] += 1
    return {
        "frame_number": frame_number,
        "timestamp_ms": round(frame_number / fps * 1000),
        "label_s3_key": label_key,
        "has_predictions": bool(frame_records),
        "prediction_count": len(frame_records),
        "class_counts": dict(class_counts),
        "label_file_empty": not frame_records,
    }


def _sampling_manifest(
    payload: dict,
    footage_bucket: str,
    source_metadata: dict,
    video: dict,
    prelabel_samples: list[dict],
) -> dict:
    return {
        "schema_version": SAMPLING_SCHEMA_VERSION,
        "status": "COMPLETE",
        "job_id": payload["job_id"],
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "created_at": datetime.now(timezone.utc)
        .isoformat()
        .replace(
            "+00:00",
            "Z",
        ),
        "source_video": {
            "bucket": footage_bucket,
            "key": payload["footage_s3_key"],
            "etag": str(source_metadata.get("ETag", "")).strip('"') or None,
            "version_id": source_metadata.get("VersionId"),
            "size_bytes": source_metadata.get("ContentLength"),
            "frame_count": video["frame_count"],
            "fps": video["fps"],
            "duration_ms": round(video["frame_count"] / video["fps"] * 1000),
            "width": video["width"],
            "height": video["height"],
        },
        "sampling": {
            "strategy": "fixed_frame_interval",
            "start_frame": 0,
            "interval_frames": FRAME_SAMPLE_INTERVAL,
            "include_empty_frames": True,
            "model_inference_scope": "all_frames",
            "sampled_frame_count": len(prelabel_samples),
        },
        "label_format": {
            "name": "YOLO",
            "coordinate_format": "normalized_xywh",
            "class_map": EXPECTED_CLASS_NAMES,
        },
        "frames": [sample["manifest_entry"] for sample in prelabel_samples],
    }


def _build_video_detections(
    records: list[dict],
    fps: float,
) -> tuple[list[dict], dict[str, float]]:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_class[record["label"]].append(record)

    detections = []
    best_frame_scores = {}
    for detection_index, (label, class_records) in enumerate(
        by_class.items(),
        start=1,
    ):
        by_frame: dict[int, list[dict]] = defaultdict(list)
        for record in class_records:
            by_frame[record["frame"]].append(record)
        best_frame_number = max(
            by_frame,
            key=lambda frame_number: (
                sum(record["confidence"] for record in by_frame[frame_number]),
                len(by_frame[frame_number]),
            ),
        )
        best_frame_records = by_frame[best_frame_number]
        best_frame_scores[label] = sum(
            record["confidence"] for record in best_frame_records
        )
        detections.append(
            {
                "detection_id": f"d-{detection_index:04d}",
                "label": label,
                "count": len(class_records),
                "best_frame_number": best_frame_number,
                "first_seen_ms": round(
                    min(record["frame"] for record in class_records) / fps * 1000
                ),
                "last_seen_ms": round(
                    max(record["frame"] for record in class_records) / fps * 1000
                ),
                "instances": [
                    {
                        "confidence": record["confidence"],
                        "bbox": record["bbox"],
                        "attributes": {
                            **record["attributes"],
                            "frame_number": record["frame"],
                            "timestamp_ms": round(record["frame"] / fps * 1000),
                        },
                    }
                    for record in class_records
                ],
            }
        )
    return detections, best_frame_scores


def _top_snapshot_instances(detections: list[dict]) -> list[dict]:
    instances = []
    for detection in detections:
        for instance in detection["instances"]:
            attributes = instance.get("attributes") or {}
            instances.append(
                {
                    "label": detection["label"],
                    "bbox": instance["bbox"],
                    "confidence": float(instance["confidence"]),
                    "frame_number": int(
                        attributes.get(
                            "frame_number",
                            detection.get("best_frame_number", 0),
                        )
                    ),
                }
            )
    instances.sort(key=lambda instance: instance["confidence"], reverse=True)
    return instances[:MAX_ANNOTATED_SNAPSHOTS]


def _save_annotated_snapshots(
    input_path: Path,
    detections: list[dict],
    output_root: Path,
) -> list[Path]:
    candidates = _top_snapshot_instances(detections)
    if not candidates:
        return []

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("Could not reopen input video for snapshots")
    snapshot_paths = []
    try:
        for candidate in candidates:
            capture.set(cv2.CAP_PROP_POS_FRAMES, candidate["frame_number"])
            ok, frame = capture.read()
            if not ok:
                logger.warning(
                    "Could not read snapshot frame %s",
                    candidate["frame_number"],
                )
                continue
            bbox = candidate["bbox"]
            start = (int(bbox["x"]), int(bbox["y"]))
            end = (
                start[0] + int(bbox["w"]),
                start[1] + int(bbox["h"]),
            )
            cv2.rectangle(frame, start, end, (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"{candidate['label']} {candidate['confidence']:.2f}",
                (start[0], max(20, start[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            snapshot_path = output_root / f"frame_{len(snapshot_paths) + 1}.jpg"
            if not cv2.imwrite(str(snapshot_path), frame):
                raise RuntimeError(f"Could not save snapshot: {snapshot_path.name}")
            snapshot_paths.append(snapshot_path)
    finally:
        capture.release()
    return snapshot_paths


def _process_video(
    model,
    device: str,
    input_path: Path,
    annotated_video_path: Path,
    prelabel_root: Path,
    prelabel_prefix: str,
) -> tuple[list[dict], object, list[dict], dict]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("Input does not contain a decodable video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Input video has invalid metadata")

    writer = cv2.VideoWriter(
        str(annotated_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create annotated MP4")

    prelabel_directory = prelabel_root / "labels"
    prelabel_directory.mkdir(parents=True)

    records = []
    prelabel_samples = []
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=DEFAULT_CONFIDENCE,
            device=device,
            verbose=False,
        )[0]
        frame_records = _extract_frame_records(result, frame_number)
        if _is_sampled_frame(frame_number):
            label_key = _prelabel_key(
                prelabel_prefix,
                frame_number,
            )
            label_path = prelabel_directory / Path(label_key).name
            for record in frame_records:
                record["attributes"]["prelabel_s3_key"] = label_key
            _write_prelabel(
                frame_records,
                label_path,
                width,
                height,
            )
            prelabel_samples.append(
                {
                    "label_path": label_path,
                    "label_key": label_key,
                    "manifest_entry": _sample_manifest_entry(
                        frame_number,
                        fps,
                        label_key,
                        frame_records,
                    ),
                }
            )
        records.extend(frame_records)
        writer.write(result.plot())
        frame_number += 1

    capture.release()
    writer.release()
    if frame_number == 0:
        raise ValueError("Input video contains no decodable frames")

    detections, _ = _build_video_detections(records, fps)
    return (
        detections,
        prelabel_samples,
        {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_number,
        },
    )


def _bucket_configuration() -> tuple[str, str]:
    footage_bucket = os.getenv("FOOTAGE_BUCKET")
    if not footage_bucket:
        raise RuntimeError("FOOTAGE_BUCKET environment variable is required")
    return footage_bucket, os.getenv("OUTPUT_BUCKET", footage_bucket)


def _upload_file(
    s3_client,
    source: Path,
    bucket: str,
    key: str,
    content_type: str,
) -> None:
    s3_client.upload_file(
        str(source),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )


def _run_prediction(payload: dict, runtime: dict) -> dict:
    footage_bucket, output_bucket = _bucket_configuration()
    s3_client = boto3.client("s3")
    job_id = payload["job_id"]
    footage_s3_key = payload["footage_s3_key"]
    output_prefix = f"jobs/{job_id}/{MODEL_ID}"
    prelabel_prefix = f"prelabels/{job_id}/{MODEL_ID}"
    manifest_key = f"{prelabel_prefix}/sampling_manifest.json"
    annotated_video_key = f"{output_prefix}/annotated.mp4"

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        video_suffix = Path(footage_s3_key).suffix.lower()
        input_path = temporary_root / f"input{video_suffix}"
        raw_annotated_video_path = temporary_root / "annotated_raw.mp4"
        annotated_video_path = temporary_root / "annotated.mp4"
        prelabel_root = temporary_root / "prelabels"
        source_metadata = s3_client.head_object(
            Bucket=footage_bucket,
            Key=footage_s3_key,
        )
        s3_client.download_file(
            footage_bucket,
            footage_s3_key,
            str(input_path),
        )
        detections, prelabel_samples, video = _process_video(
            runtime["model"],
            runtime["device"],
            input_path,
            raw_annotated_video_path,
            prelabel_root,
            prelabel_prefix,
        )
        snapshot_paths = _save_annotated_snapshots(
            input_path,
            detections,
            temporary_root,
        )
        _transcode_to_browser_mp4(
            raw_annotated_video_path,
            annotated_video_path,
        )
        _upload_file(
            s3_client,
            annotated_video_path,
            output_bucket,
            annotated_video_key,
            "video/mp4",
        )
        snapshot_keys = []
        for snapshot_path in snapshot_paths:
            snapshot_key = f"{output_prefix}/{snapshot_path.name}"
            _upload_file(
                s3_client,
                snapshot_path,
                output_bucket,
                snapshot_key,
                "image/jpeg",
            )
            snapshot_keys.append(snapshot_key)
        for sample in prelabel_samples:
            _upload_file(
                s3_client,
                sample["label_path"],
                output_bucket,
                sample["label_key"],
                "text/plain",
            )
        manifest = _sampling_manifest(
            payload,
            footage_bucket,
            source_metadata,
            video,
            prelabel_samples,
        )
        s3_client.put_object(
            Bucket=output_bucket,
            Key=manifest_key,
            Body=json.dumps(manifest).encode("utf-8"),
            ContentType="application/json",
        )
    return {
        "annotated_video_key": annotated_video_key,
        "snapshot_keys": snapshot_keys,
        "detections": detections,
    }


def predict_fn(payload: dict, runtime: dict) -> dict:
    started_at = time.perf_counter()
    _persist_json(payload["job_id"], "input.json", payload)
    try:
        prediction = _run_prediction(payload, runtime)
        response = _result_block(
            job_id=payload["job_id"],
            status="SUCCESS",
            processing_time_ms=_processing_time_ms(started_at),
            output_media=_output_media(
                prediction["annotated_video_key"],
                prediction["snapshot_keys"],
            ),
            detections=prediction["detections"],
            error=None,
        )
        _persist_json(payload["job_id"], "result.json", response)
        return response
    except Exception as error:
        logger.exception("YOLO26 fall-detection inference failed")
        response = _result_block(
            job_id=payload["job_id"],
            status="FAILED",
            processing_time_ms=_processing_time_ms(started_at),
            output_media=_output_media(),
            detections=[],
            error={
                "code": "INFERENCE_ERROR",
                "message": f"Inference failed: {type(error).__name__}",
            },
        )
        try:
            _persist_json(payload["job_id"], "result.json", response)
        except Exception:
            logger.exception("Could not persist failed fall result JSON")
        return response


def output_fn(prediction: dict, accept: str) -> tuple[str, str]:
    media_type = accept.split(";", maxsplit=1)[0].strip().lower()
    if media_type not in {"application/json", "*/*"}:
        raise ValueError("Only application/json responses are supported")
    return json.dumps(prediction), "application/json"
