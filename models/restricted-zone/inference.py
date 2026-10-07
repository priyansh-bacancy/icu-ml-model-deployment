import json
import logging
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3
import cv2
import torch

from zone_pipeline import CLASS_NAMES, FRAME_SAMPLE_INTERVAL, process_video


MODEL_FILENAME = "best.pt"
MODEL_ID = "restricted-zone"
MODEL_VERSION = "v7"
EVENT_TYPE = "restricted_zone_event"
MAX_ANNOTATED_SNAPSHOTS = 5
SAMPLING_SCHEMA_VERSION = "1.0"
UPLOADS_PREFIX = "uploads/"
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
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
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp")
    from ultralytics import YOLO

    model_root = Path(model_dir)
    model_path = model_root / MODEL_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    model = YOLO(model_path)
    expected_names = {0: "vehicle", 1: "person"}
    if model.names != expected_names:
        raise ValueError(
            f"Expected model classes {expected_names}, found {model.names}"
        )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return {
        "model": model,
        "device": device,
    }


def _validate_s3_key(value: object, field_name: str, suffixes: set[str]) -> str:
    if not isinstance(value, str) or not value.strip("/"):
        raise ValueError(f"'{field_name}' must be a non-empty S3 key")
    if value.startswith("/") or ".." in Path(value).parts:
        raise ValueError(f"'{field_name}' must be a relative S3 key")
    if not value.startswith(UPLOADS_PREFIX):
        raise ValueError(f"'{field_name}' must start with 'uploads/'")
    if Path(value).suffix.lower() not in suffixes:
        raise ValueError(f"'{field_name}' has an unsupported file type")
    return value


def input_fn(request_body: bytes | str, content_type: str) -> dict:
    media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    if media_type != "application/json":
        raise ValueError("Restricted-zone inference accepts application/json only")
    body = (
        request_body.decode("utf-8")
        if isinstance(request_body, bytes)
        else request_body
    )
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise ValueError("Request body is not valid JSON") from error

    required_fields = {"job_id", "footage_s3_key", "zones_s3_key"}
    if not isinstance(payload, dict) or set(payload) != required_fields:
        raise ValueError(
            "Request must contain exactly 'job_id', 'footage_s3_key', "
            "and 'zones_s3_key'"
        )
    job_id = payload["job_id"]
    if not isinstance(job_id, str) or not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError("'job_id' contains unsupported characters")
    footage_s3_key = _validate_s3_key(
        payload["footage_s3_key"],
        "footage_s3_key",
        SUPPORTED_VIDEO_SUFFIXES,
    )
    zones_s3_key = _validate_s3_key(
        payload["zones_s3_key"],
        "zones_s3_key",
        {".json"},
    )
    return {
        "job_id": job_id,
        "footage_s3_key": footage_s3_key,
        "zones_s3_key": zones_s3_key,
    }


def _download_file(s3_client, bucket: str, key: str, destination: Path) -> None:
    s3_client.download_file(bucket, key, str(destination))


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


def _bucket_configuration() -> tuple[str, str]:
    footage_bucket = os.getenv("FOOTAGE_BUCKET")
    if not footage_bucket:
        raise RuntimeError("FOOTAGE_BUCKET environment variable is required")
    return footage_bucket, os.getenv("OUTPUT_BUCKET", footage_bucket)


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


def _sampling_manifest(
    payload: dict,
    footage_bucket: str,
    source_metadata: dict,
    video_result,
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
            "frame_count": video_result.frame_count,
            "fps": video_result.fps,
            "duration_ms": round(video_result.frame_count / video_result.fps * 1000),
            "width": video_result.width,
            "height": video_result.height,
        },
        "sampling": {
            "strategy": "fixed_frame_interval",
            "start_frame": 0,
            "interval_frames": FRAME_SAMPLE_INTERVAL,
            "include_empty_frames": True,
            "model_inference_scope": "all_frames",
            "sampled_frame_count": len(video_result.prelabel_samples),
        },
        "label_format": {
            "name": "YOLO",
            "coordinate_format": "normalized_xywh",
            "class_map": CLASS_NAMES,
        },
        "frames": [
            sample["manifest_entry"] for sample in video_result.prelabel_samples
        ],
    }


def _run_prediction(payload: dict, runtime: dict) -> dict:
    footage_bucket, output_bucket = _bucket_configuration()
    s3_client = boto3.client("s3")
    job_id = payload["job_id"]
    footage_s3_key = payload["footage_s3_key"]
    output_prefix = f"jobs/{job_id}/{MODEL_ID}"
    prelabel_prefix = f"prelabels/{job_id}/{MODEL_ID}"
    manifest_key = f"{prelabel_prefix}/sampling_manifest.json"
    video_output_key = f"{output_prefix}/annotated.mp4"

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        input_path = temporary_root / f"input{Path(footage_s3_key).suffix.lower()}"
        raw_output_path = temporary_root / "annotated_raw.mp4"
        output_path = temporary_root / "annotated.mp4"
        prelabel_root = temporary_root / "prelabels"
        source_metadata = s3_client.head_object(
            Bucket=footage_bucket,
            Key=footage_s3_key,
        )
        _download_file(s3_client, footage_bucket, footage_s3_key, input_path)
        zones_path = temporary_root / "zones.json"
        _download_file(
            s3_client,
            footage_bucket,
            payload["zones_s3_key"],
            zones_path,
        )
        zone_config = json.loads(zones_path.read_text(encoding="utf-8"))

        result = process_video(
            runtime["model"],
            input_path,
            raw_output_path,
            zone_config,
            runtime["device"],
            prelabel_root,
            prelabel_prefix,
        )
        snapshot_paths = _save_annotated_snapshots(
            input_path,
            result.detections,
            temporary_root,
        )
        _transcode_to_browser_mp4(raw_output_path, output_path)
        _upload_file(
            s3_client,
            output_path,
            output_bucket,
            video_output_key,
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
        for sample in result.prelabel_samples:
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
            result,
        )
        s3_client.put_object(
            Bucket=output_bucket,
            Key=manifest_key,
            Body=json.dumps(manifest).encode("utf-8"),
            ContentType="application/json",
        )

    return {
        "detections": result.detections,
        "annotated_video_s3_key": video_output_key,
        "annotated_image_s3_keys": snapshot_keys,
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
                prediction["annotated_video_s3_key"],
                prediction["annotated_image_s3_keys"],
            ),
            detections=prediction["detections"],
            error=None,
        )
        _persist_json(payload["job_id"], "result.json", response)
        return response
    except Exception as error:
        logger.exception("Restricted-zone inference failed")
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
            logger.exception("Could not persist failed restricted-zone result JSON")
        return response


def output_fn(prediction: dict, accept: str) -> tuple[str, str]:
    media_type = accept.split(";", maxsplit=1)[0].strip().lower()
    if media_type not in {"application/json", "*/*"}:
        raise ValueError("Only application/json responses are supported")
    return json.dumps(prediction), "application/json"
