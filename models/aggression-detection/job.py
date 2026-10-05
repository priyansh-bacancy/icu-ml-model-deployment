"""Run one ICU Solarcam aggression job and archive its result block."""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from builder import build_detections, validate_contract
from config import AGGRESSION_THRESHOLD, EVENT_TYPE, Settings
from modeling import ModelRuntime
from processor import AggressionProcessor
from snapshots import save_ranked_snapshots
from storage import Storage, normalize_key


JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
VIDEO_SUFFIXES = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}


def validate_request(data: dict[str, Any]) -> dict[str, str]:
    if not isinstance(data, dict) or set(data) != {"job_id", "footage_s3_key"}:
        raise ValueError("request must contain exactly job_id and footage_s3_key")
    if not isinstance(data["job_id"], str) or not isinstance(
        data["footage_s3_key"], str
    ):
        raise ValueError("job_id and footage_s3_key must be strings")
    job_id = data["job_id"].strip()
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise ValueError(
            "job_id may contain only letters, numbers, '.', '_' and '-'"
        )
    footage_key = normalize_key(data["footage_s3_key"])
    footage_path = PurePosixPath(footage_key)
    if len(footage_path.parts) != 2 or footage_path.parts[0] != "uploads":
        raise ValueError(
            "ABD footage_s3_key must be a direct upload: uploads/<video>"
        )
    if footage_path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("ABD requires a supported video file")
    return {"job_id": job_id, "footage_s3_key": footage_key}


def _processing_time_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


def _failed_result(
    settings: Settings, event_id: str, started: float, exc: Exception
) -> dict:
    return {
        "model_id": settings.model_id,
        "event_type": EVENT_TYPE,
        "event_id": event_id,
        "model_version": settings.model_version,
        "status": "FAILED",
        "processing_time_ms": _processing_time_ms(started),
        "output_media": {
            "annotated_video_s3_key": None,
            "annotated_image_s3_keys": [],
        },
        "detections": [],
        "error": {
            "code": "PROCESSING_ERROR",
            "message": str(exc)[:1000],
        },
    }


def run_job(
    request_data: dict[str, Any],
    *,
    runtime: ModelRuntime,
    settings: Settings,
    storage: Storage,
) -> dict[str, Any]:
    started = time.perf_counter()
    payload = validate_request(request_data)
    output_prefix = f"jobs/{payload['job_id']}/{settings.model_id}"
    result_key = f"{output_prefix}/result.json"

    try:
        suffix = PurePosixPath(payload["footage_s3_key"]).suffix.lower()
        with tempfile.TemporaryDirectory(prefix="videomae-aggression-") as temp:
            workdir = Path(temp)
            source = workdir / f"source{suffix}"
            storage.download_file(payload["footage_s3_key"], source)

            processor = AggressionProcessor(
                runtime,
                frame_context_seconds=settings.frame_context_seconds,
                score_interval_seconds=settings.score_interval_seconds,
                batch_size=settings.batch_size,
            )
            artifacts = processor.process(source, workdir)
            metadata = artifacts.metadata
            detections = build_detections(
                artifacts.frame_scores,
                fps=metadata.fps,
                width=metadata.width,
                height=metadata.height,
                threshold=AGGRESSION_THRESHOLD,
            )
            validate_contract(detections)

            video_key = f"{output_prefix}/annotated.mp4"
            storage.upload_file(artifacts.annotated_video, video_key)
            snapshot_paths = save_ranked_snapshots(
                source,
                detections,
                is_video=True,
                workdir=workdir,
            )
            image_keys = []
            for index, snapshot_path in enumerate(snapshot_paths, start=1):
                image_key = f"{output_prefix}/frame_{index}.jpg"
                storage.upload_file(snapshot_path, image_key)
                image_keys.append(image_key)

        result = {
            "model_id": settings.model_id,
            "event_type": EVENT_TYPE,
            "event_id": payload["job_id"],
            "model_version": settings.model_version,
            "status": "SUCCESS",
            "processing_time_ms": _processing_time_ms(started),
            "output_media": {
                "annotated_video_s3_key": video_key,
                "annotated_image_s3_keys": image_keys,
            },
            "detections": detections,
            "error": None,
        }
        storage.put_json(result_key, result)
        return result
    except Exception as exc:
        result = _failed_result(settings, payload["job_id"], started, exc)
        try:
            storage.put_json(result_key, result)
        except Exception:
            pass
        return result
