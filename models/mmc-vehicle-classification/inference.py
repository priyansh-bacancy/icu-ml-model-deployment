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
import numpy as np

from modeling import (
    COLOR_INDEX_NAME,
    DEFAULT_TOP_K,
    DETECTION_CONFIDENCE,
    DETECTION_IMAGE_SIZE,
    DETECTION_IOU,
    EXPECTED_VEHICLE_CLASSES,
    MMCRuntime,
    load_runtime,
)


MODEL_ID = "mmc-vehicle-classification"
MODEL_VERSION = "v8"
EVENT_TYPE = "vehicle_event"
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


JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,62}$")
MAX_TRACK_GAP_FRAMES = 15
TRACK_IOU_THRESHOLD = 0.30
DUPLICATE_BOX_IOU_THRESHOLD = 0.90
CLASSIFICATION_SAMPLE_INTERVAL_FRAMES = 30
MAX_CLASSIFICATION_SAMPLES = 6
DEFAULT_FAISS_S3_PREFIX = "models/mmc-vehicle-classification/faiss_db/v1"
FAISS_ARTIFACTS = (
    "faiss/color_index.faiss",
    "faiss/color_index_metadata.json",
    "faiss/make_model_registry.faiss",
    "faiss/make_model_registry_metadata.json",
)
ANNOTATION_COLORS = {
    "car": (80, 220, 80),
    "truck": (80, 180, 255),
    "bus": (255, 180, 60),
    "motorcycle": (255, 80, 200),
    "vehicle": (80, 220, 80),
}

logger = logging.getLogger(__name__)


def _faiss_configuration() -> tuple[str, str, Path]:
    faiss_bucket = os.getenv("FAISS_BUCKET") or os.getenv("FOOTAGE_BUCKET")
    if not faiss_bucket:
        raise RuntimeError(
            "FAISS_BUCKET or FOOTAGE_BUCKET environment variable is required"
        )
    faiss_prefix = os.getenv(
        "FAISS_S3_PREFIX",
        DEFAULT_FAISS_S3_PREFIX,
    ).strip("/")
    if not faiss_prefix or ".." in Path(faiss_prefix).parts:
        raise RuntimeError("FAISS_S3_PREFIX must be a valid relative S3 prefix")
    local_directory = Path(os.getenv("FAISS_LOCAL_DIRECTORY", "/tmp/mmc-vector-db"))
    return faiss_bucket, faiss_prefix, local_directory


def _download_faiss_artifacts() -> Path:
    faiss_bucket, faiss_prefix, local_directory = _faiss_configuration()
    s3_client = boto3.client("s3")
    for relative_path in FAISS_ARTIFACTS:
        destination = local_directory / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            faiss_bucket,
            f"{faiss_prefix}/{relative_path}",
            str(destination),
        )
    return local_directory


def model_fn(model_dir: str) -> MMCRuntime:
    return load_runtime(
        model_dir,
        vector_db_directory=_download_faiss_artifacts(),
    )


def input_fn(request_body: bytes | str, content_type: str) -> dict:
    media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    if media_type != "application/json":
        raise ValueError("MMC video inference accepts application/json only")

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
    return {"job_id": job_id, "footage_s3_key": footage_s3_key}


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


def _class_name(names: dict | list, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id])


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
    return (
        f"{record['class_id']} "
        f"{((xmin + xmax) / 2.0) / width:.6f} "
        f"{((ymin + ymax) / 2.0) / height:.6f} "
        f"{(xmax - xmin) / width:.6f} "
        f"{(ymax - ymin) / height:.6f}"
    )


def _write_prelabel(
    records: list[dict],
    label_path: Path,
    width: int,
    height: int,
) -> None:
    content = "\n".join(_yolo_label_line(record, width, height) for record in records)
    if content:
        content += "\n"
    label_path.write_text(content, encoding="utf-8")


def _sample_manifest_entry(
    frame_number: int,
    fps: float,
    label_key: str,
    records: list[dict],
) -> dict:
    class_counts: dict[str, int] = defaultdict(int)
    for record in records:
        class_counts[record["label"]] += 1
    return {
        "frame_number": frame_number,
        "timestamp_ms": round(frame_number / fps * 1000),
        "label_s3_key": label_key,
        "has_predictions": bool(records),
        "prediction_count": len(records),
        "class_counts": dict(class_counts),
        "label_file_empty": not records,
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
            "class_map": EXPECTED_VEHICLE_CLASSES,
        },
        "frames": [sample["manifest_entry"] for sample in prelabel_samples],
    }


def _bbox_iou(first: dict, second: dict) -> float:
    intersection_x1 = max(first["x1"], second["x1"])
    intersection_y1 = max(first["y1"], second["y1"])
    intersection_x2 = min(first["x2"], second["x2"])
    intersection_y2 = min(first["y2"], second["y2"])
    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height
    first_area = (first["x2"] - first["x1"]) * (first["y2"] - first["y1"])
    second_area = (second["x2"] - second["x1"]) * (second["y2"] - second["y1"])
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def _assign_track(
    bbox: dict,
    vehicle_class: str,
    frame_number: int,
    tracks: dict[int, dict],
    claimed_track_ids: set[int],
) -> int:
    candidates = [
        (track_id, _bbox_iou(bbox, track["last_bbox"]))
        for track_id, track in tracks.items()
        if track_id not in claimed_track_ids
        and track["vehicle_class"] == vehicle_class
        and frame_number - track["last_frame"] <= MAX_TRACK_GAP_FRAMES
    ]
    if candidates:
        track_id, overlap = max(candidates, key=lambda candidate: candidate[1])
        if overlap >= TRACK_IOU_THRESHOLD:
            return track_id
    return max(tracks, default=0) + 1


def _extract_frame_records(
    result,
    frame: np.ndarray,
    frame_number: int,
    tracks: dict[int, dict],
) -> list[dict]:
    if result.boxes is None:
        return []

    image_height, image_width = frame.shape[:2]
    candidates = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        raw_x1, raw_y1, raw_x2, raw_y2 = box.xyxy[0].cpu().tolist()
        candidates.append(
            {
                "class_id": class_id,
                "vehicle_class": _class_name(result.names, class_id),
                "confidence": float(box.conf.item()),
                "bbox": {
                    "x1": max(0, min(image_width, round(raw_x1))),
                    "y1": max(0, min(image_height, round(raw_y1))),
                    "x2": max(0, min(image_width, round(raw_x2))),
                    "y2": max(0, min(image_height, round(raw_y2))),
                },
            }
        )

    claimed_track_ids: set[int] = set()
    accepted_bboxes = []
    frame_records = []
    for candidate in sorted(
        candidates,
        key=lambda item: item["confidence"],
        reverse=True,
    ):
        bbox = candidate["bbox"]
        if bbox["x2"] <= bbox["x1"] or bbox["y2"] <= bbox["y1"]:
            continue
        if any(
            _bbox_iou(bbox, accepted_bbox) >= DUPLICATE_BOX_IOU_THRESHOLD
            for accepted_bbox in accepted_bboxes
        ):
            continue
        accepted_bboxes.append(bbox)

        track_id = _assign_track(
            bbox,
            candidate["vehicle_class"],
            frame_number,
            tracks,
            claimed_track_ids,
        )
        claimed_track_ids.add(track_id)
        crop = frame[
            bbox["y1"] : bbox["y2"],
            bbox["x1"] : bbox["x2"],
        ].copy()
        track = tracks.setdefault(
            track_id,
            {
                "vehicle_class": candidate["vehicle_class"],
                "best_confidence": -1.0,
                "crop_candidates": {},
            },
        )
        track.update(
            {
                "last_bbox": bbox,
                "last_frame": frame_number,
            }
        )
        if candidate["confidence"] > track["best_confidence"]:
            track.update(
                {
                    "best_confidence": candidate["confidence"],
                    "best_frame": frame_number,
                    "best_crop": crop,
                }
            )
        sample_bucket = frame_number // CLASSIFICATION_SAMPLE_INTERVAL_FRAMES
        existing_sample = track["crop_candidates"].get(sample_bucket)
        if (
            existing_sample is None
            or candidate["confidence"] > existing_sample["confidence"]
        ):
            track["crop_candidates"][sample_bucket] = {
                "confidence": candidate["confidence"],
                "frame": frame_number,
                "crop": crop,
            }
        frame_records.append(
            {
                "frame": frame_number,
                "track_id": track_id,
                "class_id": candidate["class_id"],
                "label": candidate["vehicle_class"],
                "confidence": round(candidate["confidence"], 4),
                "bbox": {
                    "x": bbox["x1"],
                    "y": bbox["y1"],
                    "w": bbox["x2"] - bbox["x1"],
                    "h": bbox["y2"] - bbox["y1"],
                },
            }
        )
    return frame_records


def _detect_and_track(
    runtime: MMCRuntime,
    input_path: Path,
    prelabel_root: Path,
    prelabel_prefix: str,
) -> tuple[list[dict], dict[int, dict], dict, list[dict]]:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError("Input does not contain a decodable video")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Input video has invalid metadata")

    records = []
    tracks: dict[int, dict] = {}
    prelabel_directory = prelabel_root / "labels"
    prelabel_directory.mkdir(parents=True)
    prelabel_samples = []
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = runtime.vehicle_detector.predict(
            source=frame,
            imgsz=DETECTION_IMAGE_SIZE,
            conf=DETECTION_CONFIDENCE,
            iou=DETECTION_IOU,
            agnostic_nms=True,
            device=runtime.device,
            verbose=False,
        )[0]
        frame_records = _extract_frame_records(
            result,
            frame,
            frame_number,
            tracks,
        )
        if _is_sampled_frame(frame_number):
            label_key = _prelabel_key(
                prelabel_prefix,
                frame_number,
            )
            label_path = prelabel_directory / Path(label_key).name
            _write_prelabel(
                frame_records,
                label_path,
                width,
                height,
            )
            for record in frame_records:
                record["prelabel_s3_key"] = label_key
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
        frame_number += 1
    capture.release()
    if frame_number == 0:
        raise ValueError("Input video contains no decodable frames")
    return (
        records,
        tracks,
        {
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_number,
        },
        prelabel_samples,
    )


def _sampled_crops(track: dict) -> list[np.ndarray]:
    candidates = sorted(
        track["crop_candidates"].values(),
        key=lambda candidate: candidate["frame"],
    )
    if len(candidates) <= MAX_CLASSIFICATION_SAMPLES:
        return [candidate["crop"] for candidate in candidates]
    sample_indices = np.linspace(
        0,
        len(candidates) - 1,
        MAX_CLASSIFICATION_SAMPLES,
        dtype=int,
    )
    return [candidates[index]["crop"] for index in sample_indices]


def _aggregate_prediction(
    match_sets: list[list[dict]],
    group_field: str | None = None,
) -> dict:
    if not match_sets or not any(match_sets):
        return {
            "label": "unknown",
            "score": None,
            "score_type": None,
            "sample_count": len(match_sets),
        }

    first_match = next(matches[0] for matches in match_sets if matches)
    score_name = "similarity" if "similarity" in first_match else "distance"
    group_scores: dict[str, float] = defaultdict(float)
    group_matches: dict[str, list[dict]] = defaultdict(list)
    for matches in match_sets:
        best_per_group = {}
        for match in matches:
            group = str(match.get(group_field) or match["label_name"])
            existing = best_per_group.get(group)
            if (
                existing is None
                or (
                    score_name == "distance"
                    and match[score_name] < existing[score_name]
                )
                or (
                    score_name == "similarity"
                    and match[score_name] > existing[score_name]
                )
            ):
                best_per_group[group] = match
        for group, match in best_per_group.items():
            rank = max(1, int(match["rank"]))
            score = float(match[score_name])
            if score_name == "distance":
                evidence = 1.0 + 1.0 / ((max(0.0, score) + 0.1) * rank)
            else:
                evidence = 1.0 + max(0.0, score) / rank
            group_scores[group] += evidence
            group_matches[group].append(match)

    winning_group = max(group_scores, key=group_scores.get)
    if score_name == "distance":
        match = min(
            group_matches[winning_group],
            key=lambda item: item[score_name],
        )
    else:
        match = max(
            group_matches[winning_group],
            key=lambda item: item[score_name],
        )
    prediction = {
        "label": match["label_name"],
        "score": round(float(match[score_name]), 4),
        "score_type": score_name,
        "sample_count": len(match_sets),
    }
    for field in ("market", "make_model"):
        if field in match:
            prediction[field] = match[field]
    return prediction


def _classify_tracks(
    runtime: MMCRuntime,
    tracks: dict[int, dict],
    crop_prefix: str,
) -> list[dict]:
    crops = []
    for track_id, track in tracks.items():
        color_match_sets = []
        make_match_sets = []
        for crop in _sampled_crops(track):
            color_embedding = runtime.embed(crop, task="color")
            make_embedding = runtime.embed(crop, task="make")
            color_match_sets.append(
                runtime.vector_store.query(
                    COLOR_INDEX_NAME,
                    color_embedding,
                    DEFAULT_TOP_K,
                )
            )
            make_match_sets.append(
                runtime.vector_store.query(
                    runtime.make_index_name,
                    make_embedding,
                    DEFAULT_TOP_K,
                )
            )
        crop_key = f"{crop_prefix}/track_{track_id:04d}.png"
        track["attributes"] = {
            "color": _aggregate_prediction(color_match_sets),
            "make_model": _aggregate_prediction(
                make_match_sets,
                group_field="make_model",
            ),
            "crop_s3_key": crop_key,
        }
        crops.append(
            {
                "image": track["best_crop"],
                "key": crop_key,
            }
        )
    return crops


def _annotation_label(record: dict, track: dict) -> str:
    color = track["attributes"]["color"]["label"]
    make_prediction = track["attributes"]["make_model"]
    make_model = make_prediction.get("make_model", make_prediction["label"])
    return (
        f"#{record['track_id']} {record['label']} "
        f"{record['confidence']:.2f} | {color} | {make_model}"
    )


def _draw_record(frame: np.ndarray, record: dict, track: dict) -> None:
    bbox = record["bbox"]
    start = (bbox["x"], bbox["y"])
    end = (bbox["x"] + bbox["w"], bbox["y"] + bbox["h"])
    color = ANNOTATION_COLORS.get(record["label"], (80, 220, 80))
    cv2.rectangle(frame, start, end, color, 2)
    text_y = max(20, bbox["y"] - 8)
    cv2.putText(
        frame,
        _annotation_label(record, track),
        (bbox["x"], text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


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


def _render_outputs(
    input_path: Path,
    annotated_video_path: Path,
    records: list[dict],
    tracks: dict[int, dict],
    video: dict,
) -> None:
    by_frame: dict[int, list[dict]] = defaultdict(list)
    for record in records:
        by_frame[record["frame"]].append(record)
    capture = cv2.VideoCapture(str(input_path))
    writer = cv2.VideoWriter(
        str(annotated_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        video["fps"],
        (video["width"], video["height"]),
    )
    if not capture.isOpened() or not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError("Could not create annotated video")

    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        for record in by_frame.get(frame_number, []):
            _draw_record(frame, record, tracks[record["track_id"]])
        writer.write(frame)
        frame_number += 1
    capture.release()
    writer.release()


def _build_contract_detections(
    records: list[dict],
    tracks: dict[int, dict],
    fps: float,
) -> list[dict]:
    by_class: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_class[record["label"]].append(record)

    detections = []
    for detection_index, (label, class_records) in enumerate(
        by_class.items(),
        start=1,
    ):
        detections.append(
            {
                "detection_id": f"d-{detection_index:04d}",
                "label": label,
                "count": len(class_records),
                "best_frame_number": max(
                    class_records,
                    key=lambda record: record["confidence"],
                )["frame"],
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
                            "track_id": record["track_id"],
                            "frame_number": record["frame"],
                            "timestamp_ms": round(record["frame"] / fps * 1000),
                            **(
                                {
                                    "prelabel_s3_key": record["prelabel_s3_key"],
                                }
                                if "prelabel_s3_key" in record
                                else {}
                            ),
                            **tracks[record["track_id"]]["attributes"],
                        },
                    }
                    for record in class_records
                ],
            }
        )
    return detections


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


def _upload_crop(s3_client, crop: dict, bucket: str, root: Path) -> None:
    crop_path = root / Path(crop["key"]).name
    if not cv2.imwrite(str(crop_path), crop["image"]):
        raise RuntimeError(f"Could not save vehicle crop: {crop_path.name}")
    _upload_file(s3_client, crop_path, bucket, crop["key"], "image/png")


def _result_key(job_id: str) -> str:
    return f"jobs/{job_id}/{MODEL_ID}/result.json"


def _input_key(job_id: str) -> str:
    return f"jobs/{job_id}/{MODEL_ID}/input.json"


def _persist_input(payload: dict) -> None:
    _, output_bucket = _bucket_configuration()
    boto3.client("s3").put_object(
        Bucket=output_bucket,
        Key=_input_key(payload["job_id"]),
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )


def _persist_result(job_id: str, response: dict) -> None:
    _, output_bucket = _bucket_configuration()
    boto3.client("s3").put_object(
        Bucket=output_bucket,
        Key=_result_key(job_id),
        Body=json.dumps(response).encode("utf-8"),
        ContentType="application/json",
    )


def _run_prediction(payload: dict, runtime: MMCRuntime) -> dict:
    footage_bucket, output_bucket = _bucket_configuration()
    s3_client = boto3.client("s3")
    job_id = payload["job_id"]
    footage_s3_key = payload["footage_s3_key"]
    output_prefix = f"jobs/{job_id}/{MODEL_ID}"
    annotated_video_key = f"{output_prefix}/annotated.mp4"
    crop_prefix = f"{output_prefix}/crops"
    prelabel_prefix = f"prelabels/{job_id}/{MODEL_ID}"
    manifest_key = f"{prelabel_prefix}/sampling_manifest.json"

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        input_path = temporary_root / f"input{Path(footage_s3_key).suffix.lower()}"
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
        records, tracks, video, prelabel_samples = _detect_and_track(
            runtime,
            input_path,
            prelabel_root,
            prelabel_prefix,
        )
        crops = _classify_tracks(runtime, tracks, crop_prefix)
        _render_outputs(
            input_path,
            raw_annotated_video_path,
            records,
            tracks,
            video,
        )
        _transcode_to_browser_mp4(
            raw_annotated_video_path,
            annotated_video_path,
        )
        detections = _build_contract_detections(
            records,
            tracks,
            video["fps"],
        )
        snapshot_paths = _save_annotated_snapshots(
            input_path,
            detections,
            temporary_root,
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
        for crop in crops:
            _upload_crop(s3_client, crop, output_bucket, temporary_root)
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


def predict_fn(payload: dict, runtime: MMCRuntime) -> dict:
    started_at = time.perf_counter()
    _persist_input(payload)
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
        _persist_result(payload["job_id"], response)
        return response
    except Exception as error:
        logger.exception("MMC vehicle video inference failed")
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
            _persist_result(payload["job_id"], response)
        except Exception:
            logger.exception("Could not persist failed MMC result JSON")
        return response


def output_fn(prediction: dict, accept: str) -> tuple[str, str]:
    media_type = accept.split(";", maxsplit=1)[0].strip().lower()
    if media_type not in {"application/json", "*/*"}:
        raise ValueError("Only application/json responses are supported")
    return json.dumps(prediction), "application/json"
