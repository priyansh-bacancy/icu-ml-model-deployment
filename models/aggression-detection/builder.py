"""Map frame-level aggression confidence to the ICU Solarcam contract."""

from __future__ import annotations

import math

import numpy as np


def _active_runs(scores: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    active = np.asarray(scores, dtype=np.float64) >= threshold
    padded = np.pad(active.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def representative_peak_frame(scores: np.ndarray, offset: int = 0) -> int:
    """Choose the midpoint of the longest numerically tied peak plateau."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite, non-empty one-dimensional array")
    peak_indices = np.flatnonzero(
        np.isclose(values, values.max(), rtol=1e-9, atol=1e-12)
    )
    split_points = np.flatnonzero(np.diff(peak_indices) > 1) + 1
    plateaus = np.split(peak_indices, split_points)
    plateau = max(plateaus, key=lambda group: (group.size, -int(group[0])))
    return offset + (int(plateau[0]) + int(plateau[-1])) // 2


def build_detections(
    frame_scores: np.ndarray,
    *,
    fps: float,
    width: int,
    height: int,
    threshold: float,
) -> list[dict[str, object]]:
    """Return one instance for every consecutive above-threshold frame run."""
    scores = np.asarray(frame_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("frame_scores must be a non-empty one-dimensional array")
    if not np.isfinite(scores).all():
        raise ValueError("frame_scores contains a non-finite value")
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("video metadata must be positive")

    runs = _active_runs(scores, threshold)
    if not runs:
        return []

    instances = []
    for segment_id, (start_frame, end_frame) in enumerate(runs, start=1):
        segment_scores = scores[start_frame : end_frame + 1]
        highest_conf_frame = representative_peak_frame(
            segment_scores, offset=start_frame
        )
        instances.append(
            {
                "confidence": round(float(segment_scores.max()), 4),
                "bbox": {"x": 0, "y": 0, "w": width, "h": height},
                "attributes": {
                    "segment_id": segment_id,
                    "start_ms": round(start_frame * 1000 / fps),
                    "end_ms": round((end_frame + 1) * 1000 / fps),
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "highest_conf_frame": highest_conf_frame,
                    "mean_confidence": round(float(segment_scores.mean()), 4),
                },
            }
        )

    best_frame = representative_peak_frame(scores)
    first_seen_ms = instances[0]["attributes"]["start_ms"]
    last_seen_ms = instances[-1]["attributes"]["end_ms"]
    return [
        {
            "detection_id": "d-0001",
            "label": "Aggression",
            "count": len(instances),
            "best_frame_number": best_frame,
            "first_seen_ms": first_seen_ms,
            "last_seen_ms": last_seen_ms,
            "instances": instances,
        }
    ]


def validate_contract(detections: list[dict[str, object]]) -> None:
    """Fail fast if an internal change breaks the public instance contract."""
    for detection in detections:
        instances = detection["instances"]
        if detection["count"] != len(instances):
            raise ValueError("detection count must equal len(instances)")
        for instance in instances:
            confidence = float(instance["confidence"])
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("instance confidence must be between zero and one")
