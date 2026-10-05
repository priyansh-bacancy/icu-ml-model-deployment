"""Build the ICU Solarcam grouped detection contract."""

from __future__ import annotations

from typing import Any


def _best_frame(detections: list[dict[str, Any]]) -> int:
    scores: dict[int, tuple[float, int]] = {}
    for detection in detections:
        frame = int(detection["frame"])
        confidence_sum, count = scores.get(frame, (0.0, 0))
        scores[frame] = (confidence_sum + float(detection["confidence"]), count + 1)
    return max(scores, key=lambda frame: (scores[frame][0], scores[frame][1], -frame))


def build_detections(
    flat_detections: list[dict[str, Any]],
    *,
    is_video: bool,
    fps: float | None = None,
) -> list[dict[str, Any]]:
    """Group boxes by class and retain frame/timestamp metadata for videos."""
    by_class: dict[str, list[dict[str, Any]]] = {}
    for detection in flat_detections:
        by_class.setdefault(str(detection["label"]), []).append(detection)

    grouped: list[dict[str, Any]] = []
    for index, (label, detections) in enumerate(by_class.items(), start=1):
        detections = sorted(detections, key=lambda item: int(item["frame"]))
        entry: dict[str, Any] = {
            "detection_id": f"d-{index:04d}",
            "label": label,
        }

        if is_video:
            entry["best_frame_number"] = _best_frame(detections)
            if fps and fps > 0:
                entry["first_seen_ms"] = round(
                    min(int(item["frame"]) for item in detections) * 1000 / fps
                )
                entry["last_seen_ms"] = round(
                    max(int(item["frame"]) for item in detections) * 1000 / fps
                )
            else:
                entry["first_seen_ms"] = None
                entry["last_seen_ms"] = None

        entry["count"] = len(detections)
        instances = []
        for item in detections:
            attributes = dict(item.get("attributes", {}))
            if is_video:
                frame_number = int(item["frame"])
                attributes["frame_number"] = frame_number
                attributes["timestamp_ms"] = (
                    round(frame_number * 1000 / fps) if fps and fps > 0 else None
                )
            instances.append(
                {
                    "confidence": round(float(item["confidence"]), 4),
                    "bbox": item["bbox"],
                    "attributes": attributes,
                }
            )
        entry["instances"] = instances
        grouped.append(entry)

    return grouped


def select_representative_frame(flat_detections: list[dict[str, Any]]) -> int:
    return _best_frame(flat_detections) if flat_detections else 0
