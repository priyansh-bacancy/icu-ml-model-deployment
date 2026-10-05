"""Convert Ultralytics results into the model-independent flat format."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _class_name(class_names: Mapping[int, str] | Sequence[str], class_id: int) -> str:
    return str(class_names[class_id])


def extract_result(
    result: Any,
    class_names: Mapping[int, str] | Sequence[str],
    frame_number: int,
) -> list[dict[str, Any]]:
    """Flatten one YOLO result and convert xywh center boxes to top-left pixels."""
    detections: list[dict[str, Any]] = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        confidence = float(box.conf.item())
        center_x, center_y, width, height = box.xywh[0].tolist()
        detections.append(
            {
                "frame": frame_number,
                "class_id": class_id,
                "label": _class_name(class_names, class_id),
                "confidence": round(confidence, 4),
                "bbox": {
                    "x": round(center_x - width / 2),
                    "y": round(center_y - height / 2),
                    "w": round(width),
                    "h": round(height),
                },
                "attributes": {},
            }
        )
    return detections
