"""Convert and filter Ultralytics results for sleep detection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _class_name(class_names: Mapping[int, str] | Sequence[str], class_id: int) -> str:
    return str(class_names[class_id])


def class_id_for_label(
    class_names: Mapping[int, str] | Sequence[str], target_label: str
) -> int:
    items = class_names.items() if isinstance(class_names, Mapping) else enumerate(class_names)
    matches = [int(class_id) for class_id, label in items if str(label) == target_label]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one model class named {target_label!r}; classes={class_names}"
        )
    return matches[0]


def extract_result(
    result: Any,
    class_names: Mapping[int, str] | Sequence[str],
    frame_number: int,
    *,
    target_label: str,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    """Return only target boxes whose reported confidence is strictly above the threshold."""
    detections: list[dict[str, Any]] = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        label = _class_name(class_names, class_id)
        confidence = round(float(box.conf.item()), 4)

        # This second-stage check is intentionally strict. Ultralytics' `conf`
        # argument is an inference optimization; this is the public contract.
        if label != target_label or confidence <= confidence_threshold:
            continue

        center_x, center_y, width, height = box.xywh[0].tolist()
        detections.append(
            {
                "frame": frame_number,
                "class_id": class_id,
                "label": label,
                "confidence": confidence,
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
