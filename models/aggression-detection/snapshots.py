"""Create confidence-ranked annotated evidence snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2


MAX_SNAPSHOTS = 5


def _ranked_instances(
    detections: list[dict[str, Any]], *, is_video: bool
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for detection in detections:
        for instance in detection.get("instances", []):
            attributes = instance.get("attributes", {})
            frame_number = 0
            if is_video:
                frame_number = int(
                    attributes.get(
                        "frame_number",
                        attributes.get(
                            "highest_conf_frame",
                            detection.get("best_frame_number", 0),
                        ),
                    )
                )
            ranked.append(
                {
                    "label": str(detection["label"]),
                    "bbox": instance["bbox"],
                    "confidence": float(instance["confidence"]),
                    "frame_number": frame_number,
                }
            )
    ranked.sort(key=lambda item: item["confidence"], reverse=True)
    return ranked[:MAX_SNAPSHOTS]


def save_ranked_snapshots(
    source: Path,
    detections: list[dict[str, Any]],
    *,
    is_video: bool,
    workdir: Path,
) -> list[Path]:
    """Save up to five top-confidence instances on their actual source frames."""
    instances = _ranked_instances(detections, is_video=is_video)
    if not instances:
        return []

    capture = cv2.VideoCapture(str(source)) if is_video else None
    source_image = None if is_video else cv2.imread(str(source))
    if is_video and (capture is None or not capture.isOpened()):
        raise ValueError(f"unable to decode video for snapshots: {source.name}")
    if not is_video and source_image is None:
        raise ValueError(f"unable to decode image for snapshots: {source.name}")

    paths: list[Path] = []
    try:
        for index, instance in enumerate(instances, start=1):
            if capture is not None:
                capture.set(cv2.CAP_PROP_POS_FRAMES, instance["frame_number"])
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise ValueError(
                        f"unable to read snapshot frame {instance['frame_number']}"
                    )
            else:
                frame = source_image.copy()

            height, width = frame.shape[:2]
            bbox = instance["bbox"]
            x1 = max(0, min(width - 1, int(bbox["x"])))
            y1 = max(0, min(height - 1, int(bbox["y"])))
            x2 = max(0, min(width - 1, x1 + int(bbox["w"])))
            y2 = max(0, min(height - 1, y1 + int(bbox["h"])))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f"{instance['label']} {instance['confidence']:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            path = workdir / f"frame_{index}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"unable to save {path.name}")
            paths.append(path)
    finally:
        if capture is not None:
            capture.release()

    return paths
