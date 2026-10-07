import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


CONFIDENCE_THRESHOLD = 0.50
MIN_DIRECTION_HISTORY = 5
DIRECTION_MISMATCH_THRESHOLD = -0.30
DEFAULT_DWELL_THRESHOLD_SECONDS = 5.0
STATIONARY_RADIUS_PX = 55
STATIONARY_SPEED_PX_PER_SECOND = 25
DWELL_ANCHOR_EMA_ALPHA = 0.15
DWELL_GRACE_SECONDS = 0.6
CLASS_NAMES = {0: "vehicle", 1: "person"}
FRAME_SAMPLE_INTERVAL = 15
CLASS_COLORS = {
    0: (80, 200, 80),
    1: (255, 180, 0),
}
ZONE_COLORS = {
    "restricted": (70, 23, 255),
    "entry": (0, 214, 255),
    "exit": (255, 121, 40),
    "dwell": (200, 60, 170),
    "capacity": (255, 229, 0),
}


@dataclass
class VideoResult:
    detections: list[dict]
    prelabel_samples: list[dict]
    fps: float
    frame_count: int
    width: int
    height: int


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
        class_counts[CLASS_NAMES[record["class_id"]]] += 1
    return {
        "frame_number": frame_number,
        "timestamp_ms": round(frame_number / fps * 1000),
        "label_s3_key": label_key,
        "has_predictions": bool(records),
        "prediction_count": len(records),
        "class_counts": dict(class_counts),
        "label_file_empty": not records,
    }


def _zone_points(zone: dict) -> np.ndarray:
    points = zone.get("points")
    if isinstance(points, list) and len(points) >= 3:
        return np.asarray(points, dtype=np.int32)
    return np.asarray(
        [
            [zone["x1"], zone["y1"]],
            [zone["x2"], zone["y1"]],
            [zone["x2"], zone["y2"]],
            [zone["x1"], zone["y2"]],
        ],
        dtype=np.int32,
    )


def _allowed_class_ids(zone: dict) -> set[int]:
    applies_to = zone.get("applies_to", "both")
    if applies_to == "both":
        return set(CLASS_NAMES)
    return {
        class_id
        for class_id, class_name in CLASS_NAMES.items()
        if class_name == applies_to
    }


def _direction_vector(zone: dict) -> tuple[float, float] | None:
    if zone["type"] not in {"entry", "exit"}:
        return None
    raw_vector = zone.get("direction_vector")
    if raw_vector is None:
        legacy = zone.get("direction", "right" if zone["type"] == "entry" else "left")
        raw_vector = {
            "right": (1.0, 0.0),
            "left": (-1.0, 0.0),
            "down": (0.0, 1.0),
            "up": (0.0, -1.0),
        }[legacy]
    vx, vy = (float(value) for value in raw_vector)
    magnitude = math.hypot(vx, vy)
    if magnitude == 0:
        raise ValueError("Zone direction vector must not be zero")
    return vx / magnitude, vy / magnitude


def _direction_degrees(vx: float, vy: float) -> float:
    return math.degrees(math.atan2(vy, vx)) % 360.0


def prepare_zones(zone_config: dict, width: int, height: int) -> list[dict]:
    if zone_config.get("orig_w") != width or zone_config.get("orig_h") != height:
        raise ValueError(
            "Zone dimensions do not match video dimensions: "
            f"zones={zone_config.get('orig_w')}x{zone_config.get('orig_h')}, "
            f"video={width}x{height}"
        )

    prepared = []
    zone_ids = set()
    for index, raw_zone in enumerate(zone_config.get("zones", []), start=1):
        zone_id = raw_zone.get("id", f"zone-{index:04d}")
        if not isinstance(zone_id, str) or not zone_id.strip():
            raise ValueError(f"Zone {index} id must be a non-empty string")
        if zone_id in zone_ids:
            raise ValueError(f"Duplicate zone id: {zone_id}")
        zone_ids.add(zone_id)
        points = _zone_points(raw_zone)
        if np.any(points[:, 0] < 0) or np.any(points[:, 0] >= width):
            raise ValueError(f"Zone {index} contains an out-of-range x coordinate")
        if np.any(points[:, 1] < 0) or np.any(points[:, 1] >= height):
            raise ValueError(f"Zone {index} contains an out-of-range y coordinate")
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points.reshape((-1, 1, 2))], 255)
        prepared.append(
            {
                **raw_zone,
                "zone_id": zone_id,
                "points_array": points,
                "mask": mask,
                "allowed_class_ids": _allowed_class_ids(raw_zone),
                "direction": _direction_vector(raw_zone),
            }
        )
    if not prepared:
        raise ValueError("At least one zone is required")
    return prepared


def _movement_vector(history: deque) -> tuple[float, float] | None:
    if len(history) < MIN_DIRECTION_HISTORY:
        return None
    points = list(history)
    middle = len(points) // 2
    start_x = sum(point[0] for point in points[:middle]) / middle
    start_y = sum(point[1] for point in points[:middle]) / middle
    end_count = len(points) - middle
    end_x = sum(point[0] for point in points[middle:]) / end_count
    end_y = sum(point[1] for point in points[middle:]) / end_count
    dx, dy = end_x - start_x, end_y - start_y
    magnitude = math.hypot(dx, dy)
    if magnitude < 1:
        return None
    return dx / magnitude, dy / magnitude


def _instant_speed(history: deque, fps: float) -> float:
    if len(history) < MIN_DIRECTION_HISTORY:
        return 0.0
    start_x, start_y = history[0]
    end_x, end_y = history[-1]
    elapsed_seconds = (len(history) - 1) / fps
    if elapsed_seconds <= 0:
        return 0.0
    return math.hypot(end_x - start_x, end_y - start_y) / elapsed_seconds


def _inside(zone: dict, anchor: tuple[int, int]) -> bool:
    x, y = anchor
    return bool(zone["mask"][y, x])


def _base_observation(zone: dict) -> dict:
    return {
        "zone_id": zone["zone_id"],
        "zone_type": zone["type"],
        "applies_to": zone.get("applies_to", "both"),
        "inside": True,
        "violated": False,
        "rule": None,
    }


def _draw_zones(frame: np.ndarray, zones: list[dict]) -> None:
    overlay = frame.copy()
    for zone in zones:
        color = ZONE_COLORS[zone["type"]]
        points = zone["points_array"].reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(frame, [points], True, color, 2, cv2.LINE_AA)
        x, y = zone["points_array"].min(axis=0)
        label = (
            f"{zone['zone_id']} {zone['type'].upper()} "
            f"({zone.get('applies_to', 'both')})"
        )
        cv2.putText(
            frame,
            label,
            (int(x) + 4, max(int(y) + 18, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
        direction = zone["direction"]
        if direction is not None:
            center = zone["points_array"].mean(axis=0)
            vx, vy = direction
            start = (
                round(center[0] - vx * 40),
                round(center[1] - vy * 40),
            )
            end = (
                round(center[0] + vx * 40),
                round(center[1] + vy * 40),
            )
            cv2.arrowedLine(
                frame,
                start,
                end,
                (0, 230, 118),
                2,
                cv2.LINE_AA,
                tipLength=0.35,
            )
    cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)


def _build_observations(
    tracked: list[dict],
    zones: list[dict],
    histories: dict[int, deque],
    dwell_states: dict[tuple[int, str], dict],
    frame_number: int,
    fps: float,
) -> list[list[dict]]:
    observations = [[] for _ in tracked]
    present_dwell_keys = set()
    for zone in zones:
        matching_indices = [
            index
            for index, item in enumerate(tracked)
            if item["class_id"] in zone["allowed_class_ids"]
            and _inside(zone, item["anchor"])
        ]
        occupancy = {
            class_name: sum(
                tracked[index]["class_id"] == class_id for index in matching_indices
            )
            for class_id, class_name in CLASS_NAMES.items()
        }

        for index in matching_indices:
            item = tracked[index]
            observation = _base_observation(zone)
            if zone["type"] == "restricted":
                observation.update(
                    violated=True,
                    rule="presence_not_allowed",
                )
            elif zone["type"] in {"entry", "exit"}:
                movement = _movement_vector(histories[item["internal_track_id"]])
                if movement is not None:
                    allowed_x, allowed_y = zone["direction"]
                    alignment = movement[0] * allowed_x + movement[1] * allowed_y
                    observation.update(
                        allowed_direction_deg=round(
                            float(
                                zone.get(
                                    "direction_deg",
                                    _direction_degrees(allowed_x, allowed_y),
                                )
                            ),
                            2,
                        ),
                        observed_direction_deg=round(
                            _direction_degrees(*movement),
                            2,
                        ),
                        direction_alignment=round(alignment, 4),
                        violated=alignment < DIRECTION_MISMATCH_THRESHOLD,
                        rule=(
                            "wrong_direction"
                            if alignment < DIRECTION_MISMATCH_THRESHOLD
                            else None
                        ),
                    )
            elif zone["type"] == "dwell":
                state_key = (item["internal_track_id"], zone["zone_id"])
                present_dwell_keys.add(state_key)
                anchor_x, anchor_y = item["anchor"]
                state = dwell_states.get(state_key)
                if state is None:
                    state = {
                        "start_frame": frame_number,
                        "anchor": (float(anchor_x), float(anchor_y)),
                        "last_seen_frame": frame_number,
                    }
                    dwell_states[state_key] = state
                else:
                    reference_x, reference_y = state["anchor"]
                    drift = math.hypot(
                        anchor_x - reference_x,
                        anchor_y - reference_y,
                    )
                    speed = _instant_speed(
                        histories[item["internal_track_id"]],
                        fps,
                    )
                    if (
                        drift > STATIONARY_RADIUS_PX
                        or speed > STATIONARY_SPEED_PX_PER_SECOND
                    ):
                        state = {
                            "start_frame": frame_number,
                            "anchor": (float(anchor_x), float(anchor_y)),
                            "last_seen_frame": frame_number,
                        }
                        dwell_states[state_key] = state
                    else:
                        state["anchor"] = (
                            reference_x
                            + DWELL_ANCHOR_EMA_ALPHA * (anchor_x - reference_x),
                            reference_y
                            + DWELL_ANCHOR_EMA_ALPHA * (anchor_y - reference_y),
                        )
                        state["last_seen_frame"] = frame_number

                threshold_seconds = float(
                    zone.get(
                        "dwell_threshold_sec",
                        DEFAULT_DWELL_THRESHOLD_SECONDS,
                    )
                )
                dwell_seconds = (frame_number - state["start_frame"]) / fps
                violated = dwell_seconds >= threshold_seconds
                observation.update(
                    dwell_seconds=round(dwell_seconds, 2),
                    threshold_seconds=round(threshold_seconds, 2),
                    stationary=True,
                    violated=violated,
                    rule="dwell_time_exceeded" if violated else None,
                )
            elif zone["type"] == "capacity":
                limits = zone.get(
                    "capacity_limits",
                    {"mode": "combined", "max_count": 5},
                )
                if limits["mode"] == "combined":
                    count = sum(occupancy.values())
                    limit = int(limits["max_count"])
                else:
                    class_name = CLASS_NAMES[item["class_id"]]
                    count = occupancy[class_name]
                    limit = int(limits[f"max_{class_name}"])
                observation.update(
                    occupancy=count,
                    max_count=limit,
                    violated=count > limit,
                    rule="capacity_exceeded" if count > limit else None,
                )
            observations[index].append(observation)

    grace_frames = max(1, round(fps * DWELL_GRACE_SECONDS))
    stale_keys = [
        state_key
        for state_key, state in dwell_states.items()
        if state_key not in present_dwell_keys
        and frame_number - state["last_seen_frame"] > grace_frames
    ]
    for state_key in stale_keys:
        del dwell_states[state_key]
    return observations


def _annotate(frame: np.ndarray, tracked: list[dict]) -> None:
    for item in tracked:
        x, y, width, height = item["bbox"].values()
        violation = item["attributes"]["has_zone_violation"]
        color = (0, 0, 255) if violation else CLASS_COLORS[item["class_id"]]
        cv2.rectangle(
            frame,
            (x, y),
            (x + width, y + height),
            color,
            2,
            cv2.LINE_AA,
        )
        label = CLASS_NAMES[item["class_id"]]
        if violation:
            label += " ZONE VIOLATION"
        cv2.putText(
            frame,
            f"{label} {item['confidence']:.2f}",
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def process_video(
    model,
    input_path: Path,
    output_path: Path,
    zone_config: dict,
    device: str,
    prelabel_root: Path,
    prelabel_prefix: str,
) -> VideoResult:
    capture = cv2.VideoCapture(str(input_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not capture.isOpened() or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Input video has invalid metadata")

    zones = prepare_zones(zone_config, width, height)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create annotated MP4")

    history_length = max(round(fps * 0.5), 8)
    histories = defaultdict(lambda: deque(maxlen=history_length))
    dwell_states = {}
    class_frames: dict[str, list[dict]] = defaultdict(list)
    prelabel_directory = prelabel_root / "labels"
    prelabel_directory.mkdir(parents=True)
    prelabel_samples = []

    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.track(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            device=device,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]
        tracked = []
        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for box in boxes:
                class_id = int(box.cls.item())
                if class_id not in CLASS_NAMES:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                anchor = (
                    int(np.clip((x1 + x2) / 2, 0, width - 1)),
                    int(np.clip(y2, 0, height - 1)),
                )
                internal_track_id = int(box.id.item())
                histories[internal_track_id].append(anchor)
                tracked.append(
                    {
                        "internal_track_id": internal_track_id,
                        "class_id": class_id,
                        "confidence": round(float(box.conf.item()), 4),
                        "bbox": {
                            "x": round(x1),
                            "y": round(y1),
                            "w": round(x2 - x1),
                            "h": round(y2 - y1),
                        },
                        "anchor": anchor,
                    }
                )

        zone_observations = _build_observations(
            tracked,
            zones,
            histories,
            dwell_states,
            frame_number,
            fps,
        )
        label_key = None
        if _is_sampled_frame(frame_number):
            label_key = _prelabel_key(
                prelabel_prefix,
                frame_number,
            )
            label_path = prelabel_directory / Path(label_key).name
            _write_prelabel(
                tracked,
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
                        tracked,
                    ),
                }
            )
        for item, observations in zip(
            tracked,
            zone_observations,
            strict=True,
        ):
            item["attributes"] = {
                "anchor_point": {
                    "x": item["anchor"][0],
                    "y": item["anchor"][1],
                },
                "has_zone_violation": any(
                    observation["violated"] for observation in observations
                ),
                "zone_observations": observations,
            }
            if label_key is not None:
                item["attributes"]["prelabel_s3_key"] = label_key
            class_frames[CLASS_NAMES[item["class_id"]]].append(
                {
                    "frame": frame_number,
                    "confidence": item["confidence"],
                    "bbox": item["bbox"],
                    "attributes": item["attributes"],
                }
            )

        _draw_zones(frame, zones)
        _annotate(frame, tracked)
        writer.write(frame)
        frame_number += 1

    capture.release()
    writer.release()
    if frame_number == 0:
        raise ValueError("Input video contains no decodable frames")

    detections = []
    for detection_index, (label, records) in enumerate(
        class_frames.items(),
        start=1,
    ):
        by_frame: dict[int, list[dict]] = defaultdict(list)
        for record in records:
            by_frame[record["frame"]].append(record)
        selected_frame = max(
            by_frame,
            key=lambda frame: (
                any(
                    record["attributes"]["has_zone_violation"]
                    for record in by_frame[frame]
                ),
                sum(
                    record["confidence"]
                    for record in by_frame[frame]
                    if record["attributes"]["has_zone_violation"]
                ),
                sum(record["confidence"] for record in by_frame[frame]),
                len(by_frame[frame]),
            ),
        )
        chosen = by_frame[selected_frame]
        violating_records = [
            record for record in records if record["attributes"]["has_zone_violation"]
        ]
        time_records = violating_records or records
        detections.append(
            {
                "detection_id": f"d-{detection_index:04d}",
                "label": label,
                "count": len(chosen),
                "best_frame_number": selected_frame,
                "first_seen_ms": round(
                    min(record["frame"] for record in time_records) / fps * 1000
                ),
                "last_seen_ms": round(
                    max(record["frame"] for record in time_records) / fps * 1000
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
                    for record in chosen
                ],
            }
        )
    return VideoResult(
        detections=detections,
        prelabel_samples=prelabel_samples,
        fps=fps,
        frame_count=frame_number,
        width=width,
        height=height,
    )
