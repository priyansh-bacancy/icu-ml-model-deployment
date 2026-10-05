"""Image/video inference and artifact generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
import shutil
import subprocess
from typing import Any

import cv2

from extractor import extract_result


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
TRAINING_FRAME_STRIDE = 15


def transcode_for_browser(source: Path, output: Path) -> None:
    """Create a fast-start H.264 MP4 that is safe for browser playback."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode annotated videos")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-c:v",
            "libx264",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            "-an",
            str(output),
        ],
        check=True,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("ffmpeg did not produce an annotated MP4")


@dataclass(frozen=True)
class MediaArtifacts:
    is_video: bool
    fps: float | None
    flat_detections: list[dict[str, Any]]
    annotated_video: Path | None


class MediaProcessor:
    def __init__(
        self,
        model: Any,
        *,
        confidence: float,
        image_size: int,
        device: str,
    ) -> None:
        self.model = model
        self.confidence = confidence
        self.image_size = image_size
        self.device = device

    def _infer(self, frame: Any) -> Any:
        return self.model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
            save=False,
        )[0]

    def process(self, source: Path, workdir: Path) -> MediaArtifacts:
        if source.suffix.lower() in IMAGE_SUFFIXES:
            return self._process_image(source, workdir)
        return self._process_video(source, workdir)

    def _process_image(self, source: Path, workdir: Path) -> MediaArtifacts:
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"unable to decode image: {source.name}")

        result = self._infer(frame)
        flat = extract_result(result, self.model.names, 0)
        return MediaArtifacts(False, None, flat, None)

    def _process_video(self, source: Path, workdir: Path) -> MediaArtifacts:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError(f"unable to decode video: {source.name}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            fps = 25.0

        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            capture.release()
            raise ValueError("video contains no readable frames")

        height, width = first_frame.shape[:2]
        raw_video = workdir / "annotated_raw.mp4"
        annotated_video = workdir / "annotated.mp4"
        writer = cv2.VideoWriter(
            str(raw_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("failed to initialize MP4 video writer")

        flat: list[dict[str, Any]] = []
        frame_number = 0
        frame = first_frame

        try:
            while frame is not None:
                result = self._infer(frame)
                frame_detections = extract_result(result, self.model.names, frame_number)
                flat.extend(frame_detections)
                annotated = result.plot()
                writer.write(annotated)

                ok, next_frame = capture.read()
                if not ok:
                    break
                frame_number += 1
                frame = next_frame
        finally:
            writer.release()
            capture.release()

        transcode_for_browser(raw_video, annotated_video)
        raw_video.unlink()

        return MediaArtifacts(True, fps, flat, annotated_video)


def _class_ids_by_label(
    class_names: Mapping[int, str] | Sequence[str],
) -> dict[str, int]:
    if isinstance(class_names, Mapping):
        return {str(label): int(class_id) for class_id, label in class_names.items()}
    return {str(label): class_id for class_id, label in enumerate(class_names)}


def _normalize_yolo_box(
    class_id: int,
    bbox: dict[str, Any],
    frame_width: int,
    frame_height: int,
) -> str:
    """Convert a top-left pixel bbox to one clipped YOLO label row."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")

    x1 = max(0.0, min(float(frame_width), float(bbox["x"])))
    y1 = max(0.0, min(float(frame_height), float(bbox["y"])))
    x2 = max(0.0, min(float(frame_width), float(bbox["x"]) + float(bbox["w"])))
    y2 = max(0.0, min(float(frame_height), float(bbox["y"]) + float(bbox["h"])))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox has no area after clipping: {bbox}")

    center_x = ((x1 + x2) / 2) / frame_width
    center_y = ((y1 + y2) / 2) / frame_height
    width = (x2 - x1) / frame_width
    height = (y2 - y1) / frame_height
    return (
        f"{class_id} {center_x:.6f} {center_y:.6f} "
        f"{width:.6f} {height:.6f}"
    )


def _index_training_boxes(
    detections: list[dict[str, Any]],
    *,
    is_video: bool,
    class_names: Mapping[int, str] | Sequence[str],
) -> dict[int, list[tuple[int, dict[str, Any]]]]:
    """Index the public result JSON boxes by their clean source frame."""
    class_ids = _class_ids_by_label(class_names)
    indexed: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for detection in detections:
        label = str(detection["label"])
        if label not in class_ids:
            raise ValueError(f"result label is not present in model classes: {label}")
        class_id = class_ids[label]
        for instance in detection["instances"]:
            frame_number = (
                int(instance["attributes"]["frame_number"]) if is_video else 0
            )
            indexed.setdefault(frame_number, []).append((class_id, instance["bbox"]))

    for boxes in indexed.values():
        boxes.sort(key=lambda item: item[0])
    return dict(sorted(indexed.items()))


def _write_training_label(
    training_root: Path,
    frame_number: int,
    clean_frame: Any,
    boxes: list[tuple[int, dict[str, Any]]],
) -> None:
    height, width = clean_frame.shape[:2]
    label_path = training_root / "labels" / f"frame_{frame_number:05d}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        _normalize_yolo_box(class_id, bbox, width, height)
        for class_id, bbox in boxes
    ]
    label_path.write_text("\n".join(rows), encoding="utf-8")


def export_training_artifacts(
    source: Path,
    detections: list[dict[str, Any]],
    *,
    is_video: bool,
    class_names: Mapping[int, str] | Sequence[str],
    training_root: Path,
) -> list[int]:
    """Export a YOLO label file for every fifteenth source frame."""
    indexed = _index_training_boxes(
        detections,
        is_video=is_video,
        class_names=class_names,
    )

    if not is_video:
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"unable to decode image for training export: {source.name}")
        _write_training_label(training_root, 0, frame, indexed.get(0, []))
        return [0]

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"unable to decode video for training export: {source.name}")

    exported: list[int] = []
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_number % TRAINING_FRAME_STRIDE == 0:
                _write_training_label(
                    training_root,
                    frame_number,
                    frame,
                    indexed.get(frame_number, []),
                )
                exported.append(frame_number)
            frame_number += 1
    finally:
        capture.release()

    if not exported:
        raise ValueError(f"video contains no readable frames: {source.name}")
    return exported
