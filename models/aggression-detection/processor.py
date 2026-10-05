"""Window inference, frame confidence interpolation, and media annotation."""

from __future__ import annotations

import math
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import decord
import numpy as np
import torch

from config import AGGRESSION_THRESHOLD
from modeling import (
    ModelRuntime,
    decode_frame_context,
    preprocess_frames,
)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    duration_seconds: float
    width: int
    height: int


@dataclass(frozen=True)
class AggressionArtifacts:
    metadata: VideoMetadata
    frame_scores: np.ndarray
    annotated_video: Path


def interpolate_anchor_scores(
    anchor_frames: list[int], anchor_scores: list[float], frame_count: int
) -> np.ndarray:
    """Create a distinct displayed confidence for every source frame."""
    if not anchor_frames or len(anchor_frames) != len(anchor_scores):
        raise ValueError("anchor frames and scores must be non-empty and aligned")
    if anchor_frames[0] != 0 or anchor_frames[-1] != frame_count - 1:
        raise ValueError("anchor frames must include the first and final frame")
    return np.clip(
        np.interp(
            np.arange(frame_count, dtype=np.float64),
            np.asarray(anchor_frames, dtype=np.float64),
            np.asarray(anchor_scores, dtype=np.float64),
        ),
        0,
        1,
    )


def draw_confidence(frame: np.ndarray, confidence: float) -> np.ndarray:
    """Show confidence at top-left and a red border at the ABD threshold."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    panel_width = min(width - 16, max(190, round(width * 0.34)))
    panel_height = min(height - 16, max(58, round(height * 0.14)))
    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (8, 8),
        (8 + panel_width, 8 + panel_height),
        (15, 17, 20),
        -1,
    )
    annotated = cv2.addWeighted(overlay, 0.70, annotated, 0.30, 0)
    font_scale = max(0.42, min(0.75, width / 1000))
    cv2.putText(
        annotated,
        "Aggression",
        (18, min(34, panel_height)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"Confidence: {confidence:.4f}",
        (18, min(58, panel_height)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 0.78,
        (235, 238, 242),
        1,
        cv2.LINE_AA,
    )
    if confidence >= AGGRESSION_THRESHOLD:
        thickness = max(3, round(min(width, height) * 0.008))
        cv2.rectangle(
            annotated,
            (2, 2),
            (width - 3, height - 3),
            (30, 30, 235),
            thickness,
        )
    return annotated


class AggressionProcessor:
    def __init__(
        self,
        runtime: ModelRuntime,
        *,
        frame_context_seconds: float,
        score_interval_seconds: float,
        batch_size: int,
    ) -> None:
        self.runtime = runtime
        self.frame_context_seconds = frame_context_seconds
        self.score_interval_seconds = score_interval_seconds
        self.batch_size = batch_size

    def _score(self, source: Path) -> tuple[VideoMetadata, np.ndarray]:
        reader = decord.VideoReader(
            str(source), ctx=decord.cpu(0), num_threads=2
        )
        frame_count = len(reader)
        fps = float(reader.get_avg_fps())
        if frame_count <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("input video has invalid frame metadata")

        duration_seconds = frame_count / fps
        first_frame = reader[0].asnumpy()
        score_step_frames = max(1, round(self.score_interval_seconds * fps))
        anchor_frames = list(range(0, frame_count, score_step_frames))
        if anchor_frames[-1] != frame_count - 1:
            anchor_frames.append(frame_count - 1)
        anchor_scores: list[float] = []
        for offset in range(0, len(anchor_frames), self.batch_size):
            batch_frames = anchor_frames[offset : offset + self.batch_size]
            clips = [
                preprocess_frames(
                    decode_frame_context(
                        reader,
                        center_frame,
                        fps,
                        self.frame_context_seconds,
                    )
                )
                for center_frame in batch_frames
            ]
            confidences = self.runtime.predict(torch.stack(clips))
            anchor_scores.extend(float(value) for value in confidences)
            completed = min(offset + len(batch_frames), len(anchor_frames))
            if (
                offset == 0
                or completed == len(anchor_frames)
                or completed % (self.batch_size * 10) == 0
            ):
                LOGGER.info(
                    "Scored %d/%d live-confidence anchors",
                    completed,
                    len(anchor_frames),
                )

        metadata = VideoMetadata(
            fps=fps,
            frame_count=frame_count,
            duration_seconds=duration_seconds,
            width=int(first_frame.shape[1]),
            height=int(first_frame.shape[0]),
        )
        return metadata, interpolate_anchor_scores(
            anchor_frames, anchor_scores, frame_count
        )

    @staticmethod
    def _encode_for_browser(silent: Path, output: Path) -> None:
        """Encode the annotated frames as a fast-start, browser-safe H.264 MP4."""
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
                str(silent),
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
        silent.unlink()

    def _annotate(
        self,
        source: Path,
        workdir: Path,
        metadata: VideoMetadata,
        scores: np.ndarray,
    ) -> Path:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError("OpenCV could not decode the input video")
        silent_video = workdir / "annotated_silent.mp4"
        output_video = workdir / "annotated.mp4"
        writer = cv2.VideoWriter(
            str(silent_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata.fps,
            (metadata.width, metadata.height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("OpenCV could not initialize the MP4 writer")

        frame_number = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                confidence = float(scores[min(frame_number, scores.size - 1)])
                annotated = draw_confidence(frame, confidence)
                writer.write(annotated)
                frame_number += 1
        finally:
            capture.release()
            writer.release()

        if frame_number == 0:
            raise ValueError("input video contains no readable frames")
        self._encode_for_browser(silent_video, output_video)
        return output_video

    def process(self, source: Path, workdir: Path) -> AggressionArtifacts:
        metadata, scores = self._score(source)
        annotated_video = self._annotate(
            source, workdir, metadata, scores
        )
        return AggressionArtifacts(
            metadata=metadata,
            frame_scores=scores,
            annotated_video=annotated_video,
        )
