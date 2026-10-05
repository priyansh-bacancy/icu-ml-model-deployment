from __future__ import annotations

import numpy as np

from builder import build_detections, validate_contract


def test_consecutive_frames_become_one_instance() -> None:
    scores = np.array([0.2, 0.70, 0.81, 0.76, 0.4])
    detections = build_detections(
        scores,
        fps=10.0,
        width=768,
        height=432,
        threshold=0.70,
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection["count"] == 1
    assert detection["best_frame_number"] == 2
    assert detection["first_seen_ms"] == 100
    assert detection["last_seen_ms"] == 400
    instance = detection["instances"][0]
    assert instance["confidence"] == 0.81
    assert instance["bbox"] == {"x": 0, "y": 0, "w": 768, "h": 432}
    assert instance["attributes"] == {
        "segment_id": 1,
        "start_ms": 100,
        "end_ms": 400,
        "start_frame": 1,
        "end_frame": 3,
        "highest_conf_frame": 2,
        "mean_confidence": 0.7567,
    }
    validate_contract(detections)


def test_separated_runs_become_separate_instances() -> None:
    scores = np.array([0.72, 0.71, 0.2, 0.8, 0.9])
    detection = build_detections(
        scores,
        fps=25.0,
        width=320,
        height=240,
        threshold=0.70,
    )[0]

    assert detection["count"] == 2
    assert detection["instances"][0]["attributes"]["start_frame"] == 0
    assert detection["instances"][0]["attributes"]["end_frame"] == 1
    assert detection["instances"][1]["attributes"]["start_frame"] == 3
    assert detection["instances"][1]["attributes"]["end_frame"] == 4
    assert detection["best_frame_number"] == 4


def test_no_threshold_crossing_returns_no_detection() -> None:
    detections = build_detections(
        np.array([0.2, 0.6999, 0.4]),
        fps=30.0,
        width=1920,
        height=1080,
        threshold=0.70,
    )
    assert detections == []


def test_flat_peak_uses_temporal_midpoint() -> None:
    detection = build_detections(
        np.full(150, 0.9889),
        fps=30.0,
        width=1280,
        height=720,
        threshold=0.70,
    )[0]
    assert detection["best_frame_number"] == 74
    assert detection["instances"][0]["attributes"]["highest_conf_frame"] == 74
