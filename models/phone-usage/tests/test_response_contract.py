from __future__ import annotations

import time

from config import EVENT_TYPE, Settings
from predict import _failed_block


def _settings() -> Settings:
    return Settings(
        model_id="phone-usage-detection",
        model_version="v1.0.0",
        model_path="weights/best.pt",
        storage_backend="local",
        storage_bucket=None,
        storage_root="/tmp/storage",
        aws_region="ap-southeast-2",
        confidence_threshold=0.25,
        image_size=1280,
        device="cpu",
    )


def test_processing_failure_preserves_event_contract() -> None:
    result = _failed_block(
        _settings(), "phone-contract-test", time.perf_counter(), RuntimeError("boom")
    )

    assert list(result) == [
        "model_id",
        "event_type",
        "event_id",
        "model_version",
        "status",
        "processing_time_ms",
        "output_media",
        "detections",
        "error",
    ]
    assert result["event_type"] == EVENT_TYPE == "phone_usage_event"
    assert result["event_id"] == "phone-contract-test"
    assert result["status"] == "FAILED"
    assert result["detections"] == []
