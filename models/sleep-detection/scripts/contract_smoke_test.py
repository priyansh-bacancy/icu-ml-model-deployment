"""Verify that this image exposes the sleep event response contract."""

from __future__ import annotations

import time

from config import EVENT_TYPE, Settings
from predict import _failed_block


settings = Settings(
    model_id="sleep-detection",
    model_version="v1.0.0",
    model_path="/app/weights/best.pt",
    storage_backend="local",
    storage_bucket=None,
    storage_root="/storage",
    aws_region="ap-southeast-2",
    confidence_threshold=0.70,
    image_size=640,
    device="cpu",
    target_label="SLEEPING",
)
result = _failed_block(
    settings, "sleep-image-contract", time.perf_counter(), RuntimeError("test")
)
assert result["event_type"] == EVENT_TYPE == "sleep_event"
assert result["event_id"] == "sleep-image-contract"
assert result["status"] == "FAILED"
print("response contract ok: sleep_event")
