"""Verify that this image exposes the phone event response contract."""

from __future__ import annotations

import time

from config import EVENT_TYPE, Settings
from predict import _failed_block


settings = Settings(
    model_id="phone-usage-detection",
    model_version="v1.0.0",
    model_path="/app/weights/best.pt",
    storage_backend="local",
    storage_bucket=None,
    storage_root="/storage",
    aws_region="ap-southeast-2",
    confidence_threshold=0.25,
    image_size=1280,
    device="cpu",
)
result = _failed_block(
    settings, "phone-image-contract", time.perf_counter(), RuntimeError("test")
)
assert result["event_type"] == EVENT_TYPE == "phone_usage_event"
assert result["event_id"] == "phone-image-contract"
assert result["status"] == "FAILED"
print("response contract ok: phone_usage_event")
