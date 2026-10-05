"""Verify that this image exposes the aggression event response contract."""

from __future__ import annotations

import time

from config import EVENT_TYPE, Settings
from job import _failed_result


settings = Settings(
    model_id="videomae-aggression",
    model_version="v1",
    model_path="/app/weights/best_model.pt",
    storage_backend="local",
    storage_bucket=None,
    storage_root="/storage",
    aws_region="ap-southeast-2",
    device="cpu",
    frame_context_seconds=5.0,
    score_interval_seconds=1.0,
    batch_size=2,
)
result = _failed_result(
    settings, "aggression-image-contract", time.perf_counter(), RuntimeError("test")
)
assert result["event_type"] == EVENT_TYPE == "aggression_event"
assert result["event_id"] == "aggression-image-contract"
assert result["status"] == "FAILED"
print("response contract ok: aggression_event")
