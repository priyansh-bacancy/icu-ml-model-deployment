from __future__ import annotations

import pytest

from config import EVENT_TYPE, Settings
from job import _failed_result, validate_request


def test_request_contract_is_exact() -> None:
    assert validate_request(
        {"job_id": "fight-05", "footage_s3_key": "uploads/fight_05.avi"}
    ) == {
        "job_id": "fight-05",
        "footage_s3_key": "uploads/fight_05.avi",
    }


def test_processing_failure_preserves_event_contract() -> None:
    settings = Settings(
        model_id="videomae-aggression",
        model_version="v1",
        model_path="weights/best_model.pt",
        storage_backend="local",
        storage_bucket=None,
        storage_root="/tmp/storage",
        aws_region="ap-southeast-2",
        device="cpu",
        frame_context_seconds=5.0,
        score_interval_seconds=1.0,
        batch_size=2,
    )
    result = _failed_result(settings, "abd-contract-test", 0.0, RuntimeError("boom"))

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
    assert result["event_type"] == EVENT_TYPE == "aggression_event"
    assert result["event_id"] == "abd-contract-test"
    assert result["status"] == "FAILED"
    assert result["detections"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {"footage_s3_key": "uploads/fight_05.avi"},
        {
            "job_id": "fight-05",
            "footage_s3_key": "uploads/fight_05.avi",
            "threshold": 0.1,
        },
        {"job_id": "../bad", "footage_s3_key": "uploads/fight_05.avi"},
        {"job_id": "fight-05", "footage_s3_key": "../fight_05.avi"},
        {
            "job_id": "fight-05",
            "footage_s3_key": "uploads/aggression-behavior-detection/fight_05.avi",
        },
        {"job_id": "fight-05", "footage_s3_key": "videos/fight_05.avi"},
    ],
)
def test_invalid_requests_are_rejected(payload: dict) -> None:
    with pytest.raises(ValueError):
        validate_request(payload)
