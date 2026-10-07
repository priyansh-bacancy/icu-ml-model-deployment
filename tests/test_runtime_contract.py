"""Recovery regression tests: no cloud access or checkpoint execution."""

import json
import subprocess
from pathlib import Path
from types import ModuleType

import pytest


def payload(runtime: ModuleType) -> dict:
    request = {"job_id": "unit-job", "footage_s3_key": "uploads/test.mp4"}
    if runtime.MODEL_ID == "restricted-zone":
        request["zones_s3_key"] = "uploads/zones.json"
    return request


def test_accepts_documented_request(runtime: ModuleType) -> None:
    request = payload(runtime)

    parsed = runtime.input_fn(
        json.dumps(request).encode(), "application/json; charset=utf-8"
    )

    assert parsed == request


@pytest.mark.parametrize("key", ["unknown", "event_type"])
def test_rejects_extra_keys(runtime: ModuleType, key: str) -> None:
    request = {**payload(runtime), key: "unexpected"}

    with pytest.raises(ValueError, match="exactly"):
        runtime.input_fn(json.dumps(request), "application/json")


@pytest.mark.parametrize(
    "key", ["source.mp4", "../uploads/test.mp4", "/uploads/test.mp4", "uploads/a.txt"]
)
def test_rejects_invalid_media_key(runtime: ModuleType, key: str) -> None:
    request = {**payload(runtime), "footage_s3_key": key}

    with pytest.raises(ValueError):
        runtime.input_fn(json.dumps(request), "application/json")


@pytest.mark.parametrize("job_id", ["", "../job", "job space", "a" * 64])
def test_rejects_invalid_job_id(runtime: ModuleType, job_id: str) -> None:
    request = {**payload(runtime), "job_id": job_id}

    with pytest.raises(ValueError, match="job_id"):
        runtime.input_fn(json.dumps(request), "application/json")


def test_rejects_malformed_json(runtime: ModuleType) -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        runtime.input_fn("{broken", "application/json")


def test_rejects_non_json_content_type(runtime: ModuleType) -> None:
    with pytest.raises(ValueError):
        runtime.input_fn(json.dumps(payload(runtime)), "text/plain")


def test_response_has_nine_fields_and_plural_snapshots(runtime: ModuleType) -> None:
    response = runtime._result_block(
        "unit-job",
        "SUCCESS",
        123,
        runtime._output_media("jobs/unit-job/video.mp4", ["jobs/unit-job/frame_1.jpg"]),
        [],
        None,
    )

    serialized, content_type = runtime.output_fn(response, "application/json")

    assert set(response) == {
        "model_id",
        "event_type",
        "event_id",
        "model_version",
        "status",
        "processing_time_ms",
        "output_media",
        "detections",
        "error",
    }
    assert response["model_id"] == runtime.MODEL_ID
    assert response["event_id"] == "unit-job"
    assert response["output_media"]["annotated_image_s3_keys"] == [
        "jobs/unit-job/frame_1.jpg"
    ]
    assert content_type == "application/json"
    assert json.loads(serialized) == response


def test_transcode_uses_browser_codec_and_removes_intermediate(
    runtime: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.mp4"
    final = tmp_path / "final.mp4"
    raw.write_bytes(b"raw")
    commands = []

    def run(command: list[str], **options: object) -> None:
        commands.append(command)
        assert options["check"] is True
        final.write_bytes(b"synthetic-output")

    monkeypatch.setattr(runtime.subprocess, "run", run)

    runtime._transcode_to_browser_mp4(raw, final)

    assert "libx264" in commands[0]
    assert "yuv420p" in commands[0]
    assert "+faststart" in commands[0]
    assert not raw.exists()


def test_failed_transcode_keeps_source_and_reports_failure(
    runtime: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"raw")

    def run(command: list[str], **options: object) -> None:
        raise subprocess.CalledProcessError(1, command, stderr="encoder failed")

    monkeypatch.setattr(runtime.subprocess, "run", run)

    with pytest.raises(RuntimeError, match="transcode"):
        runtime._transcode_to_browser_mp4(raw, tmp_path / "final.mp4")

    assert raw.exists()


def test_prelabels_sample_frame_zero_and_empty_frames(
    runtime: ModuleType,
    tmp_path: Path,
) -> None:
    if runtime.MODEL_ID == "restricted-zone":
        from zone_pipeline import _is_sampled_frame, _write_prelabel
    else:
        _is_sampled_frame = runtime._is_sampled_frame
        _write_prelabel = runtime._write_prelabel

    label = tmp_path / "frame_000000.txt"

    _write_prelabel([], label, 640, 480)

    assert [frame for frame in range(46) if _is_sampled_frame(frame)] == [0, 15, 30, 45]
    assert label.read_text() == ""


def test_predict_failure_retains_contract(
    runtime: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writes = []
    if runtime.MODEL_ID == "mmc-vehicle-classification":
        monkeypatch.setattr(
            runtime, "_persist_input", lambda request: writes.append(request)
        )
        monkeypatch.setattr(
            runtime, "_persist_result", lambda job, result: writes.append(result)
        )
    else:
        monkeypatch.setattr(
            runtime,
            "_persist_json",
            lambda job, filename, result: writes.append(result),
        )

    def fail(request: dict, model: object) -> dict:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(runtime, "_run_prediction", fail)

    response = runtime.predict_fn(payload(runtime), {})

    assert response["status"] == "FAILED"
    assert response["error"]["code"] == "INFERENCE_ERROR"
    assert response["detections"] == []
    assert writes[-1] == response
