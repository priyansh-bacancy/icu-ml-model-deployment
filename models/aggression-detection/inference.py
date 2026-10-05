"""SageMaker inference-toolkit handler using the same production pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from config import Settings
from job import run_job, validate_request
from modeling import ModelRuntime
from storage import make_storage


def model_fn(model_dir: str) -> ModelRuntime:
    settings = Settings.from_env()
    model_data_path = Path(model_dir) / "best_model.pt"
    checkpoint_path = (
        str(model_data_path) if model_data_path.is_file() else settings.model_path
    )
    return ModelRuntime.load(checkpoint_path, settings.device)


def input_fn(
    request_body: str | bytes, content_type: str = "application/json"
) -> dict[str, str]:
    if content_type.split(";", 1)[0].lower() != "application/json":
        raise ValueError("Content-Type must be application/json")
    body = request_body.decode("utf-8") if isinstance(request_body, bytes) else request_body
    return validate_request(json.loads(body))


def predict_fn(payload: dict[str, str], runtime: ModelRuntime) -> dict:
    settings = Settings.from_env()
    return run_job(
        payload,
        runtime=runtime,
        settings=settings,
        storage=make_storage(settings),
    )


def output_fn(prediction: dict, accept: str = "application/json") -> tuple[str, str]:
    if accept.split(";", 1)[0].lower() not in {"application/json", "*/*"}:
        raise ValueError("Accept must be application/json")
    return json.dumps(prediction), "application/json"
