"""SageMaker and ICU Manager HTTP service for phone detection."""

from __future__ import annotations

import logging
import re
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import FastAPI, Request, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.concurrency import run_in_threadpool
from ultralytics import YOLO

from builder import build_detections
from config import EVENT_TYPE, Settings
from media import MediaProcessor, export_training_artifacts
from snapshots import save_ranked_snapshots
from storage import Storage, make_storage, normalize_key


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("phone-usage-detection")
JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    job_id: str = Field(min_length=1, max_length=128)
    footage_s3_key: str = Field(min_length=1, max_length=1024)

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        if not JOB_ID_PATTERN.fullmatch(value):
            raise ValueError("job_id may contain only letters, numbers, '.', '_' and '-'")
        return value

    @field_validator("footage_s3_key")
    @classmethod
    def validate_footage_key(cls, value: str) -> str:
        return normalize_key(value)


class ModelRuntime:
    def __init__(self) -> None:
        self.model: YOLO | None = None
        self.lock = threading.Lock()

    def load(self, model_path: str) -> None:
        LOGGER.info("Loading phone detector from %s", model_path)
        self.model = YOLO(model_path)
        LOGGER.info("Model ready with classes: %s", self.model.names)

    @property
    def ready(self) -> bool:
        return self.model is not None


SETTINGS: Settings | None = None
STORAGE: Storage | None = None
RUNTIME = ModelRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global SETTINGS, STORAGE
    SETTINGS = Settings.from_env()
    STORAGE = make_storage(SETTINGS)
    RUNTIME.load(SETTINGS.model_path)
    yield


app = FastAPI(
    title="ICU Solarcam Phone Usage Detection",
    version="1.0.0",
    lifespan=lifespan,
)


def _failed_block(
    settings: Settings, event_id: str, started: float, exc: Exception
) -> dict[str, Any]:
    return {
        "model_id": settings.model_id,
        "event_type": EVENT_TYPE,
        "event_id": event_id,
        "model_version": settings.model_version,
        "status": "FAILED",
        "processing_time_ms": round((time.perf_counter() - started) * 1000),
        "output_media": {
            "annotated_video_s3_key": None,
            "annotated_image_s3_keys": [],
        },
        "detections": [],
        "error": {
            "code": "PROCESSING_ERROR",
            "message": str(exc)[:1000],
        },
    }


def _upload_training(storage: Storage, training_root: Path, prefix: str) -> None:
    if not training_root.exists():
        return
    for file_path in sorted(training_root.rglob("*")):
        if file_path.is_file():
            relative = file_path.relative_to(training_root).as_posix()
            storage.upload_file(file_path, f"{prefix}/{relative}")


def _predict(payload: PredictionRequest) -> dict[str, Any]:
    if SETTINGS is None or STORAGE is None or RUNTIME.model is None:
        raise RuntimeError("service is not initialized")

    settings = SETTINGS
    storage = STORAGE
    started = time.perf_counter()
    output_prefix = f"jobs/{payload.job_id}/{settings.model_id}"
    result_key = f"{output_prefix}/result.json"

    try:
        suffix = PurePosixPath(payload.footage_s3_key).suffix.lower() or ".bin"
        with tempfile.TemporaryDirectory(prefix="phone-usage-detection-") as temp:
            workdir = Path(temp)
            source = workdir / f"source{suffix}"
            storage.download_file(payload.footage_s3_key, source)

            processor = MediaProcessor(
                RUNTIME.model,
                confidence=settings.confidence_threshold,
                image_size=settings.image_size,
                device=settings.device,
            )
            with RUNTIME.lock:
                artifacts = processor.process(source, workdir)

            detections = build_detections(
                artifacts.flat_detections,
                is_video=artifacts.is_video,
                fps=artifacts.fps,
            )
            training_root = workdir / "training"
            export_training_artifacts(
                source,
                detections,
                is_video=artifacts.is_video,
                class_names=RUNTIME.model.names,
                training_root=training_root,
            )

            snapshot_paths = save_ranked_snapshots(
                source,
                detections,
                is_video=artifacts.is_video,
                workdir=workdir,
            )
            image_keys = []
            for index, snapshot_path in enumerate(snapshot_paths, start=1):
                image_key = f"{output_prefix}/frame_{index}.jpg"
                storage.upload_file(snapshot_path, image_key)
                image_keys.append(image_key)

            video_key: str | None = None
            if artifacts.annotated_video is not None:
                video_key = f"{output_prefix}/annotated.mp4"
                storage.upload_file(artifacts.annotated_video, video_key)

            training_prefix = f"training/{payload.job_id}/{settings.model_id}"
            _upload_training(storage, training_root, training_prefix)

            result = {
                "model_id": settings.model_id,
                "event_type": EVENT_TYPE,
                "event_id": payload.job_id,
                "model_version": settings.model_version,
                "status": "SUCCESS",
                "processing_time_ms": round((time.perf_counter() - started) * 1000),
                "output_media": {
                    "annotated_video_s3_key": video_key,
                    "annotated_image_s3_keys": image_keys,
                },
                "detections": detections,
                "error": None,
            }
            storage.put_json(result_key, result)
            return result
    except Exception as exc:
        LOGGER.exception("Job %s failed", payload.job_id)
        result = _failed_block(settings, payload.job_id, started, exc)
        try:
            storage.put_json(result_key, result)
        except Exception:
            LOGGER.exception("Failed to archive result for job %s", payload.job_id)
        return result


@app.get("/ping")
def ping() -> Response:
    return Response(status_code=200 if RUNTIME.ready else 503)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok" if RUNTIME.ready else "loading", "model_loaded": RUNTIME.ready}


async def _handle_prediction(payload: PredictionRequest) -> dict[str, Any]:
    return await run_in_threadpool(_predict, payload)


@app.post("/invocations")
async def invocations(payload: PredictionRequest, request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type != "application/json":
        # SageMaker maps this response to the InvokeEndpoint caller.
        from fastapi import HTTPException

        raise HTTPException(status_code=415, detail="Content-Type must be application/json")
    return await _handle_prediction(payload)


@app.post("/predict")
async def predict(payload: PredictionRequest) -> dict[str, Any]:
    return await _handle_prediction(payload)
