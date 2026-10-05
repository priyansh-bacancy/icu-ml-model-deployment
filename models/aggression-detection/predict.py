"""FastAPI service implementing SageMaker /ping and /invocations."""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, field_validator
from starlette.concurrency import run_in_threadpool

from config import AGGRESSION_THRESHOLD, Settings
from job import run_job, validate_request
from modeling import ModelRuntime
from storage import Storage, make_storage


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("videomae-aggression")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    job_id: str
    footage_s3_key: str

    @field_validator("job_id")
    @classmethod
    def validate_job_id(cls, value: str) -> str:
        return validate_request(
            {"job_id": value, "footage_s3_key": "uploads/placeholder.mp4"}
        )["job_id"]

    @field_validator("footage_s3_key")
    @classmethod
    def validate_footage_key(cls, value: str) -> str:
        return validate_request(
            {"job_id": "validation", "footage_s3_key": value}
        )["footage_s3_key"]

    def validated(self) -> dict[str, str]:
        return validate_request(self.model_dump())


class ServiceRuntime:
    def __init__(self) -> None:
        self.model: ModelRuntime | None = None
        self.lock = threading.Lock()

    def load(self, settings: Settings) -> None:
        LOGGER.info("Loading VideoMAE checkpoint from %s", settings.model_path)
        self.model = ModelRuntime.load(settings.model_path, settings.device)
        LOGGER.info(
            "VideoMAE ready on %s; threshold=%.2f; temperature=%.6f",
            self.model.device,
            AGGRESSION_THRESHOLD,
            self.model.temperature,
        )

    @property
    def ready(self) -> bool:
        return self.model is not None


SETTINGS: Settings | None = None
STORAGE: Storage | None = None
RUNTIME = ServiceRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    global SETTINGS, STORAGE
    SETTINGS = Settings.from_env()
    STORAGE = make_storage(SETTINGS)
    RUNTIME.load(SETTINGS)
    yield


app = FastAPI(
    title="ICU Solarcam VideoMAE Aggression Detection",
    version="1.0.0",
    lifespan=lifespan,
)


def _predict(payload: PredictionRequest) -> dict[str, Any]:
    if SETTINGS is None or STORAGE is None or RUNTIME.model is None:
        raise RuntimeError("service is not initialized")
    validated = payload.validated()
    with RUNTIME.lock:
        result = run_job(
            validated,
            runtime=RUNTIME.model,
            settings=SETTINGS,
            storage=STORAGE,
        )
    if result["status"] == "FAILED":
        LOGGER.error("Job %s failed: %s", validated["job_id"], result["error"])
    return result


@app.get("/ping")
def ping() -> Response:
    return Response(status_code=200 if RUNTIME.ready else 503)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if RUNTIME.ready else "loading",
        "model_loaded": RUNTIME.ready,
        "threshold": AGGRESSION_THRESHOLD,
    }


async def _handle(payload: PredictionRequest) -> dict[str, Any]:
    return await run_in_threadpool(_predict, payload)


@app.post("/invocations")
async def invocations(payload: PredictionRequest, request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0].lower()
    if content_type != "application/json":
        raise HTTPException(
            status_code=415, detail="Content-Type must be application/json"
        )
    return await _handle(payload)


@app.post("/predict")
async def predict(payload: PredictionRequest) -> dict[str, Any]:
    return await _handle(payload)
