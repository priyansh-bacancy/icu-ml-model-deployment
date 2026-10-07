"""Expose model inference through SageMaker-compatible HTTP routes."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, Request, Response
from starlette.concurrency import run_in_threadpool


MODEL_DIRECTORY = "/app/weights"
LOCAL_S3_ROOT = os.getenv("LOCAL_S3_ROOT")
inference = importlib.import_module("inference")


class LocalS3Client:
    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()

    def _object_path(self, bucket: str, key: str) -> Path:
        object_path = (self.root / bucket / key).resolve()
        if not object_path.is_relative_to(self.root):
            raise ValueError("S3 key resolves outside the local storage root")
        return object_path

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        shutil.copyfile(self._object_path(bucket, key), filename)

    def head_object(self, Bucket: str, Key: str) -> dict:
        object_path = self._object_path(Bucket, Key)
        content = object_path.read_bytes()
        return {
            "ContentLength": len(content),
            "ETag": f'"{hashlib.md5(content, usedforsecurity=False).hexdigest()}"',
        }

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict | None = None,
    ) -> None:
        del ExtraArgs
        destination = self._object_path(bucket, key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(filename, destination)

    def put_object(
        self,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
    ) -> None:
        del ContentType
        destination = self._object_path(Bucket, Key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Body)


if LOCAL_S3_ROOT:
    local_s3_client = LocalS3Client(LOCAL_S3_ROOT)
    inference.boto3 = SimpleNamespace(client=lambda service: local_s3_client)


runtime = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global runtime
    runtime = inference.model_fn(MODEL_DIRECTORY)
    yield


app = FastAPI(title="ICU Solarcam Model Inference", lifespan=lifespan)


@app.get("/ping")
def ping() -> Response:
    return Response(status_code=200 if runtime is not None else 503)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if runtime is not None else "loading",
        "model_loaded": runtime is not None,
    }


def _invoke(request_body: bytes, content_type: str, accept: str) -> Response:
    if runtime is None:
        raise RuntimeError("service is not initialized")
    payload = inference.input_fn(request_body, content_type)
    prediction = inference.predict_fn(payload, runtime)
    response_body, response_type = inference.output_fn(prediction, accept)
    return Response(content=response_body, media_type=response_type)


async def _handle_request(request: Request) -> Response:
    content_type = request.headers.get(
        "content-type",
        "application/octet-stream",
    )
    accept = request.headers.get("accept", "application/json")
    request_body = await request.body()
    try:
        return await run_in_threadpool(
            _invoke,
            request_body,
            content_type,
            accept,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=json.dumps({"error": type(error).__name__}),
        ) from error


@app.post("/invocations")
async def invocations(request: Request) -> Response:
    return await _handle_request(request)


@app.post("/predict")
async def predict(request: Request) -> Response:
    return await _handle_request(request)
