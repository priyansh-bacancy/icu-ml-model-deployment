"""Environment-backed configuration for the phone-usage-detection service."""

from __future__ import annotations

import os
from dataclasses import dataclass


EVENT_TYPE = "phone_usage_event"


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _float_env(name: str, default: float, minimum: float, maximum: float) -> float:
    value = float(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


@dataclass(frozen=True)
class Settings:
    model_id: str
    model_version: str
    model_path: str
    storage_backend: str
    storage_bucket: str | None
    storage_root: str
    aws_region: str
    confidence_threshold: float
    image_size: int
    device: str

    @classmethod
    def from_env(cls) -> "Settings":
        backend = os.getenv("STORAGE_BACKEND", "s3").strip().lower()
        if backend not in {"s3", "local"}:
            raise ValueError("STORAGE_BACKEND must be 's3' or 'local'")

        bucket = (
            os.getenv("FOOTAGE_BUCKET", "").strip()
            or os.getenv("S3_BUCKET", "").strip()
            or None
        )
        if backend == "s3" and not bucket:
            raise ValueError(
                "FOOTAGE_BUCKET or S3_BUCKET is required when STORAGE_BACKEND=s3"
            )

        return cls(
            model_id=os.getenv("MODEL_ID", "phone-usage-detection").strip(),
            model_version=os.getenv("MODEL_VERSION", "v1.0.0").strip(),
            model_path=os.getenv("MODEL_PATH", "weights/best.pt"),
            storage_backend=backend,
            storage_bucket=bucket,
            storage_root=os.getenv("STORAGE_ROOT", "/storage"),
            aws_region=os.getenv("AWS_REGION", "ap-southeast-2"),
            confidence_threshold=_float_env("CONFIDENCE_THRESHOLD", 0.25, 0.0, 1.0),
            image_size=_int_env("IMAGE_SIZE", 1280, 32),
            device=os.getenv("DEVICE", "cpu").strip(),
        )
