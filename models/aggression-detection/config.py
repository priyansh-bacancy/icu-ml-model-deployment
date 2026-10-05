"""Environment-backed settings for VideoMAE aggression inference."""

from __future__ import annotations

import os
from dataclasses import dataclass


# ABD deliberately has one decision threshold. It is not environment tunable so
# local validation and the eventual SageMaker image cannot silently diverge.
AGGRESSION_THRESHOLD = 0.70
EVENT_TYPE = "aggression_event"


def _positive_float(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
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
    device: str
    frame_context_seconds: float
    score_interval_seconds: float
    batch_size: int

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
            model_id=os.getenv("MODEL_ID", "videomae-aggression").strip(),
            model_version=os.getenv("MODEL_VERSION", "v1").strip(),
            model_path=os.getenv(
                "MODEL_PATH", "/app/weights/best_model.pt"
            ).strip(),
            storage_backend=backend,
            storage_bucket=bucket,
            storage_root=os.getenv("STORAGE_ROOT", "/storage").strip(),
            aws_region=os.getenv("AWS_REGION", "ap-southeast-2").strip(),
            device=os.getenv("DEVICE", "auto").strip().lower(),
            frame_context_seconds=_positive_float("FRAME_CONTEXT_SECONDS", 5.0),
            score_interval_seconds=_positive_float("SCORE_INTERVAL_SECONDS", 1.0),
            batch_size=_positive_int("BATCH_SIZE", 2),
        )
