"""Local and S3 storage adapters with identical object-key semantics."""

from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from config import Settings


def normalize_key(key: str) -> str:
    raw = key.strip().replace("\\", "/")
    candidate = PurePosixPath(raw)
    if not raw or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("storage key must be relative and must not contain '..'")
    return candidate.as_posix()


class Storage(Protocol):
    def download_file(self, key: str, destination: Path) -> None: ...

    def upload_file(self, source: Path, key: str) -> None: ...

    def put_json(self, key: str, value: dict[str, Any]) -> None: ...


class LocalStorage:
    def __init__(self, root: str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / normalize_key(key)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("storage key escapes STORAGE_ROOT")
        return path

    def download_file(self, key: str, destination: Path) -> None:
        source = self._path(key)
        if not source.is_file():
            raise FileNotFoundError(f"input object not found: {normalize_key(key)}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def upload_file(self, source: Path, key: str) -> None:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    def put_json(self, key: str, value: dict[str, Any]) -> None:
        destination = self._path(key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(value, indent=2), encoding="utf-8")


class S3Storage:
    def __init__(self, bucket: str, region: str) -> None:
        import boto3

        self.bucket = bucket
        self.client = boto3.client("s3", region_name=region)

    def download_file(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, normalize_key(key), str(destination))

    def upload_file(self, source: Path, key: str) -> None:
        content_types = {".mp4": "video/mp4", ".jpg": "image/jpeg"}
        extra_args = {"ContentType": content_types[source.suffix.lower()]}
        self.client.upload_file(
            str(source), self.bucket, normalize_key(key), ExtraArgs=extra_args
        )

    def put_json(self, key: str, value: dict[str, Any]) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=normalize_key(key),
            Body=json.dumps(value, separators=(",", ":")).encode("utf-8"),
            ContentType="application/json",
        )


def make_storage(settings: Settings) -> Storage:
    if settings.storage_backend == "local":
        return LocalStorage(settings.storage_root)
    assert settings.storage_bucket is not None
    return S3Storage(settings.storage_bucket, settings.aws_region)
