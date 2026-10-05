import os
from unittest.mock import patch

from config import Settings


def test_eks_environment_controls_identity_version_and_bucket() -> None:
    with patch.dict(
        os.environ,
        {
            "STORAGE_BACKEND": "s3",
            "FOOTAGE_BUCKET": "eks-bucket",
            "S3_BUCKET": "legacy-bucket",
            "MODEL_ID": "videomae-aggression",
            "MODEL_VERSION": "v1.0.5",
        },
        clear=True,
    ):
        settings = Settings.from_env()

    assert settings.storage_bucket == "eks-bucket"
    assert settings.model_id == "videomae-aggression"
    assert settings.model_version == "v1.0.5"


def test_s3_bucket_remains_a_supported_fallback() -> None:
    with patch.dict(
        os.environ,
        {"STORAGE_BACKEND": "s3", "S3_BUCKET": "legacy-bucket"},
        clear=True,
    ):
        settings = Settings.from_env()

    assert settings.storage_bucket == "legacy-bucket"
