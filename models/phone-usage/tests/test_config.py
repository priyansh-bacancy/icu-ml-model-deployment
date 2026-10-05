import os
import unittest
from unittest.mock import patch

from config import Settings


class ConfigTests(unittest.TestCase):
    def test_footage_bucket_takes_precedence_for_eks(self):
        with patch.dict(
            os.environ,
            {
                "STORAGE_BACKEND": "s3",
                "FOOTAGE_BUCKET": "eks-bucket",
                "S3_BUCKET": "legacy-bucket",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.storage_bucket, "eks-bucket")

    def test_s3_bucket_remains_a_supported_fallback(self):
        with patch.dict(
            os.environ,
            {"STORAGE_BACKEND": "s3", "S3_BUCKET": "legacy-bucket"},
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.storage_bucket, "legacy-bucket")


if __name__ == "__main__":
    unittest.main()
