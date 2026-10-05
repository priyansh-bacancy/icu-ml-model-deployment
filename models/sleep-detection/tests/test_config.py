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

    def test_sleep_defaults_match_checkpoint_and_contract(self):
        with patch.dict(
            os.environ,
            {
                "STORAGE_BACKEND": "local",
                "STORAGE_ROOT": "test-storage",
                "CONFIDENCE_THRESHOLD": "0.70",
                "IMAGE_SIZE": "640",
                "TARGET_LABEL": "SLEEPING",
            },
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.model_id, "sleep-detection")
        self.assertEqual(settings.confidence_threshold, 0.70)
        self.assertEqual(settings.image_size, 640)
        self.assertEqual(settings.target_label, "SLEEPING")

    def test_rejects_out_of_range_threshold(self):
        with patch.dict(
            os.environ,
            {"STORAGE_BACKEND": "local", "CONFIDENCE_THRESHOLD": "1.01"},
            clear=True,
        ), self.assertRaises(ValueError):
            Settings.from_env()


if __name__ == "__main__":
    unittest.main()
