"""Run the real sample video through the complete local-storage request path."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    source_video = PROJECT_ROOT / "aws-annotated-photoshoot.mp4"
    model_path = PROJECT_ROOT / "best.pt"
    if not source_video.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            "best.pt and aws-annotated-photoshoot.mp4 are required"
        )

    with tempfile.TemporaryDirectory(prefix="phone-usage-local-e2e-") as temp:
        storage_root = Path(temp) / "storage"
        input_path = storage_root / "uploads" / source_video.name
        input_path.parent.mkdir(parents=True)
        shutil.copy2(source_video, input_path)

        os.environ.update(
            {
                "STORAGE_BACKEND": "local",
                "STORAGE_ROOT": str(storage_root),
                "MODEL_PATH": str(model_path),
                "CONFIDENCE_THRESHOLD": "0.25",
                "IMAGE_SIZE": "1280",
                "DEVICE": "cpu",
            }
        )

        import predict
        from config import Settings
        from storage import make_storage

        settings = Settings.from_env()
        assert settings.device == "cpu", settings.device
        predict.SETTINGS = settings
        predict.STORAGE = make_storage(settings)
        predict.RUNTIME.load(settings.model_path)
        result = predict._predict(
            predict.PredictionRequest(
                job_id="local-phone-e2e",
                footage_s3_key=f"uploads/{source_video.name}",
            )
        )

        assert result["status"] == "SUCCESS", result
        assert result["event_type"] == "phone_usage_event", result
        assert result["event_id"] == "local-phone-e2e", result
        assert result["error"] is None, result
        assert result["detections"], "sample video produced no phone detection"
        assert {item["label"] for item in result["detections"]} <= {
            "in_hand",
            "on_ear",
        }

        output_media = result["output_media"]
        media_keys = [output_media["annotated_video_s3_key"]]
        media_keys.extend(output_media["annotated_image_s3_keys"])
        assert 1 <= len(output_media["annotated_image_s3_keys"]) <= 5
        for key in media_keys:
            path = storage_root / key
            assert path.is_file() and path.stat().st_size > 0, path

        instances = [
            instance
            for detection in result["detections"]
            for instance in detection["instances"]
        ]
        print(
            "local e2e ok: "
            f"groups={len(result['detections'])}; "
            f"instances={len(instances)}; "
            f"processing_ms={result['processing_time_ms']}"
        )


if __name__ == "__main__":
    main()
