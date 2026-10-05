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
    source_video = PROJECT_ROOT / "sleeping_video07.mp4"
    model_path = PROJECT_ROOT / "best-sleep.pt"
    if not source_video.is_file() or not model_path.is_file():
        raise FileNotFoundError("best-sleep.pt and sleeping_video07.mp4 are required")

    with tempfile.TemporaryDirectory(prefix="sleep-detection-local-e2e-") as temp:
        storage_root = Path(temp) / "storage"
        input_path = storage_root / "uploads" / source_video.name
        input_path.parent.mkdir(parents=True)
        shutil.copy2(source_video, input_path)

        os.environ.update(
            {
                "STORAGE_BACKEND": "local",
                "STORAGE_ROOT": str(storage_root),
                "MODEL_PATH": str(model_path),
                "CONFIDENCE_THRESHOLD": "0.70",
                "IMAGE_SIZE": "640",
                "TARGET_LABEL": "SLEEPING",
            }
        )

        import predict
        from config import Settings
        from storage import make_storage

        settings = Settings.from_env()
        assert settings.device == "cpu", settings.device
        predict.SETTINGS = settings
        predict.STORAGE = make_storage(settings)
        predict.RUNTIME.load(settings.model_path, settings.target_label)
        result = predict._predict(
            predict.PredictionRequest(
                job_id="local-sleep-e2e",
                footage_s3_key=f"uploads/{source_video.name}",
            )
        )

        assert result["status"] == "SUCCESS", result
        assert result["event_type"] == "sleep_event", result
        assert result["event_id"] == "local-sleep-e2e", result
        assert result["error"] is None, result
        assert result["detections"], "sample video produced no SLEEPING detection > 0.70"
        assert {item["label"] for item in result["detections"]} == {"SLEEPING"}
        instances = [
            instance
            for item in result["detections"]
            for instance in item["instances"]
        ]
        assert instances
        assert all(instance["confidence"] > 0.70 for instance in instances)

        output_media = result["output_media"]
        media_keys = [output_media["annotated_video_s3_key"]]
        media_keys.extend(output_media["annotated_image_s3_keys"])
        assert 1 <= len(output_media["annotated_image_s3_keys"]) <= 5
        for key in media_keys:
            path = storage_root / key
            assert path.is_file() and path.stat().st_size > 0, path

        training_labels = sorted(
            (storage_root / "training" / "local-sleep-e2e" / "sleep-detection" / "labels")
            .glob("*.txt")
        )
        assert training_labels, "no filtered training labels were exported"
        for label_file in training_labels:
            for row in label_file.read_text(encoding="utf-8").splitlines():
                assert row.split(maxsplit=1)[0] == "0", (label_file, row)

        print(
            "local e2e ok: "
            f"instances={len(instances)}; "
            f"confidence_range={min(x['confidence'] for x in instances):.4f}-"
            f"{max(x['confidence'] for x in instances):.4f}; "
            f"training_frames={len(training_labels)}; "
            f"processing_ms={result['processing_time_ms']}"
        )


if __name__ == "__main__":
    main()
