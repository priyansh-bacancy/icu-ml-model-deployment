"""Run the short fight sample through the complete local-storage job path."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    source_video = PROJECT_ROOT / "fight_05.avi"
    model_path = PROJECT_ROOT / "best_videoMAE.pt"
    if not source_video.is_file() or not model_path.is_file():
        raise FileNotFoundError("best_videoMAE.pt and fight_05.avi are required")

    job_id = "local-aggression-e2e"
    with tempfile.TemporaryDirectory(prefix="aggression-local-e2e-") as temp:
        storage_root = Path(temp) / "storage"
        input_path = storage_root / "uploads" / source_video.name
        input_path.parent.mkdir(parents=True)
        shutil.copy2(source_video, input_path)

        os.environ.update(
            {
                "STORAGE_BACKEND": "local",
                "STORAGE_ROOT": str(storage_root),
                "MODEL_PATH": str(model_path),
                "DEVICE": "cpu",
                "FRAME_CONTEXT_SECONDS": "5.0",
                "SCORE_INTERVAL_SECONDS": "1.0",
                "BATCH_SIZE": "2",
            }
        )

        from config import Settings
        from job import run_job
        from modeling import ModelRuntime
        from storage import make_storage

        settings = Settings.from_env()
        runtime = ModelRuntime.load(settings.model_path, settings.device)
        result = run_job(
            {"job_id": job_id, "footage_s3_key": f"uploads/{source_video.name}"},
            runtime=runtime,
            settings=settings,
            storage=make_storage(settings),
        )

        assert result["status"] == "SUCCESS", result
        assert result["event_type"] == "aggression_event", result
        assert result["event_id"] == job_id, result
        assert result["error"] is None, result

        output_media = result["output_media"]
        media_keys = [output_media["annotated_video_s3_key"]]
        media_keys.extend(output_media["annotated_image_s3_keys"])
        assert len(output_media["annotated_image_s3_keys"]) <= 5
        for key in media_keys:
            path = storage_root / key
            assert path.is_file() and path.stat().st_size > 0, path

        result_path = (
            storage_root / "jobs" / job_id / "videomae-aggression" / "result.json"
        )
        archived = json.loads(result_path.read_text(encoding="utf-8"))
        assert archived == result

        print(
            "local e2e ok: "
            f"groups={len(result['detections'])}; "
            f"processing_ms={result['processing_time_ms']}"
        )


if __name__ == "__main__":
    main()
