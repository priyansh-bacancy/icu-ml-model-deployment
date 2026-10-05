"""Invoke a running local container and validate its saved result contract."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18083/invocations")
    parser.add_argument("--job-id", default="fight-05-local")
    parser.add_argument("--footage-key", default="uploads/fight_05.avi")
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()

    body = json.dumps(
        {"job_id": args.job_id, "footage_s3_key": args.footage_key}
    ).encode("utf-8")
    request = urllib.request.Request(
        args.url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=3600) as response:
        result = json.loads(response.read())

    assert result["model_id"] == "videomae-aggression"
    assert result["event_type"] == "aggression_event"
    assert result["event_id"] == args.job_id
    expected_version = os.getenv("EXPECTED_MODEL_VERSION", "v1.0.5")
    assert result["model_version"] == expected_version
    assert result["status"] == "SUCCESS", result.get("error")
    assert result["error"] is None
    assert result["output_media"]["annotated_video_s3_key"]
    image_keys = result["output_media"]["annotated_image_s3_keys"]
    assert isinstance(image_keys, list)
    assert len(image_keys) <= 5
    for detection in result["detections"]:
        assert detection["count"] == len(detection["instances"])
        for instance in detection["instances"]:
            attributes = instance["attributes"]
            assert instance["confidence"] >= 0.70
            assert attributes["start_frame"] <= attributes["highest_conf_frame"]
            assert attributes["highest_conf_frame"] <= attributes["end_frame"]

    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
