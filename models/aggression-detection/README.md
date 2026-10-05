# VideoMAE Aggression Behavior Detection (ABD)

This folder is the local-first SageMaker package for ICU Solarcam aggression
behavior detection. It uses the exact VideoMAE Base architecture and
preprocessing from `abd-videomae.ipynb`, exposes SageMaker-compatible HTTP
routes, and writes the same object-key layout locally that it will use in S3.

## Output behavior

- There is one fixed confidence threshold: **0.70**.
- A moving five-second VideoMAE context is centered and scored every second.
  Boundary frames use replicated context rather than one fixed full-video
  score. Anchor scores are interpolated across the source timeline, so the
  displayed confidence updates on every frame.
- Every consecutive run of frames with confidence `>= 0.70` is one aggression
  clip and one entry in `instances`.
- `instances[].confidence` is the highest frame confidence inside that clip.
- `instances[].attributes.highest_conf_frame` is that clip's peak frame.
- `best_frame_number` uses the highest-confidence frame in the entire video.
- Up to five detected aggression clips produce confidence-ranked snapshots;
  each snapshot uses that clip's `highest_conf_frame`.
- The annotated video and snapshots always show confidence at top-left.
- A red frame border is drawn only when confidence is `>= 0.70`.
- VideoMAE is temporal and does not localize a person, so `bbox` covers the
  complete source frame.

The service receives exactly:

```json
{
  "job_id": "fight-05-local",
  "footage_s3_key": "uploads/fight_05.avi"
}
```

`footage_s3_key` must always be a direct child of `uploads/`, for example
`uploads/fight_04.mp4`. ABD deliberately rejects nested input paths such as
`uploads/aggression-behavior-detection/fight_04.mp4` so its S3 input layout
matches the other ICU Solarcam models.

It writes:

```text
local-s3/
  uploads/fight_05.avi
  jobs/fight-05-local/videomae-aggression/
    annotated.mp4
    frame_1.jpg
    result.json
```

The same relative keys are used in the configured S3 bucket in AWS mode.

## Required trained checkpoint

Keep the final notebook checkpoint at:

```text
best_videoMAE.pt
```

The file must contain `model_state` from `abd-videomae.ipynb`. The final
notebook export also contains the calibration `temperature`, which is loaded
automatically. The generic Kinetics VideoMAE checkpoint is not an aggression
classifier and must not be substituted for this trained checkpoint.

## Run the real local test

Start Docker Desktop, then run from this folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_fight_05.ps1
```

The script builds `icu-solarcam/videomae-aggression:local`, starts it with the
filesystem S3 adapter, invokes `/invocations` using `fight_05.avi`, validates
the returned contract, and stores the three final artifacts under
`local-s3/jobs/fight-05-local/videomae-aggression/`.

Run any other local video with the same image and contract:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_local_video.ps1 `
  -VideoName "fight_04.mp4" `
  -JobId "fight-04-local"
```

## Manual Docker invocation

```powershell
$Root = (Resolve-Path ".").Path
$Storage = Join-Path $Root "local-s3"

docker build -t icu-solarcam/videomae-aggression:local .

docker run --rm `
  --name icu-videomae-test `
  -p 18083:8080 `
  --mount "type=bind,source=$Storage,target=/storage" `
  -e STORAGE_BACKEND=local `
  -e STORAGE_ROOT=/storage `
  -e DEVICE=cpu `
  icu-solarcam/videomae-aggression:local
```

The SageMaker boundary is:

- `GET /ping`
- `POST /invocations`
- `Content-Type: application/json`
- request body containing only `job_id` and `footage_s3_key`

## Result contract example

Actual frame numbers, timestamps, segment count, confidence values, and
processing time are produced by the real test video.

```json
{
  "model_id": "videomae-aggression",
  "event_type": "aggression_event",
  "event_id": "fight-05-local",
  "model_version": "v1",
  "status": "SUCCESS",
  "processing_time_ms": 10912,
  "output_media": {
    "annotated_video_s3_key": "jobs/fight-05-local/videomae-aggression/annotated.mp4",
    "annotated_image_s3_keys": [
      "jobs/fight-05-local/videomae-aggression/frame_1.jpg"
    ]
  },
  "detections": [
    {
      "detection_id": "d-0001",
      "label": "Aggression",
      "count": 1,
      "best_frame_number": 452,
      "first_seen_ms": 15000,
      "last_seen_ms": 22500,
      "instances": [
        {
          "confidence": 0.8206,
          "bbox": {"x": 0, "y": 0, "w": 768, "h": 432},
          "attributes": {
            "segment_id": 1,
            "start_ms": 15000,
            "end_ms": 22500,
            "start_frame": 450,
            "end_frame": 674,
            "highest_conf_frame": 452,
            "mean_confidence": 0.7912
          }
        }
      ]
    }
  ],
  "error": null
}
```

When no frame reaches `0.70`, inference is still `SUCCESS`, both media files
are saved, and `detections` is an empty list.

## SageMaker/AWS mode

Release `v1.0.1` uses SageMaker Async Inference. Live resource
names, verified AWS tests, autoscaling behavior, and the ECR security scan gate
are recorded in
[ABD_SAGEMAKER_ASYNC_DEPLOYMENT_RECORD.md](ABD_SAGEMAKER_ASYNC_DEPLOYMENT_RECORD.md).

SageMaker request JSON is sent inline and is not stored as an S3 async-input
object. The invocation must set `S3OutputPathExtension` to
`{job_id}/videomae-aggression/` and `Filename` to `result.json`, so SageMaker's
single async response lands beside the media under
`jobs/{job_id}/videomae-aggression/`. No UUID `.out` or separate async-output
object is allowed.

Set:

```text
STORAGE_BACKEND=s3
S3_BUCKET=<ICU Solarcam bucket>
AWS_REGION=ap-southeast-2
MODEL_PATH=/app/weights/best_model.pt
```

The custom container is already compatible with SageMaker's appended `serve`
command. `inference.py` also provides `model_fn`, `input_fn`, `predict_fn`, and
`output_fn` for a future inference-toolkit/model-tar packaging route.

Keep the tested threshold at `0.70`; it is intentionally not configurable.
`FRAME_CONTEXT_SECONDS` defaults to `5.0` and `SCORE_INTERVAL_SECONDS`
defaults to `1.0`; these affect temporal sampling, not the decision threshold.
