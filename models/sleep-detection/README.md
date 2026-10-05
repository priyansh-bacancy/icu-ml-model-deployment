# ICU Solarcam sleep-detection container

This directory contains the local implementation of the Sleep Detection service,
following the existing Phone Usage Detection request, response, storage, and
container pattern. No AWS resources are created or modified by these files.

The checkpoint is evaluated at `imgsz=640`. Although it contains `SLEEPING` and
`AWAKE` classes, the public contract exposes and annotates only `SLEEPING` boxes
whose four-decimal response confidence is strictly greater than `0.70`.

## Routes and contract

- `GET /ping`: SageMaker health check.
- `POST /invocations`: SageMaker inference route.
- `GET /health`: local/manager health alias.
- `POST /predict`: local/manager inference alias.

Both inference routes accept exactly:

```json
{"job_id":"job-1","footage_s3_key":"uploads/sleeping_video07.mp4"}
```

Every successful or processing-failure result includes the fixed
`event_type` and echoes the input `job_id` as `event_id`:

```json
{
  "model_id": "sleep-detection",
  "event_type": "sleep_event",
  "event_id": "job-1",
  "model_version": "v1.0.0",
  "status": "SUCCESS",
  "processing_time_ms": 12345,
  "output_media": {
    "annotated_video_s3_key": "jobs/job-1/sleep-detection/annotated.mp4",
    "annotated_image_s3_keys": [
      "jobs/job-1/sleep-detection/frame_1.jpg"
    ]
  },
  "detections": [],
  "error": null
}
```

Output media uses `jobs/{job_id}/sleep-detection/`. Clean detected frames and
class-0 YOLO labels use `training/{job_id}/sleep-detection/`.
Up to five confidence-ranked detections are returned as snapshot keys, with
each snapshot drawn on its actual source frame and containing only its box.

The live endpoint loads its checkpoint from the ECR image. A matching archival
five-file SageMaker package is stored separately at
`s3://icu-solarcam-storage-bacancy/models/sleep-detection/model.tar.gz`; the
live SageMaker model intentionally has no `ModelDataUrl`, matching Phone Usage.

## Local validation

From this directory, with the dependencies installed:

```powershell
python -B -m unittest discover -s tests -v
python -B scripts/smoke_test.py
python -B scripts/local_e2e_test.py
```

To build the Linux SageMaker-compatible image after Docker Desktop is running:

```powershell
docker build --platform linux/amd64 --provenance=false `
  -f Dockerfile.gpu `
  -t icu-solarcam/sleep-detection:v1.0.3-gpu .
docker run --rm --platform linux/amd64 `
  icu-solarcam/sleep-detection:v1.0.3-gpu python /app/scripts/smoke_test.py
```

## AWS deployment

The deployment scripts mirror the Phone Usage Detection ECR, Async Inference,
and scale-to-zero pattern. They use versioned SageMaker model/config names so an
endpoint update can be rolled back to its previous endpoint configuration.

```powershell
.\scripts\push_to_ecr.ps1 -AccountId 123456789012
.\scripts\deploy_sagemaker.ps1 `
  -AccountId 123456789012 `
  -ExecutionRoleArn arn:aws:iam::123456789012:role/example `
  -S3Bucket example-bucket
aws sagemaker wait endpoint-in-service `
  --endpoint-name sleep-detection-endpoint --region ap-southeast-2
.\scripts\configure_async_autoscaling.ps1
```

## Runtime configuration

| Variable | Default | Purpose |
|---|---:|---|
| `S3_BUCKET` | required in S3 mode | Input/output bucket |
| `AWS_REGION` | `ap-southeast-2` | Bucket and endpoint region |
| `MODEL_ID` | `sleep-detection` | Contract model identifier |
| `MODEL_VERSION` | `v1.0.0` | Contract model version |
| `CONFIDENCE_THRESHOLD` | `0.70` | Strict public threshold (`confidence > 0.70`) |
| `IMAGE_SIZE` | `640` | Inference size used during training |
| `TARGET_LABEL` | `SLEEPING` | Only class exposed and annotated |
| `DEVICE` | `cpu` | Ultralytics device (`cpu`, `0`, etc.) |
| `STORAGE_BACKEND` | `s3` | Set to `local` for local testing |
| `STORAGE_ROOT` | `/storage` | Mounted local storage root |
