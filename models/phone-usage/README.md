# ICU Solarcam phone-usage-detection container

This directory is a SageMaker-compatible inference image for `best.pt`.
The model detects the trained classes `in_hand` and `on_ear` and implements the
ICU Solarcam request, response, and output-media contracts.

## What the service exposes

- `GET /ping` — SageMaker health check.
- `POST /invocations` — SageMaker inference route.
- `GET /health` — Manager/local health alias.
- `POST /predict` — Manager/local inference alias.

Both inference routes accept only:

```json
{"job_id":"job-1","footage_s3_key":"uploads/test1.mp4"}
```

Every successful or processing-failure result includes the fixed
`event_type` and echoes the input `job_id` as `event_id`:

```json
{
  "model_id": "phone-usage-detection",
  "event_type": "phone_usage_event",
  "event_id": "job-1",
  "model_version": "v1.0.0",
  "status": "SUCCESS",
  "processing_time_ms": 12345,
  "output_media": {
    "annotated_video_s3_key": "jobs/job-1/phone-usage-detection/annotated.mp4",
    "annotated_image_s3_keys": [
      "jobs/job-1/phone-usage-detection/frame_1.jpg"
    ]
  },
  "detections": [],
  "error": null
}
```

The service downloads the input from `S3_BUCKET`, runs YOLO, uploads annotated
media under `jobs/{job_id}/phone-usage-detection/`, uploads one clean image and
matching YOLO label file per unique detected frame under
`training/{job_id}/phone-usage-detection/`, and returns the required per-model
block without archiving a job-level `result.json`. For video, every detected box is
returned with `attributes.frame_number` and `attributes.timestamp_ms`, while
`best_frame_number` identifies the representative frame for each class. Up to
five confidence-ranked instances are saved as `frame_1.jpg` through
`frame_5.jpg`, each using the instance's actual source frame and only its box.

## 1. Build and verify locally

Prerequisites: Docker Desktop running in Linux-container mode. From this folder:

```powershell
docker build --platform linux/amd64 --provenance=false `
  -f Dockerfile.gpu `
  -t icu-solarcam/phone-usage-detection:v1.0.2-gpu .
docker run --rm --platform linux/amd64 `
  icu-solarcam/phone-usage-detection:v1.0.2-gpu `
  python /app/scripts/smoke_test.py
```

The smoke test loads the baked weights, verifies both class names, and performs
one inference. The same steps are wrapped by `scripts/build_image.ps1`.

For an end-to-end request without AWS, make a local storage tree and mount it:

```powershell
New-Item -ItemType Directory -Force .\local-storage\uploads | Out-Null
Copy-Item C:\path\to\test.mp4 .\local-storage\uploads\test.mp4
docker run --rm -p 8080:8080 `
  -e STORAGE_BACKEND=local -e STORAGE_ROOT=/storage `
  -v "${PWD}\local-storage:/storage" `
  icu-solarcam/phone-usage-detection:v1.0.2-gpu
```

Then, in another terminal:

```powershell
$Body = @{job_id="job-1"; footage_s3_key="uploads/test.mp4"} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8080/predict `
  -ContentType application/json -Body $Body
```

## 2. Push to ECR in Sydney

Install/configure AWS CLI v2, then run:

```powershell
.\scripts\push_to_ecr.ps1 -AccountId 126378326989 -Tag v1.0.2-gpu
```

This creates `icu-solarcam/phone-usage-detection` if needed, enables scan-on-push,
builds `linux/amd64`, repeats the in-container model smoke test, and pushes only
after it passes.

## 3. Deploy to SageMaker

The SageMaker execution role must be able to pull the ECR image and read/write
the selected S3 bucket. All resources must be in `ap-southeast-2`.

Async Inference is the default for video jobs because processing can exceed the
normal real-time/serverless request duration:

```powershell
.\scripts\deploy_sagemaker.ps1 `
  -AccountId 123456789012 `
  -ExecutionRoleArn arn:aws:iam::123456789012:role/icu-sagemaker-execution `
  -S3Bucket my-icu-solarcam-bucket `
  -ImageTag latest `
  -Mode Async -InstanceType ml.m5.xlarge
```

For a short synchronous test, use a real-time instance:

```powershell
.\scripts\deploy_sagemaker.ps1 `
  -AccountId 123456789012 `
  -ExecutionRoleArn arn:aws:iam::123456789012:role/icu-sagemaker-execution `
  -S3Bucket my-icu-solarcam-bucket `
  -ImageTag latest `
  -Mode Instance -InstanceType ml.m5.xlarge
```

For Async Inference, send the small request JSON inline after the endpoint
reaches `InService`. Route SageMaker's single response directly to the
canonical model result key; do not create async input objects or UUID `.out`
objects:

```powershell
$Body = '{"job_id":"job-1","footage_s3_key":"uploads/test1.mp4"}'
aws sagemaker-runtime invoke-endpoint-async `
  --endpoint-name phone-usage-detection-endpoint `
  --content-type application/json `
  --body $Body `
  --cli-binary-format raw-in-base64-out `
  --inference-id job-1 `
  --s3-output-path-extension "job-1/phone-usage-detection/" `
  --filename result.json `
  --region ap-southeast-2
```

Delete a test endpoint immediately when finished to stop endpoint charges:

```powershell
aws sagemaker delete-endpoint `
  --endpoint-name phone-usage-detection-endpoint `
  --region ap-southeast-2
```

For a persistent Async endpoint, configure queue-based scale-to-zero and the
CloudWatch queue alarms after the endpoint reaches `InService`:

```powershell
.\scripts\configure_async_autoscaling.ps1 `
  -EndpointName phone-usage-detection-endpoint `
  -MinCapacity 0 -MaxCapacity 1
```

The `phone-usage-detection-endpoint-queue-empty` CloudWatch alarm scales the single
instance to zero after three empty one-minute periods. The
`HasBacklogWithoutCapacity` alarm wakes the endpoint when a request arrives at
zero capacity. Pass `-AlarmNotificationTopicArn` to also send the queue-empty
alarm to an existing SNS topic.

## Runtime configuration

| Variable | Default | Purpose |
|---|---:|---|
| `S3_BUCKET` | required in S3 mode | Input/output bucket |
| `AWS_REGION` | `ap-southeast-2` | Bucket and endpoint region |
| `MODEL_ID` | `phone-usage-detection` | Contract model identifier |
| `MODEL_VERSION` | `v1.0.0` | Contract model version |
| `CONFIDENCE_THRESHOLD` | `0.25` | YOLO box threshold |
| `IMAGE_SIZE` | `1280` | YOLO inference size used by this model |
| `DEVICE` | `cpu` | Ultralytics device (`cpu`, `0`, etc.) |
| `STORAGE_BACKEND` | `s3` | Set `local` only for local testing |
| `STORAGE_ROOT` | `/storage` | Mounted local storage root |
