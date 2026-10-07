# fall-detection

Recovered runtime version **v7**, event type **fall_event**.
Entrypoint: `entrypoint.py` starts `predict:app`; lifespan loads
`inference.model_fn("/app/weights")`. GPU is selected when CUDA is available,
otherwise CPU. CPU MMC requires sufficient RAM; fallback is not a latency guarantee.

Required checkpoint placement (not committed):
- `weights/best.pt`

Detector class IDs in order: 0=falling, 1=sitting, 2=standing.
Fall detector inference uses imgsz=640 and confidence=0.25.

## Build

From repository root, after restoring weights:
```powershell
docker build -f docker/Dockerfile.gpu-base -t icu-solarcam/pytorch-gpu:recovery-v1 .
docker build -f models/fall-detection/Dockerfile.gpu -t icu-solarcam/fall-detection:<immutable-tag> models/fall-detection
```
The shared GPU base recipe was reconstructed from image history.
Rebuilt dependencies and security findings must be validated before production;
this backup does not claim a fresh full build or scan passed.

## Runtime configuration

Required: `FOOTAGE_BUCKET`. Optional: `OUTPUT_BUCKET` (defaults to input bucket),
`AWS_DEFAULT_REGION`, `PORT`, `YOLO_CONFIG_DIR`.
No access keys in containers; use IRSA.

## Local serving

```powershell
docker run --rm -p 8083:8080 -e FOOTAGE_BUCKET=test-bucket -e OUTPUT_BUCKET=test-bucket -e LOCAL_S3_ROOT=/local-s3 -v "<absolute-local-s3-directory>:/local-s3" icu-solarcam/fall-detection:<immutable-tag>
```
Add `--gpus all` for local GPU execution. Without it CPU fallback is used.
Create local inputs under `<local-s3>/test-bucket/uploads/`.

`LOCAL_S3_ROOT` is a test adapter only; omit it in AWS.

## Request, response, deployment

Use `input.example.json`; exact keys are required, extra keys rejected.
All S3 keys must be relative and start with `uploads/`.

`response-contract.example.json` is a synthetic contract example, not measured output.
Confidence is `detections[].instances[].confidence`.
See [shared handoff](../../docs/RECOVERED_MODEL_HANDOFF.md) for validation,
S3 layout, cleanup, permissions and deployment commands.
Service: `http://fall-detection/invocations` in the default namespace.
