# mmc-vehicle-classification

Recovered runtime version **v8**, event type **vehicle_event**.
Entrypoint: `entrypoint.py` starts `predict:app`; lifespan loads
`inference.model_fn("/app/weights")`. GPU is selected when CUDA is available,
otherwise CPU. CPU MMC requires sufficient RAM; fallback is not a latency guarantee.

Required checkpoint placement (not committed):
- `weights/vehicle_detector.pt`
- `weights/color_best.pt`
- `weights/make_model_best.pt`

Detector class IDs in order: 0=car, 1=truck, 2=bus, 3=motorcycle.
The pipeline combines a four-class YOLO vehicle detector with two ViT-B-16
embedding checkpoints (color and make/model) and FAISS similarity search.
Detector uses imgsz=1280 and confidence=0.30. FAISS indexes are separate external
artifacts, not embedded in Git.

## Build

From repository root, after restoring weights:
```powershell
docker build -f docker/Dockerfile.gpu-base -t icu-solarcam/pytorch-gpu:recovery-v1 .
docker build -f models/mmc-vehicle-classification/Dockerfile.gpu -t icu-solarcam/mmc-vehicle-classification:<immutable-tag> models/mmc-vehicle-classification
```
The shared GPU base recipe was reconstructed from image history.
Rebuilt dependencies and security findings must be validated before production;
this backup does not claim a fresh full build or scan passed.

## Runtime configuration

Required: `FOOTAGE_BUCKET`. Optional: `OUTPUT_BUCKET` (defaults to input bucket),
`AWS_DEFAULT_REGION`, `PORT`, `YOLO_CONFIG_DIR`.
No access keys in containers; use IRSA.

MMC: `FAISS_BUCKET` (defaults to FOOTAGE_BUCKET),
`FAISS_S3_PREFIX=models/mmc-vehicle-classification/faiss_db/v1`,
`FAISS_LOCAL_DIRECTORY=/tmp/mmc-vector-db`. At startup it downloads:
- faiss/color_index.faiss
- faiss/color_index_metadata.json
- faiss/make_model_registry.faiss
- faiss/make_model_registry_metadata.json

Full recovered FAISS backup additionally contains the fallback make_model_index
pair, catalog CSV and manifest. Keep them together for reproducibility.

## Local serving

```powershell
docker run --rm -p 8083:8080 -e FOOTAGE_BUCKET=test-bucket -e OUTPUT_BUCKET=test-bucket -e LOCAL_S3_ROOT=/local-s3 -v "<absolute-local-s3-directory>:/local-s3" icu-solarcam/mmc-vehicle-classification:<immutable-tag>
```
Add `--gpus all` for local GPU execution. Without it CPU fallback is used.
Create local inputs under `<local-s3>/test-bucket/uploads/`.
Mirror the FAISS prefix under the same local test bucket before startup.
`LOCAL_S3_ROOT` is a test adapter only; omit it in AWS.

## Request, response, deployment

Use `input.example.json`; exact keys are required, extra keys rejected.
All S3 keys must be relative and start with `uploads/`.

`response-contract.example.json` is a synthetic contract example, not measured output.
Confidence is `detections[].instances[].confidence`.
See [shared handoff](../../docs/RECOVERED_MODEL_HANDOFF.md) for validation,
S3 layout, cleanup, permissions and deployment commands.
Service: `http://mmc-vehicle-classification/invocations` in the default namespace.
