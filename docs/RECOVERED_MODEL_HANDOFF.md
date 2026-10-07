# Recovered model deployment packages

Fall v7, Restricted Zone v7 and MMC v8 were recovered from retained
application files and pinned ECR image digests. Restricted Zone includes the
subsequent tested caller-zone-ID pass-through fix. These additions do not modify
Priyansh's three existing models, their manifests, or Manager.

## Provenance and limitations

Original local Git history, current Dockerfiles, and most tests were lost.
The new full-build Dockerfiles are reconstructed from retained image history,
not falsely represented as recovered originals. Runtime files were copied, not
redesigned. Existing large legacy modules remain intact for review.
The added tests are recovery regression tests, not the lost original suite.
Historical August SageMaker/CodeBuild sources are deliberately excluded: they
predate current H.264, frame sampling and multi-snapshot behavior.
No cloud resources are changed by this backup PR.

Exact recovery image digests (repository prefix supplied by your administrator):

| Repository | Version | Digest |
|---|---|---|
| fall-detection | v7 | sha256:2c2f1098ced527530b97b675682574e5545a3e3db73a858500aa5b95a1f8420c |
| restricted-zone | v7 | sha256:9b57cb16b7dbacee9af9e74972cdec5b512adf044fd7521b673325fe63fb5800 |
| mmc-vehicle-classification | v8 | sha256:14b2da6d10f4d53d25f119e7ea81012243f5b00b492803ec9b3d5c8c50554ad7 |

## Uploading rebuilt images and packaging checkpoints

After authorized local build/testing, set your own non-secret profile/region,
registry, repository and immutable tag. Repeat for each model:

```powershell
$awsProfile = '<authorized-profile>'
$awsRegion = 'ap-southeast-2'
$registry = '<account-id>.dkr.ecr.ap-southeast-2.amazonaws.com'
$modelId = 'fall-detection' # or restricted-zone / mmc-vehicle-classification
$imageTag = '<new-immutable-tag>'
aws ecr get-login-password --profile $awsProfile --region $awsRegion | docker login --username AWS --password-stdin $registry
if ($LASTEXITCODE -ne 0) { throw 'Registry authentication failed' }
docker tag "icu-solarcam/${modelId}:$imageTag" "$registry/icu-solarcam/${modelId}:$imageTag"
if ($LASTEXITCODE -ne 0) { throw 'Image tagging failed' }
docker push "$registry/icu-solarcam/${modelId}:$imageTag"
if ($LASTEXITCODE -ne 0) { throw 'Image push failed' }
aws ecr describe-images --repository-name "icu-solarcam/$modelId" --image-ids "imageTag=$imageTag" --profile $awsProfile --region $awsRegion --query 'imageDetails[0].[imageDigest,imageManifestMediaType]' --output table
```

Registry repositories must already exist or be created by the administrator.
Use the verified pushed digest in the private Kubernetes overlay, not a mutable
latest tag. Do not overwrite another team's tags. Archive checkpoints separately:

```powershell
tar -czf <model-id>-checkpoints.tar.gz -C models/<model-id> weights
```

This checkpoint archive is not the historical SageMaker code-plus-weights
archive. Archive packaging and uploads are not prerequisites for EKS inference.

## Current deployment approach

EKS HTTP services are current. The Python runtime keeps SageMaker-compatible
`/ping`, `/invocations` and `/predict` endpoints; it does not require a
SageMaker model resource or an S3 model archive to serve embedded weights.

New manifests are isolated under `deploy/eks-recovered/` and start at zero
replicas. Existing `deploy/eks-sydney/` is not changed. Configure images,
bucket and role in a private overlay; never commit production configuration.
The namespace is `default`. An administrator must authorize each exact
ServiceAccount in the cluster OIDC IRSA trust policy. Pod role needs S3 GetObject
on source clips and zone JSON, PutObject on jobs/prelabels; MMC additionally
needs GetObject on its FAISS prefix. SSE-KMS buckets also need appropriate KMS
permissions. Node role needs ECR pull access and GPU device plugin/driver.

Commands for a later authorized deployment (not run during this backup):

```powershell
kubectl kustomize deploy/eks-recovered
# Configure placeholders in a private overlay before applying.
kubectl apply -k <private-overlay-directory>
kubectl scale deployment/<model-id> --replicas=1 -n default
kubectl rollout status deployment/<model-id> -n default --timeout=15m
kubectl port-forward service/<model-id> 8083:80 -n default
# In a second terminal:
Invoke-RestMethod http://127.0.0.1:8083/ping
Invoke-RestMethod http://127.0.0.1:8083/invocations -Method Post -ContentType application/json -Body (Get-Content models/<model-id>/input.example.json -Raw)
# After the test, stop port-forwarding and scale the same Deployment to zero.
```

Short service names work only inside the same Kubernetes namespace:
`http://fall-detection/invocations`,
`http://restricted-zone/invocations`,
`http://mmc-vehicle-classification/invocations`.

## Shared response and storage behavior

Exactly nine top-level keys: model_id, event_type, event_id, model_version,
status, processing_time_ms, output_media, detections, error.
Confidence is numeric under `detections[].instances[].confidence`, not on
the detection itself. Examples are synthetic, not customer inference records.
Output media uses `annotated_video_s3_key` and plural
`annotated_image_s3_keys` (up to five snapshots), not the earlier singular key.
No detections can legitimately yield an empty snapshot list.

Latest recovered runtime writes:
```text
uploads/<clip>                             # Existing input, not created by inference
uploads/<zones>.json                       # Restricted Zone config, supplied upstream
jobs/<job_id>/<model_id>/input.json         # Saved when inference starts
jobs/<job_id>/<model_id>/result.json        # Saved success/failure response
jobs/<job_id>/<model_id>/annotated.mp4      # Successful browser-compatible H.264 output
jobs/<job_id>/<model_id>/frame_1.jpg        # Up to frame_5.jpg, selected annotated snapshots
jobs/<job_id>/mmc-vehicle-classification/crops/  # MMC classified track crops
prelabels/<job_id>/<model_id>/labels/frame_000000.txt
prelabels/<job_id>/<model_id>/labels/frame_000015.txt
prelabels/<job_id>/<model_id>/sampling_manifest.json
models/mmc-vehicle-classification/faiss_db/v1/faiss/
```
Important: despite older discussion saying training/labels, the recovered
October runtime actually writes `prelabels/`. This backup preserves actual
behavior. It samples frames 0,15,30,... including empty label files on empty
frames, while inference evaluates all frames. No prelabel images are uploaded.
These are machine predictions awaiting review, not approved training data.
Job temporary files use TemporaryDirectory cleanup after success/failure;
MMC's FAISS cache is intentionally retained for the process lifetime.

## Verification and external artifacts

```powershell
uv sync --locked
uv run pytest
uv run ruff check models/fall-detection models/restricted-zone models/mmc-vehicle-classification tests
uv run ruff format --check models/fall-detection models/restricted-zone models/mmc-vehicle-classification tests
kubectl kustomize deploy/eks-recovered
```

Unit verification does not load checkpoints or assert GPU availability.
Full GPU image rebuilding, live inference, vulnerability scanning, and legacy
model suites require separate appropriate dependencies/artifacts and approval.
Do not interpret offline rendering as server-side Kubernetes validation.
Restore checkpoint files into each package's `weights/` directory.
Restore FAISS indexes and paired metadata together, preserving relative paths.
The incomplete historical MMC archive is not a valid restore source; current
MMC checkpoints and the complete FAISS set were independently recovered.
