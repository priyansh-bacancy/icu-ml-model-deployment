# ICU ML Model Deployment

Deployment-ready source code for the ICU video analytics models maintained by
Priyansh. The repository keeps the runtime, tests, container definitions and EKS
manifests required to modify or redeploy the services without depending on one
developer machine.

## Models

| Model | Source | Kubernetes workload |
|---|---|---|
| Phone usage | `models/phone-usage/` | `phone-usage-detection` |
| Sleep detection | `models/sleep-detection/` | `sleep-detection` |
| Aggression detection | `models/aggression-detection/` | `videomae-aggression` |

Shared EKS resources and the manager routing patch are under
`deploy/eks-sydney/`. The validated deployment state is documented in
`docs/EKS_DEPLOYMENT_HANDOFF.md`.

## Repository policy

This repository stores code and deployment configuration only. It deliberately
excludes:

- trained model weights and packaged model archives;
- datasets, test videos and generated images;
- AWS credentials, tokens and local environment files;
- inference responses, local S3 mirrors and Python caches.

The immutable ECR image digests in the EKS manifests are the deployment recovery
source. When rebuilding an image, restore the required checkpoint from the
approved S3 model-artifact location and place it in the model directory using the
filename documented by that model's Dockerfile. See `artifacts.example.yaml` for
the expected inventory format.

## Local validation

Run the unit tests from the repository root:

```powershell
python -m pytest models/phone-usage/tests
python -m pytest models/sleep-detection/tests
python -m pytest models/aggression-detection/tests
```

Build a model image from its own directory after restoring its checkpoint:

```powershell
Set-Location models/phone-usage
.\scripts\build_image.ps1
```

Equivalent build scripts are available under the other model directories.

Validate the EKS manifests before applying them:

```powershell
kubectl kustomize deploy/eks-sydney/base
kubectl apply --dry-run=client -k deploy/eks-sydney/base
```

Only one of these GPU model Deployments should be scaled to one replica at a
time unless the cluster has been intentionally resized and validated for
concurrent GPU scheduling.

## Adding another model

Add each independently owned model through a separate pull request. Include its
runtime code, container files, tests, response contract and deployment manifest;
keep all generated artifacts and credentials outside Git.

