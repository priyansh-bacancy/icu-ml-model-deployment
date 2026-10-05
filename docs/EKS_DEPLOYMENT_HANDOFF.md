# ICU Solarcam EKS Sydney deployment handoff

Generated: 2026-08-18 14:45 IST

## Status

The Phone, Sleep, and Aggression images are uploaded, digest-pinned, installed in
EKS, and validated sequentially on
`ip-172-31-25-73.ap-southeast-2.compute.internal` using one NVIDIA GPU at a time.

After each successful inference its Deployment was scaled back to zero and its pod
was deleted before the next model started. Final state:

- Three ServiceAccounts, ClusterIP Services, and Deployments exist.
- All three new Deployments are at `replicas: 0`; no new model pod or GPU claim is
  active.
- Fire-Smoke, People-Count, and Manager remain unchanged and `1/1 Ready`.
- The selected node is `DiskPressure=False`, has no taints, and exposes one
  allocatable NVIDIA GPU.
- The prepared Manager routing image remains undeployed.

## Target

- AWS account: `126378326989`
- Region: `ap-southeast-2`
- EKS cluster: `icu-solarcam-cluster-sydney`
- Namespace: `default`
- S3 bucket: `icu-solarcam-storage-bacancy-ap-southeast-2`
- Shared IRSA role: `arn:aws:iam::126378326989:role/icu-eks-ppe-s3-role-sydney`

## Deployed images

| Model | Release tag | Returned version | Immutable ECR image |
|---|---|---|---|
| Phone usage | `v1.0.4-eks-gpu` | `v1.0.4` | `icu-solarcam/phone-usage-detection@sha256:cd18bdc04c4d64a709463d2d47d22b655467177e208da94097f0898935e8eaa2` |
| Sleep | `v1.0.5-eks-gpu` | `v1.0.5` | `icu-solarcam/sleep-detection@sha256:4a73895a4f0523c4b3fb3bd2c2431b7f587eb4fad54d7f03c4d052a6c06dc0ed` |
| Aggression | `v1.0.5-eks-gpu` | `v1.0.5` | `icu-solarcam/videomae-aggression@sha256:b4c13e9c1ee289cf1e0ae148a16ed1df85940bc096cb8604711f7605cc187478` |

The full registry prefix is
`126378326989.dkr.ecr.ap-southeast-2.amazonaws.com/`. All images are
`linux/amd64`.

Manager routing image prepared but not activated:
`icu-solarcam/manager@sha256:92b3519e8342242c5a79fbfe16b4004b934b00258575cb1f1af49fb3a6adf2fb`.

## Security decision

Local Trivy scans using fixable High/Critical gating passed before upload. Amazon
Inspector Enhanced Scanning subsequently reported these remaining fixable active
High/Critical findings on the runnable child manifests:

| Image | Fixable High/Critical |
|---|---:|
| Phone usage | 2 |
| Sleep | 2 |
| Aggression | 21 |

On 2026-08-18 the user explicitly instructed deployment to proceed without further
vulnerability remediation. These findings remain accepted deployment risk and have
not been suppressed or marked resolved.

## EKS GPU inference evidence

Every model was `1/1 Ready`, pinned to the selected hostname, and had both
`requests` and `limits` of `nvidia.com/gpu: 1` before invocation.

| Model | Input | Job ID | Result | Processing | Detection result |
|---|---|---|---|---:|---|
| Phone | `uploads/photoshoot.mp4` | `eks-phone-sequential-20260818` | `SUCCESS` | 64,995 ms | 27 `in_hand`, 1 `on_ear` |
| Sleep | `uploads/sleeping_video07.mp4` | `eks-sleep-sequential-20260818` | `SUCCESS` | 9,131 ms | 87 `SLEEPING` instances |
| Aggression | `uploads/fight_04.mp4` | `eks-aggression-sequential-20260818` | `SUCCESS` | 42,930 ms | 9 aggression segments |

For every model `/ping` returned HTTP 200 and `/invocations` returned the expected
model ID, event type, model version, output-media schema, and `error: null`.

Verified S3 result prefixes:

- `jobs/eks-phone-sequential-20260818/phone-usage-detection/`
- `jobs/eks-sleep-sequential-20260818/sleep-detection/`
- `jobs/eks-aggression-sequential-20260818/videomae-aggression/`

Phone and Sleep created annotated video/image objects and labels-only training
exports. Phone wrote 27 label files, Sleep wrote 87, and neither created a
`training/.../images/` prefix. Aggression created annotated video/image objects and
`result.json`.

## Kubernetes and IAM state

The shared IRSA trust policy currently contains 13 service-account subjects. The
following three required subjects are present alongside ten subjects belonging to
existing or concurrently staged services:

- `system:serviceaccount:default:phone-usage-detection-sa`
- `system:serviceaccount:default:sleep-detection-sa`
- `system:serviceaccount:default:videomae-aggression-sa`

The guide-shaped resources under `deploy/eks-sydney/base/` passed client-side and
server-side dry runs before apply. They use digest-pinned images, `Recreate`, the
`gpu-nodes` selector, one NVIDIA GPU, `/ping` probes, S3 storage, Sydney region, and
`DEVICE=cuda:0`.

The live Deployments also retain the temporary sequential-test hostname selector
for `ip-172-31-25-73.ap-southeast-2.compute.internal` and a disk-pressure scheduling
toleration. All three are currently scaled to zero.

Current workload snapshot:

- `fire-smoke-detection`: desired 1, ready 1.
- `people-count`: desired 1, ready 1.
- `manager`: desired 1, ready 1, image remains `icu-solarcam/manager:latest`.
- Concurrently staged `fall-detection`, `mmc-vehicle-classification`, and
  `restricted-zone`: desired 0, with no pods.
- `gpu-nodes`: min 1, max 3, desired 3.

## Re-running sequentially

Scale only one new Deployment to one at a time. Wait for Ready, invoke it, then
scale it back to zero and wait for its pod to be deleted before starting the next.
This preserves the validated one-node/one-GPU workflow and avoids disk pressure
from simultaneous image pulls.

Do not deploy the prepared Manager routing image while all three model Deployments
are at zero replicas. Manager activation requires a separate decision to keep the
model backends running.

## Source-control state

Repository HEAD during deployment:
`4df2727c358afd347ce8be7678e637523d0f5e77` on branch `main`.

No commit was created because the worktree contains unrelated user changes.
