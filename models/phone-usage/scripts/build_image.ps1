param(
    [string]$ImageName = "icu-solarcam/phone-usage-detection",
    [string]$Tag = "v1.0.4-eks-gpu"
)

$ErrorActionPreference = "Stop"
$Image = "${ImageName}:${Tag}"

docker info | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Docker is not ready" }
docker build --platform linux/amd64 --provenance=false `
    --file Dockerfile.gpu --tag $Image .
if ($LASTEXITCODE -ne 0) { throw "Docker build failed" }
docker run --rm --platform linux/amd64 $Image python /app/scripts/smoke_test.py
if ($LASTEXITCODE -ne 0) { throw "Container smoke test failed" }
docker run --rm --platform linux/amd64 $Image `
    python /app/scripts/contract_smoke_test.py
if ($LASTEXITCODE -ne 0) { throw "Response contract smoke test failed" }

Write-Host "Built and verified $Image"
