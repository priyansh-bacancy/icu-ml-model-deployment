param(
    [Parameter(Mandatory = $true)]
    [string]$AccountId,
    [string]$Region = "ap-southeast-2",
    [string]$Repository = "icu-solarcam/phone-usage-detection",
    [string]$Tag = "v1.0.4-eks-gpu"
)

$ErrorActionPreference = "Stop"
$Registry = "${AccountId}.dkr.ecr.${Region}.amazonaws.com"
$ImageUri = "${Registry}/${Repository}:${Tag}"

function Assert-NativeSuccess([string]$Action) {
    if ($LASTEXITCODE -ne 0) { throw "$Action failed with exit code $LASTEXITCODE" }
}

aws sts get-caller-identity | Out-Null
Assert-NativeSuccess "AWS identity check"
aws ecr describe-repositories --repository-names $Repository --region $Region 2>$null
if ($LASTEXITCODE -ne 0) {
    aws ecr create-repository `
        --repository-name $Repository `
        --image-scanning-configuration scanOnPush=true `
        --region $Region | Out-Null
    Assert-NativeSuccess "ECR repository creation"
}

aws ecr get-login-password --region $Region |
    docker login --username AWS --password-stdin $Registry
Assert-NativeSuccess "ECR login"

docker build --platform linux/amd64 --provenance=false `
    --file Dockerfile.gpu --tag $ImageUri .
Assert-NativeSuccess "Docker build"
docker run --rm --platform linux/amd64 $ImageUri python /app/scripts/smoke_test.py
Assert-NativeSuccess "Container smoke test"
docker run --rm --platform linux/amd64 $ImageUri `
    python /app/scripts/contract_smoke_test.py
Assert-NativeSuccess "Response contract smoke test"
docker push $ImageUri
Assert-NativeSuccess "Docker push"

Write-Output $ImageUri
