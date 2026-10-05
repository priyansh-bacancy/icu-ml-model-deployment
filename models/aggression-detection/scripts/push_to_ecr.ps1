param(
    [Parameter(Mandatory = $true)]
    [string]$AccountId,
    [string]$Region = "ap-southeast-2",
    [string]$Repository = "icu-solarcam/videomae-aggression",
    [string]$Tag = "v1.0.5-eks-gpu",
    [string]$LocalImage = "icu-solarcam/videomae-aggression:v1.0.5-eks-gpu"
)

$ErrorActionPreference = "Stop"
$Registry = "${AccountId}.dkr.ecr.${Region}.amazonaws.com"
$ImageUri = "${Registry}/${Repository}:${Tag}"

function Assert-NativeSuccess([string]$Action) {
    if ($LASTEXITCODE -ne 0) { throw "$Action failed with exit code $LASTEXITCODE" }
}

aws sts get-caller-identity | Out-Null
Assert-NativeSuccess "AWS identity check"
aws ecr describe-repositories --repository-names $Repository --region $Region | Out-Null
Assert-NativeSuccess "ECR repository lookup"
docker image inspect $LocalImage | Out-Null
Assert-NativeSuccess "Local image lookup"
docker run --rm --platform linux/amd64 --env DEVICE=cpu $LocalImage `
    python /app/scripts/model_smoke_test.py
Assert-NativeSuccess "Container model smoke test"
docker run --rm --platform linux/amd64 --env DEVICE=cpu $LocalImage `
    python /app/scripts/contract_smoke_test.py
Assert-NativeSuccess "Response contract smoke test"

aws ecr get-login-password --region $Region |
    docker login --username AWS --password-stdin $Registry
Assert-NativeSuccess "ECR login"
docker tag $LocalImage $ImageUri
Assert-NativeSuccess "Docker tag"
docker push $ImageUri
Assert-NativeSuccess "Docker push"

Write-Output $ImageUri
