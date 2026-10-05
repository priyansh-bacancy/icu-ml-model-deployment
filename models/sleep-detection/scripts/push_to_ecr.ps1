param(
    [Parameter(Mandatory = $true)]
    [string]$AccountId,
    [string]$Region = "ap-southeast-2",
    [string]$Repository = "icu-solarcam/sleep-detection",
    [string]$Tag = "v1.0.5-eks-gpu",
    [string]$LocalImage = "icu-solarcam/sleep-detection:v1.0.5-eks-gpu"
)

$ErrorActionPreference = "Stop"
$Registry = "${AccountId}.dkr.ecr.${Region}.amazonaws.com"
$ImageUri = "${Registry}/${Repository}:${Tag}"

function Assert-NativeSuccess([string]$Action) {
    if ($LASTEXITCODE -ne 0) { throw "$Action failed with exit code $LASTEXITCODE" }
}

aws sts get-caller-identity | Out-Null
Assert-NativeSuccess "AWS identity check"

$PreviousErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
aws ecr describe-repositories --repository-names $Repository --region $Region 2>$null | Out-Null
$RepositoryLookupExitCode = $LASTEXITCODE
$ErrorActionPreference = $PreviousErrorActionPreference
if ($RepositoryLookupExitCode -ne 0) {
    aws ecr create-repository `
        --repository-name $Repository `
        --image-scanning-configuration scanOnPush=true `
        --region $Region | Out-Null
    Assert-NativeSuccess "ECR repository creation"
}

docker image inspect $LocalImage | Out-Null
Assert-NativeSuccess "Local image lookup"
docker run --rm --platform linux/amd64 $LocalImage python /app/scripts/smoke_test.py
Assert-NativeSuccess "Container smoke test"
docker run --rm --platform linux/amd64 $LocalImage `
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
