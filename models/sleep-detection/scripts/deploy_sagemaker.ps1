param(
    [Parameter(Mandatory = $true)]
    [string]$AccountId,
    [Parameter(Mandatory = $true)]
    [string]$ExecutionRoleArn,
    [Parameter(Mandatory = $true)]
    [string]$S3Bucket,
    [string]$Region = "ap-southeast-2",
    [string]$Repository = "icu-solarcam/sleep-detection",
    [string]$ImageTag = "v1.0.3-gpu",
    [string]$EndpointName = "sleep-detection-endpoint",
    [string]$InstanceType = "ml.g4dn.xlarge",
    [string]$AsyncOutputPrefix = "jobs"
)

$ErrorActionPreference = "Stop"
$Suffix = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")
$ModelName = "sleep-detection-${Suffix}"
$ConfigName = "${EndpointName}-${Suffix}"
$ImageUri = "${AccountId}.dkr.ecr.${Region}.amazonaws.com/${Repository}:${ImageTag}"
$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) "sleep-detection-${Suffix}"
New-Item -ItemType Directory -Path $TempRoot | Out-Null

function Assert-NativeSuccess([string]$Action) {
    if ($LASTEXITCODE -ne 0) { throw "$Action failed with exit code $LASTEXITCODE" }
}

function Write-Utf8NoBom([string]$Path, [string]$Value) {
    [System.IO.File]::WriteAllText(
        $Path,
        $Value,
        [System.Text.UTF8Encoding]::new($false)
    )
}

try {
    $Container = @{
        Image = $ImageUri
        Environment = @{
            S3_BUCKET = $S3Bucket
            AWS_REGION = $Region
            MODEL_ID = "sleep-detection"
            MODEL_VERSION = "v1.0.3"
            STORAGE_BACKEND = "s3"
            DEVICE = "cuda:0"
            IMAGE_SIZE = "640"
            CONFIDENCE_THRESHOLD = "0.70"
            TARGET_LABEL = "SLEEPING"
        }
    }
    $ContainerPath = Join-Path $TempRoot "container.json"
    Write-Utf8NoBom $ContainerPath ($Container | ConvertTo-Json -Depth 5)

    aws sagemaker create-model `
        --model-name $ModelName `
        --execution-role-arn $ExecutionRoleArn `
        --primary-container "file://$($ContainerPath.Replace('\', '/'))" `
        --region $Region | Out-Null
    Assert-NativeSuccess "SageMaker model creation"

    $Variant = @{
        VariantName = "AllTraffic"
        ModelName = $ModelName
        InitialVariantWeight = 1.0
        InitialInstanceCount = 1
        InstanceType = $InstanceType
    }
    $VariantPath = Join-Path $TempRoot "variant.json"
    Write-Utf8NoBom $VariantPath (ConvertTo-Json -InputObject @($Variant) -Depth 5)

    $NormalizedAsyncPrefix = $AsyncOutputPrefix.Trim('/')
    $AsyncConfig = @{
        OutputConfig = @{
            S3OutputPath = "s3://${S3Bucket}/${NormalizedAsyncPrefix}/"
            S3FailurePath = "s3://${S3Bucket}/${NormalizedAsyncPrefix}/"
        }
        ClientConfig = @{
            MaxConcurrentInvocationsPerInstance = 1
        }
    }
    $AsyncConfigPath = Join-Path $TempRoot "async-config.json"
    Write-Utf8NoBom $AsyncConfigPath ($AsyncConfig | ConvertTo-Json -Depth 5)

    aws sagemaker create-endpoint-config `
        --endpoint-config-name $ConfigName `
        --production-variants "file://$($VariantPath.Replace('\', '/'))" `
        --async-inference-config "file://$($AsyncConfigPath.Replace('\', '/'))" `
        --region $Region | Out-Null
    Assert-NativeSuccess "SageMaker endpoint configuration creation"

    $PreviousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    aws sagemaker describe-endpoint `
        --endpoint-name $EndpointName `
        --region $Region 2>$null | Out-Null
    $EndpointLookupExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousErrorActionPreference
    if ($EndpointLookupExitCode -eq 0) {
        aws sagemaker update-endpoint `
            --endpoint-name $EndpointName `
            --endpoint-config-name $ConfigName `
            --retain-all-variant-properties `
            --region $Region | Out-Null
        Assert-NativeSuccess "SageMaker endpoint update"
        $Operation = "update"
    } else {
        aws sagemaker create-endpoint `
            --endpoint-name $EndpointName `
            --endpoint-config-name $ConfigName `
            --region $Region | Out-Null
        Assert-NativeSuccess "SageMaker endpoint creation"
        $Operation = "create"
    }

    [pscustomobject]@{
        Operation = $Operation
        EndpointName = $EndpointName
        EndpointConfigName = $ConfigName
        ModelName = $ModelName
        ImageUri = $ImageUri
    } | ConvertTo-Json -Compress
} finally {
    if (Test-Path -LiteralPath $TempRoot) {
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
    }
}
