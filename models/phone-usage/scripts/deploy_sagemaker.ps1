param(
    [Parameter(Mandatory = $true)]
    [string]$AccountId,
    [Parameter(Mandatory = $true)]
    [string]$ExecutionRoleArn,
    [Parameter(Mandatory = $true)]
    [string]$S3Bucket,
    [string]$Region = "ap-southeast-2",
    [string]$Repository = "icu-solarcam/phone-usage-detection",
    [string]$ImageTag = "v1.0.2-gpu",
    [string]$ModelName = "phone-usage-detection",
    [string]$EndpointName = "phone-usage-detection-endpoint",
    [ValidateSet("Async", "Serverless", "Instance")]
    [string]$Mode = "Async",
    [string]$InstanceType = "ml.g4dn.xlarge",
    [string]$AsyncOutputPrefix = "jobs"
)

$ErrorActionPreference = "Stop"
$Suffix = (Get-Date).ToUniversalTime().ToString("yyyyMMdd-HHmmss")
$ConfigName = $EndpointName
$ImageUri = "${AccountId}.dkr.ecr.${Region}.amazonaws.com/${Repository}:${ImageTag}"
$TempRoot = Join-Path ([System.IO.Path]::GetTempPath()) "phone-usage-detection-${Suffix}"
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
            MODEL_ID = "phone-usage-detection"
            MODEL_VERSION = "v1.0.2"
            STORAGE_BACKEND = "s3"
            DEVICE = "cuda:0"
            IMAGE_SIZE = "1280"
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
    }
    if ($Mode -eq "Serverless") {
        $Variant.ServerlessConfig = @{
            MemorySizeInMB = 6144
            MaxConcurrency = 5
        }
    } else {
        $Variant.InitialInstanceCount = 1
        $Variant.InstanceType = $InstanceType
    }

    $VariantPath = Join-Path $TempRoot "variant.json"
    Write-Utf8NoBom $VariantPath (ConvertTo-Json -InputObject @($Variant) -Depth 5)
    if ($Mode -eq "Async") {
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
    } else {
        aws sagemaker create-endpoint-config `
            --endpoint-config-name $ConfigName `
            --production-variants "file://$($VariantPath.Replace('\', '/'))" `
            --region $Region | Out-Null
    }
    Assert-NativeSuccess "SageMaker endpoint configuration creation"

    aws sagemaker create-endpoint `
        --endpoint-name $EndpointName `
        --endpoint-config-name $ConfigName `
        --region $Region | Out-Null
    Assert-NativeSuccess "SageMaker endpoint creation"

    Write-Host "Endpoint deployment started. Wait with:"
    Write-Host "aws sagemaker wait endpoint-in-service --endpoint-name $EndpointName --region $Region"
    Write-Output $EndpointName
} finally {
    if (Test-Path -LiteralPath $TempRoot) {
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
    }
}
