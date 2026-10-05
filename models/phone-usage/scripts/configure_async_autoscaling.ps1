param(
    [string]$EndpointName = "phone-usage-detection-endpoint",
    [string]$VariantName = "AllTraffic",
    [string]$Region = "ap-southeast-2",
    [int]$MinCapacity = 0,
    [int]$MaxCapacity = 1,
    [int]$EmptyQueuePeriods = 3,
    [int]$ScaleInCooldownSeconds = 300,
    [int]$ScaleOutCooldownSeconds = 60,
    [string]$AlarmNotificationTopicArn = ""
)

$ErrorActionPreference = "Stop"

function Resolve-AwsCli {
    $Command = Get-Command aws -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }

    $Candidate = Join-Path $env:LOCALAPPDATA "Programs\Amazon\AWSCLIV2\aws.exe"
    if (Test-Path -LiteralPath $Candidate) { return $Candidate }
    throw "AWS CLI was not found"
}

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

$AwsCli = Resolve-AwsCli
$ResourceId = "endpoint/${EndpointName}/variant/${VariantName}"
$ScaleFromZeroPath = [System.IO.Path]::GetTempFileName()
$ScaleToZeroPath = [System.IO.Path]::GetTempFileName()
$QueueMetricsPath = [System.IO.Path]::GetTempFileName()

try {
    & $AwsCli application-autoscaling register-scalable-target `
        --service-namespace sagemaker `
        --resource-id $ResourceId `
        --scalable-dimension sagemaker:variant:DesiredInstanceCount `
        --min-capacity $MinCapacity `
        --max-capacity $MaxCapacity `
        --region $Region | Out-Null
    Assert-NativeSuccess "Scalable target registration"

    $ScaleFromZeroConfig = @{
        AdjustmentType = "ChangeInCapacity"
        MetricAggregationType = "Average"
        Cooldown = $ScaleOutCooldownSeconds
        StepAdjustments = @(
            @{ MetricIntervalLowerBound = 0; ScalingAdjustment = 1 }
        )
    } | ConvertTo-Json -Compress -Depth 10
    Write-Utf8NoBom $ScaleFromZeroPath $ScaleFromZeroConfig

    $ScaleFromZero = & $AwsCli application-autoscaling put-scaling-policy `
        --policy-name "${EndpointName}-scale-from-zero" `
        --service-namespace sagemaker `
        --resource-id $ResourceId `
        --scalable-dimension sagemaker:variant:DesiredInstanceCount `
        --policy-type StepScaling `
        --step-scaling-policy-configuration "file://$($ScaleFromZeroPath.Replace('\', '/'))" `
        --region $Region `
        --output json | ConvertFrom-Json
    Assert-NativeSuccess "Scale-from-zero policy creation"

    $ScaleToZeroConfig = @{
        AdjustmentType = "ChangeInCapacity"
        MetricAggregationType = "Average"
        Cooldown = $ScaleInCooldownSeconds
        StepAdjustments = @(
            @{ MetricIntervalUpperBound = 0; ScalingAdjustment = -1 }
        )
    } | ConvertTo-Json -Compress -Depth 10
    Write-Utf8NoBom $ScaleToZeroPath $ScaleToZeroConfig

    $ScaleToZero = & $AwsCli application-autoscaling put-scaling-policy `
        --policy-name "${EndpointName}-scale-to-zero" `
        --service-namespace sagemaker `
        --resource-id $ResourceId `
        --scalable-dimension sagemaker:variant:DesiredInstanceCount `
        --policy-type StepScaling `
        --step-scaling-policy-configuration "file://$($ScaleToZeroPath.Replace('\', '/'))" `
        --region $Region `
        --output json | ConvertFrom-Json
    Assert-NativeSuccess "Scale-to-zero policy creation"

    & $AwsCli cloudwatch put-metric-alarm `
        --alarm-name "${EndpointName}-has-backlog-without-capacity" `
        --alarm-description "Wake the async endpoint when work is queued at zero capacity." `
        --namespace AWS/SageMaker `
        --metric-name HasBacklogWithoutCapacity `
        --dimensions "Name=EndpointName,Value=${EndpointName}" `
        --statistic Average `
        --period 60 `
        --evaluation-periods 1 `
        --datapoints-to-alarm 1 `
        --threshold 1 `
        --comparison-operator GreaterThanOrEqualToThreshold `
        --treat-missing-data notBreaching `
        --alarm-actions $ScaleFromZero.PolicyARN `
        --region $Region
    Assert-NativeSuccess "Scale-from-zero CloudWatch alarm creation"

    $QueueEmptyAlarmActions = @($ScaleToZero.PolicyARN)
    if ($AlarmNotificationTopicArn) {
        $QueueEmptyAlarmActions += $AlarmNotificationTopicArn
    }
    $QueueMetrics = @(
        @{
            Id = "backlog"
            MetricStat = @{
                Metric = @{
                    Namespace = "AWS/SageMaker"
                    MetricName = "ApproximateBacklogSize"
                    Dimensions = @(
                        @{ Name = "EndpointName"; Value = $EndpointName }
                    )
                }
                Period = 60
                Stat = "Average"
            }
            ReturnData = $false
        },
        @{
            Id = "backlog_without_capacity"
            MetricStat = @{
                Metric = @{
                    Namespace = "AWS/SageMaker"
                    MetricName = "HasBacklogWithoutCapacity"
                    Dimensions = @(
                        @{ Name = "EndpointName"; Value = $EndpointName }
                    )
                }
                Period = 60
                Stat = "Average"
            }
            ReturnData = $false
        },
        @{
            Id = "effective_backlog"
            Expression = "MAX([FILL(backlog,0),FILL(backlog_without_capacity,0)])"
            Label = "Effective async backlog"
            ReturnData = $true
        }
    ) | ConvertTo-Json -Compress -Depth 10
    Write-Utf8NoBom $QueueMetricsPath $QueueMetrics

    & $AwsCli cloudwatch put-metric-alarm `
        --alarm-name "${EndpointName}-queue-empty" `
        --alarm-description "Scale the async endpoint to zero when its queue stays empty." `
        --metrics "file://$($QueueMetricsPath.Replace('\', '/'))" `
        --evaluation-periods $EmptyQueuePeriods `
        --datapoints-to-alarm $EmptyQueuePeriods `
        --threshold 1 `
        --comparison-operator LessThanThreshold `
        --treat-missing-data breaching `
        --alarm-actions $QueueEmptyAlarmActions `
        --region $Region
    Assert-NativeSuccess "Queue-empty CloudWatch alarm creation"

    Write-Output $ResourceId
} finally {
    Remove-Item -LiteralPath $ScaleFromZeroPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $ScaleToZeroPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $QueueMetricsPath -Force -ErrorAction SilentlyContinue
}
