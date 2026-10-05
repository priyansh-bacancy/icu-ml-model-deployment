param(
    [string]$OutputPath = "dist/model.tar.gz"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ResolvedOutput = if ([IO.Path]::IsPathRooted($OutputPath)) {
    [IO.Path]::GetFullPath($OutputPath)
} else {
    [IO.Path]::GetFullPath((Join-Path $ProjectRoot $OutputPath))
}
$OutputDirectory = Split-Path -Parent $ResolvedOutput
$TempRoot = Join-Path ([IO.Path]::GetTempPath()) (
    "sleep-detection-model-" + [guid]::NewGuid().ToString("N")
)

New-Item -ItemType Directory -Path $TempRoot | Out-Null
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
try {
    Copy-Item -LiteralPath (Join-Path $ProjectRoot "best-sleep.pt") `
        -Destination (Join-Path $TempRoot "best.pt")
    foreach ($Name in @("inference.py", "extractor.py", "builder.py", "requirements.txt")) {
        Copy-Item -LiteralPath (Join-Path $ProjectRoot $Name) `
            -Destination (Join-Path $TempRoot $Name)
    }
    tar -czf $ResolvedOutput -C $TempRoot `
        best.pt inference.py extractor.py builder.py requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "model archive creation failed" }
    Write-Output $ResolvedOutput
} finally {
    if (Test-Path -LiteralPath $TempRoot) {
        Remove-Item -LiteralPath $TempRoot -Recurse -Force
    }
}
