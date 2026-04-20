param(
    [string]$ServiceName = $(if ($env:SERVICE_NAME) { $env:SERVICE_NAME } else { "power-rangers" }),
    [string]$Region = $(if ($env:REGION) { $env:REGION } else { "asia-south1" })
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $env:SCRAPER_PROXY_URL) {
    $dotenvPath = Join-Path $PSScriptRoot ".env"
    if (Test-Path $dotenvPath) {
        foreach ($line in Get-Content $dotenvPath) {
            $trimmed = $line.Trim()
            if (-not $trimmed -or $trimmed.StartsWith("#") -or -not $trimmed.Contains("=")) {
                continue
            }

            $key, $value = $trimmed.Split("=", 2)
            if ($key.Trim() -ne "SCRAPER_PROXY_URL") {
                continue
            }

            $value = $value.Trim()
            if ($value.Length -ge 2 -and $value.StartsWith('"') -and $value.EndsWith('"')) {
                $value = $value.Substring(1, $value.Length - 2)
            }

            $env:SCRAPER_PROXY_URL = $value
            break
        }
    }
}

if (-not $env:JWT_SECRET_KEY) {
    $bytes = New-Object byte[] 48
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $rng.GetBytes($bytes)
    }
    finally {
        $rng.Dispose()
    }

    $env:JWT_SECRET_KEY = [Convert]::ToBase64String($bytes)
}

$envVars = @(
    "JWT_SECRET_KEY=$env:JWT_SECRET_KEY"
    "AUTH_DB_PATH=/tmp/auth.db"
    "ENABLE_DUMMY_FALLBACK=false"
    "OPERATIONAL_DIR=/tmp/power-rangers/operational"
    "SLDC_LOAD_CACHE_DIR=/tmp/power-rangers/operational/raw"
    "SCRAPER_PROXY_URL=$env:SCRAPER_PROXY_URL"
    "OPENMETEO_CACHE_NAME=/tmp/power-rangers/openmeteo"
) -join ","

gcloud run deploy $ServiceName `
    --source . `
    --region $Region `
    --allow-unauthenticated `
    --memory 4Gi `
    --cpu 4 `
    --concurrency 1 `
    --max-instances 1 `
    --min-instances 1 `
    --timeout 900 `
    --set-env-vars $envVars