param(
    [string]$ServiceName = $(if ($env:SERVICE_NAME) { $env:SERVICE_NAME } else { "power-rangers" }),
    [string]$Region = $(if ($env:REGION) { $env:REGION } else { "asia-south1" })
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

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