# Deploy the recognition backend to Google Cloud Run.
#
#   .\scripts\deploy-cloudrun.ps1 -DatabaseUrl "postgresql+psycopg://..." -SessionSecret "<64 hex>"
#
# Builds in the cloud (Cloud Build), so Docker is NOT needed locally.
# Re-run it any time to redeploy; pass -SkipEnv to keep the existing env vars.

param(
    [string]$Service      = "azsl-recognition",
    [string]$Region       = "europe-west1",
    [string]$DatabaseUrl  = "",
    [string]$SessionSecret = "",
    [switch]$SkipEnv
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Get-Command gcloud -ErrorAction SilentlyContinue)) {
    Write-Host "gcloud CLI not found. Install it first:" -ForegroundColor Red
    Write-Host "  https://cloud.google.com/sdk/docs/install" -ForegroundColor Yellow
    exit 1
}

if (-not $SkipEnv -and (-not $DatabaseUrl -or -not $SessionSecret)) {
    Write-Host "Need -DatabaseUrl and -SessionSecret on the first deploy" -ForegroundColor Red
    Write-Host "  (or pass -SkipEnv to reuse what's already set on the service)" -ForegroundColor Yellow
    exit 1
}

$gcloudArgs = @(
    "run", "deploy", $Service,
    "--source", ".",
    "--region", $Region,
    "--allow-unauthenticated",
    "--memory", "1Gi",
    "--cpu", "1",
    # WebSockets are long-lived requests; the 5-minute default would cut every
    # session off. 3600s is the Cloud Run maximum.
    "--timeout", "3600",
    # ~50 MB per concurrent viewer on top of ~300 MB idle — keep well inside 1Gi.
    "--concurrency", "8",
    "--min-instances", "0",       # scale to zero => stays in the free tier
    "--max-instances", "3",
    "--cpu-boost"                 # faster cold start (model load is ~15 s)
)

if (-not $SkipEnv) {
    # ^##^ picks '##' as the delimiter so a URL containing commas is safe.
    $envs = "^##^DATABASE_URL=$DatabaseUrl##SESSION_SECRET=$SessionSecret##SESSION_COOKIE_SECURE=1"
    $gcloudArgs += @("--set-env-vars", $envs)
}

Write-Host "Deploying '$Service' to $Region (first build ~10 min: torch + MediaPipe)..." -ForegroundColor Cyan
& gcloud @gcloudArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$url = (& gcloud run services describe $Service --region $Region --format "value(status.url)")
Write-Host ""
Write-Host "Deployed: $url" -ForegroundColor Green
Write-Host ""
Write-Host "Use it directly (everything on one origin, no Vercel needed):" -ForegroundColor Green
Write-Host "  $url"
Write-Host ""
Write-Host "Or keep Vercel as the front door - set this on Vercel and redeploy:" -ForegroundColor Green
Write-Host ("  RECOGNITION_WS_URL = " + ($url -replace '^https://', 'wss://'))
