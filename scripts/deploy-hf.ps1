# Push a single-commit snapshot of the working tree to a Hugging Face Space.
#
# Why not just `git push hf main`? This repo's history still contains the old
# committed node_modules (~553 MB of pack). The Space only needs the current
# files (~20 MB), so we push an orphan commit instead — much faster, and the
# Space stays small. History stays intact on GitHub.
#
#   .\scripts\deploy-hf.ps1 -Space "your-username/azsl-sign-demo"
#
# First run will ask for credentials: username = your HF username,
# password = a Hugging Face access token with WRITE scope
# (https://huggingface.co/settings/tokens).

param(
    [Parameter(Mandatory = $true)]
    [string]$Space,                 # e.g. "faig-tybv/azsl-sign-demo"
    [string]$Branch = "hf-deploy"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$remoteUrl = "https://huggingface.co/spaces/$Space"

# Refuse to run with uncommitted work — the snapshot would be ambiguous.
if (git status --porcelain) {
    Write-Host "Working tree is dirty. Commit or stash first:" -ForegroundColor Red
    git status --short
    exit 1
}

$current = git rev-parse --abbrev-ref HEAD
Write-Host "Snapshotting '$current' -> $remoteUrl (branch: main)" -ForegroundColor Cyan

if (-not (git remote | Select-String -Quiet '^hf$')) {
    git remote add hf $remoteUrl
} else {
    git remote set-url hf $remoteUrl
}

try {
    git branch -D $Branch 2>$null | Out-Null
    git checkout --orphan $Branch
    git add -A
    git commit -q -m "Deploy snapshot from $current ($(Get-Date -Format 'yyyy-MM-dd HH:mm'))"
    git push -f hf "${Branch}:main"
    Write-Host ""
    Write-Host "Pushed. Watch the build at $remoteUrl (Logs tab)." -ForegroundColor Green
    Write-Host "First build takes ~10 minutes (torch + MediaPipe)." -ForegroundColor Green
}
finally {
    git checkout -q $current
    git branch -D $Branch 2>$null | Out-Null
}
