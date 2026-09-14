param([int]$Port = 3001, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Invoke-Docker {
    & docker @args
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed (exit $LASTEXITCODE)." }
}
function Get-DockerProbe {
    $ErrorActionPreference = 'Continue'
    $output = & docker @args 2>$null
    return @{ Code = $LASTEXITCODE; Output = $output }
}
try {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw 'Please install Docker Desktop (Linux containers), then retry.'
    }
    $probe = Get-DockerProbe info --format '{{.OSType}}'
    if ($probe.Code -ne 0) {
        $desktop = Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'
        if (Test-Path -LiteralPath $desktop) { Start-Process -FilePath $desktop -WindowStyle Hidden }
        Write-Host 'Waiting for Docker Desktop...'
        $ready = $false
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep -Seconds 3
            $probe = Get-DockerProbe info --format '{{.OSType}}'
            if ($probe.Code -eq 0) { $ready = $true; break }
        }
        if (-not $ready) { throw 'Docker Desktop did not become ready. Start Docker Desktop and retry.' }
    }
    if ($probe.Output -ne 'linux') { throw 'Switch Docker Desktop to Linux containers.' }
    Invoke-Docker compose version
    $manifest = Get-Content -LiteralPath 'images.json' -Raw | ConvertFrom-Json
    $load = $false
    foreach ($entry in $manifest.images) {
        $probe = Get-DockerProbe image inspect $entry.tag --format '{{.Id}}'
        if ($probe.Code -ne 0 -or $probe.Output -ne $entry.id) { $load = $true }
    }
    if ($load) {
        Write-Host 'Verifying and loading demo images. Please wait...'
        $archive = Join-Path $PSScriptRoot $manifest.archive
        if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $manifest.sha256) {
            throw 'Image archive checksum mismatch. Copy the complete demo folder again.'
        }
        Invoke-Docker load -i $archive
    }
    New-Item -ItemType Directory -Force -Path 'data','data/jobs','data/engine' | Out-Null
    if (-not (Test-Path -LiteralPath 'data/.runtime-model.json')) {
        [IO.File]::WriteAllText((Join-Path $PSScriptRoot 'data/.runtime-model.json'), '{}')
    }
    if (-not (Test-Path -LiteralPath 'data/.rules-imported.json')) {
        [IO.File]::WriteAllText((Join-Path $PSScriptRoot 'data/.rules-imported.json'), '[]')
    }
    if ($Port -lt 1024 -or $Port -gt 65535) { throw 'Port must be between 1024 and 65535.' }
    $env:DEMO_PORT = "$Port"
    Write-Host 'Starting application and local clustering model...'
    Invoke-Docker compose up -d --wait --wait-timeout 600
    $url = "http://localhost:$Port"
    $response = Invoke-WebRequest -Uri "$url/api/model" -UseBasicParsing -TimeoutSec 15
    if ($response.StatusCode -ne 200) { throw 'Application API health check failed.' }
    Write-Host "Demo is ready: $url" -ForegroundColor Green
    if (-not $NoBrowser) { Start-Process $url }
} catch {
    Write-Host $_.Exception.Message -ForegroundColor Red
    if (Get-Command docker -ErrorAction SilentlyContinue) { & docker compose logs --tail 40 }
    exit 1
}
