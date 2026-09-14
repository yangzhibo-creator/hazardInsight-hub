param([string]$OutputDirectory = 'D:\datas\images\智能识别和定级\演示-v5', [switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$source = $PSScriptRoot
$root = (Resolve-Path (Join-Path $source '../..')).Path
function Run-Docker {
    & docker @args
    if ($LASTEXITCODE -ne 0) { throw "Docker command failed (exit $LASTEXITCODE)." }
}
if (-not $SkipBuild) {
    Run-Docker build --platform linux/amd64 -f (Join-Path $source 'web.Dockerfile') -t nuclear-hazard-demo:demo-v5 $root
    Run-Docker build --platform linux/amd64 -f (Join-Path $source 'python.Dockerfile') -t nuclear-hazard-clustering:demo-v5 $root
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$output = (Resolve-Path -LiteralPath $OutputDirectory).Path
foreach ($name in @('compose.yaml','start.ps1','start.bat','stop.bat','logs.bat','start.sh','stop.sh','启动说明.md')) {
    Copy-Item -LiteralPath (Join-Path $source $name) -Destination $output
}
Copy-Item -LiteralPath (Join-Path $source 'start.bat') -Destination (Join-Path $output '启动演示.bat')
Copy-Item -LiteralPath (Join-Path $source 'stop.bat') -Destination (Join-Path $output '停止演示.bat')
$archive = 'hazard-demo-v5.tar'
Run-Docker save -o (Join-Path $output $archive) nuclear-hazard-demo:demo-v5 nuclear-hazard-clustering:demo-v5
$digest = (Get-FileHash -LiteralPath (Join-Path $output $archive) -Algorithm SHA256).Hash.ToLowerInvariant()
$images = foreach ($tag in @('nuclear-hazard-demo:demo-v5','nuclear-hazard-clustering:demo-v5')) {
    $id = & docker image inspect $tag --format '{{.Id}}'
    if ($LASTEXITCODE -ne 0) { throw "Cannot inspect $tag" }
    @{ tag = $tag; id = $id }
}
$manifest = @{ archive = $archive; sha256 = $digest; images = @($images); platform = 'linux/amd64'; builtAt = (Get-Date -Format o) }
[IO.File]::WriteAllText((Join-Path $output 'images.json'), ($manifest | ConvertTo-Json -Depth 5))
[IO.File]::WriteAllText((Join-Path $output 'images.sha256'), "$digest  $archive`n")
$packages = & docker run --rm --entrypoint pip nuclear-hazard-clustering:demo-v5 freeze
if ($LASTEXITCODE -ne 0) { throw 'Cannot export Python dependency versions.' }
[IO.File]::WriteAllLines((Join-Path $output 'python-dependencies.txt'), [string[]]$packages)
Write-Host "Demo package ready: $output"
