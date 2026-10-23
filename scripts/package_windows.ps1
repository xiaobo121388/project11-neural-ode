param([string]$OutputDir = (Join-Path $PSScriptRoot '..\dist\WindowsDemo'))
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
Push-Location $Root
try {
    cargo build --release --locked
    if ($LASTEXITCODE -ne 0) { throw 'Release build failed' }
    $Target = if ($env:CARGO_TARGET_DIR) { $env:CARGO_TARGET_DIR } else { Join-Path $Root 'target' }
    New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
    foreach ($Name in @('pendulum-lab.exe','project11.exe')) {
        Copy-Item -LiteralPath (Join-Path $Target ('release\' + $Name)) -Destination $OutputDir
    }
    Copy-Item -LiteralPath (Join-Path $Root 'LICENSE') -Destination $OutputDir
    Copy-Item -LiteralPath (Join-Path $Root 'assets\LICENSE.lucide') -Destination $OutputDir
    if (Test-Path -LiteralPath (Join-Path $Root 'docs\WINDOWS_DEMO.txt')) {
        Copy-Item -LiteralPath (Join-Path $Root 'docs\WINDOWS_DEMO.txt') -Destination $OutputDir
    }
    Write-Output ('Offline Windows demo prepared at ' + $OutputDir)
} finally { Pop-Location }
