# Stage the seven release artifacts of build-aji-release.bat and zip them flat
# into <OutDir>\<tag>\aji-windows-x64.zip (see docs\BUILD-WINDOWS.md).
param(
    [Parameter(Mandatory)][string]$Tag,
    [string]$BuildDir = (Join-Path $PSScriptRoot '..\..\build-win-release'),
    [string]$OutDir = (Join-Path ($env:AJI_WIN ?? (Join-Path $env:USERPROFILE 'aji-win')) 'dist')
)
$ErrorActionPreference = 'Stop'
$files = 'aji.dll', 'aji_dml.dll', 'aji_trt.dll', 'aji_encode.exe',
         'aji_harness.exe', 'aji_harness_dml.exe', 'aji_kernel_test.exe'
$stage = Join-Path $OutDir "$Tag\stage"
Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $stage | Out-Null
foreach ($f in $files) {
    $p = Join-Path $BuildDir $f
    if (-not (Test-Path $p)) { throw "missing build artifact: $p" }
    Copy-Item $p $stage
}
$zip = Join-Path $OutDir "$Tag\aji-windows-x64.zip"
Remove-Item -Force $zip -ErrorAction SilentlyContinue
Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip
Remove-Item -Recurse -Force $stage
Add-Type -AssemblyName System.IO.Compression.FileSystem
$z = [System.IO.Compression.ZipFile]::OpenRead($zip)
$z.Entries | ForEach-Object { '{0,12:N0}  {1}' -f $_.Length, $_.FullName }
$z.Dispose()
'{0}  ({1:N1} MB)' -f $zip, ((Get-Item $zip).Length / 1MB)
