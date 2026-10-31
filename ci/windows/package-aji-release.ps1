# Stage the release artifacts of build-aji-release.bat and zip them flat into
# two assets under <OutDir>\<tag>\ (see docs\BUILD-WINDOWS.md):
#   aji-windows-x64.zip         the engine: dispatcher, aji_trt, aji_dml, tools
#   aji-trtrtx-windows-x64.zip  the opt-in TensorRT for RTX backend: aji_trtrtx.dll
#                               and the two runtime DLLs it needs (never
#                               tensorrt_rtx.exe, which the SDK licence keeps
#                               internal). Extracted flat on top of the first.
# A consumer that never offers backend=TensorRT-RTX skips the second (~91 MB);
# with it missing, the dispatcher falls back to TensorRT.
param(
    [Parameter(Mandatory)][string]$Tag,
    [string]$BuildDir = (Join-Path $PSScriptRoot '..\..\build-win-release'),
    [string]$OutDir = (Join-Path ($env:AJI_WIN ?? (Join-Path $env:USERPROFILE 'aji-win')) 'dist'),
    [string]$TrtRtxRoot = ($env:AJI_TRTRTX_ROOT ??
        (Join-Path ($env:AJI_WIN ?? (Join-Path $env:USERPROFILE 'aji-win')) 'trtrtx171'))
)
$ErrorActionPreference = 'Stop'

function Get-Artifact([string]$name) {
    $p = Join-Path $BuildDir $name
    if (-not (Test-Path $p)) { throw "missing build artifact: $p" }
    $p
}

$engine = foreach ($f in 'aji.dll', 'aji_dml.dll', 'aji_trt.dll', 'aji_encode.exe',
                         'aji_harness.exe', 'aji_harness_dml.exe', 'aji_kernel_test.exe') {
    Get-Artifact $f
}
$trtrtx = @(Get-Artifact 'aji_trtrtx.dll')
# the TensorRT-RTX runtime: exactly one (versioned) core and ONNX parser DLL
foreach ($pat in 'tensorrt_rtx_*.dll', 'tensorrt_onnxparser_rtx_*.dll') {
    $hit = @(Get-ChildItem (Join-Path $TrtRtxRoot $pat) -ErrorAction SilentlyContinue)
    if ($hit.Count -ne 1) { throw "expected one $pat in ${TrtRtxRoot}, found $($hit.Count)" }
    $trtrtx += $hit[0].FullName
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
function New-FlatZip([string]$name, [string[]]$paths) {
    $stage = Join-Path $OutDir "$Tag\stage"
    Remove-Item -Recurse -Force $stage -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Force $stage | Out-Null
    foreach ($p in $paths) { Copy-Item $p $stage }
    $zip = Join-Path $OutDir "$Tag\$name"
    Remove-Item -Force $zip -ErrorAction SilentlyContinue
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $zip
    Remove-Item -Recurse -Force $stage
    $z = [System.IO.Compression.ZipFile]::OpenRead($zip)
    $z.Entries | ForEach-Object { '{0,12:N0}  {1}' -f $_.Length, $_.FullName }
    $z.Dispose()
    '{0}  ({1:N1} MB)' -f $zip, ((Get-Item $zip).Length / 1MB)
}

New-FlatZip 'aji-windows-x64.zip' $engine
New-FlatZip 'aji-trtrtx-windows-x64.zip' $trtrtx
