param([Parameter(Mandatory)][string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $root) { throw 'Use a new test directory.' }
$build = Join-Path $root 'build'
$trt = Join-Path $root 'trt/lib'
$rtx = Join-Path $root 'rtx'
New-Item -ItemType Directory -Path $build,$trt,(Join-Path $rtx 'doc') -Force | Out-Null
foreach ($name in 'aji.dll','aji_dml.dll','aji_trt.dll','aji_encode.exe','aji_harness.exe','aji_harness_dml.exe','aji_kernel_test.exe','aji_trtrtx.dll','aji_analysis.dll') {
    Set-Content -LiteralPath (Join-Path $build $name) -Value 'fixture'
}
foreach ($name in 'nvonnxparser_11.dll','nvinfer_builder_resource_11.dll','nvinfer_lean_11.dll') {
    Set-Content -LiteralPath (Join-Path $trt $name) -Value 'fixture'
}
foreach ($name in 'tensorrt_rtx_1.dll','tensorrt_onnxparser_rtx_1.dll','doc/Acknowledgements.txt') {
    Set-Content -LiteralPath (Join-Path $rtx $name) -Value 'fixture'
}
foreach ($mode in 'OFF','ON') {
    Set-Content -LiteralPath (Join-Path $build 'CMakeCache.txt') -Value "AJI_BUILD_ANALYSIS:BOOL=$mode"
    # All generated stage/delete paths remain beneath this fresh fixture root.
    & (Join-Path $PSScriptRoot '../ci/windows/package-aji-release.ps1') -Tag $mode -BuildDir $build -OutDir (Join-Path $root 'dist') -TrtRoot (Split-Path $trt) -TrtRtxRoot $rtx | Out-Null
    $zip = [IO.Compression.ZipFile]::OpenRead((Join-Path $root "dist/$mode/aji-windows-x64.zip"))
    try {
        $names = @($zip.Entries.FullName)
        foreach ($name in 'aji_analysis.dll','nvonnxparser_11.dll','nvinfer_builder_resource_11.dll','nvinfer_lean_11.dll') {
            if (($names -contains $name) -ne ($mode -eq 'ON')) { throw "Incorrect $mode package member: $name" }
        }
    } finally { $zip.Dispose() }
    Write-Output "PASS analysis $mode packaging (stale analysis DLL present)"
}
