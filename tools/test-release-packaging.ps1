param([Parameter(Mandatory)][string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $root) { throw 'Use a new test directory.' }
$build = Join-Path $root 'build'
$rtx = Join-Path $root 'rtx'
New-Item -ItemType Directory -Path $build,(Join-Path $rtx 'doc') -Force | Out-Null
foreach ($name in 'aji.dll','aji_dml.dll','aji_trt.dll','aji_encode.exe','aji_harness.exe','aji_harness_dml.exe','aji_kernel_test.exe','aji_trtrtx.dll','aji_analysis.dll','aji_analysis_trt.dll') {
    Set-Content -LiteralPath (Join-Path $build $name) -Value 'fixture'
}
foreach ($name in 'tensorrt_rtx_1.dll','tensorrt_onnxparser_rtx_1.dll','doc/Acknowledgements.txt') {
    Set-Content -LiteralPath (Join-Path $rtx $name) -Value 'fixture'
}
foreach ($mode in 'OFF','ON') {
    Set-Content -LiteralPath (Join-Path $build 'CMakeCache.txt') -Value "AJI_BUILD_ANALYSIS:BOOL=$mode"
    # All generated stage/delete paths remain beneath this fresh fixture root.
    & (Join-Path $PSScriptRoot '../ci/windows/package-aji-release.ps1') -Tag $mode -BuildDir $build -OutDir (Join-Path $root 'dist') -TrtRtxRoot $rtx | Out-Null
    $zip = [IO.Compression.ZipFile]::OpenRead((Join-Path $root "dist/$mode/aji-windows-x64.zip"))
    try {
        $names = @($zip.Entries.FullName)
        foreach ($name in 'aji_analysis.dll','aji_analysis_trt.dll') {
            if (($names -contains $name) -ne ($mode -eq 'ON')) { throw "Incorrect $mode package member: $name" }
        }
        # TensorRT comes from the consumer's install or the shared runtime packs.
        $bundled = @($names | Where-Object { $_ -like 'nv*' })
        if ($bundled.Count) { throw "$mode package bundles TensorRT files: $($bundled -join ', ')" }
    } finally { $zip.Dispose() }
    Write-Output "PASS analysis $mode packaging (stale analysis DLL present)"
}
