# Run from the repository root in an x64 MSVC developer shell.
# SDKs are build dependencies only. This script does not publish a release.
$ErrorActionPreference = 'Stop'
$deps = [IO.Path]::GetFullPath('build-deps')
[IO.Directory]::CreateDirectory($deps) | Out-Null
function Fetch([string]$Url, [string]$Destination, [string]$Sha256 = '') {
    if (-not (Test-Path -LiteralPath $Destination)) { Invoke-WebRequest $Url -OutFile $Destination }
    if ($Sha256 -and (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash -ne $Sha256) { throw "SDK checksum mismatch: $Destination" }
}
if (-not (Test-Path -LiteralPath "$deps/cuda/bin/nvcc.exe")) {
    $manifest = Invoke-RestMethod 'https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.0.json'
    foreach ($component in @('cuda_nvcc', 'cuda_crt', 'cuda_cudart', 'cccl')) {
        $asset = $manifest.$component.'windows-x86_64'
        if (-not $asset.relative_path) { throw "Missing CUDA component: $component" }
        Fetch "https://developer.download.nvidia.com/compute/cuda/redist/$($asset.relative_path)" "$deps/$component.zip" $asset.sha256
        Expand-Archive -LiteralPath "$deps/$component.zip" -DestinationPath "$deps/$component" -Force
        $root = @(Get-ChildItem -LiteralPath "$deps/$component" -Directory)
        if ($root.Count -ne 1) { throw 'Unexpected CUDA SDK archive layout' }
        [IO.Directory]::CreateDirectory("$deps/cuda") | Out-Null
        Copy-Item -Path "$($root[0].FullName)/*" -Destination "$deps/cuda" -Recurse -Force
    }
}
if (-not (Test-Path -LiteralPath "$deps/trt/include/NvInfer.h")) {
    Fetch 'https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/11.1.0/zip/TensorRT-Enterprise-11.1.0.106-Windows-amd64-cuda-13.3-Release-external.zip' "$deps/trt.zip"
    # Extract only headers/import libraries from NVIDIA's full SDK.
    $zip = [IO.Compression.ZipFile]::OpenRead("$deps/trt.zip")
    try {
        foreach ($entry in $zip.Entries) {
            if ($entry.FullName -notmatch '(?:^|/)(include/[^/]+\.h|lib/[^/]+\.lib)$') { continue }
            $relative = $Matches[1]
            $target = Join-Path "$deps/trt" $relative
            [IO.Directory]::CreateDirectory((Split-Path $target)) | Out-Null
            [IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $target, $true)
        }
    } finally { $zip.Dispose() }
}
foreach ($package in @(@('Microsoft.ML.OnnxRuntime.DirectML','1.24.4','ort'), @('Microsoft.AI.DirectML','1.15.4','dml'))) {
    if (Test-Path -LiteralPath "$deps/$($package[2])") { continue }
    Fetch "https://api.nuget.org/v3-flatcontainer/$($package[0].ToLowerInvariant())/$($package[1])/$($package[0].ToLowerInvariant()).$($package[1]).nupkg" "$deps/$($package[2]).zip"
    Expand-Archive -LiteralPath "$deps/$($package[2]).zip" -DestinationPath "$deps/$($package[2])"
}
$env:CUDA_PATH = "$deps/cuda"
$env:PATH = "$env:CUDA_PATH/bin;$env:PATH"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_COMPILER=$deps/cuda/bin/nvcc.exe" "-DAJI_TRT_ROOT=$deps/trt" "-DAJI_TRT_LIB=$deps/trt/lib" -DAJI_NVINFER=nvinfer_11 "-DAJI_ORT_ROOT=$deps/ort" "-DAJI_DML_ROOT=$deps/dml"
if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed' }
cmake --build build --parallel 2
if ($LASTEXITCODE -ne 0) { throw 'Inference build failed' }
[IO.Directory]::CreateDirectory('build/Release') | Out-Null
foreach ($name in @('aji.dll','aji_dml.dll','aji_trt.dll','aji_harness.exe','aji_harness_dml.exe','aji_kernel_test.exe')) {
    Copy-Item -LiteralPath "build/$name" -Destination 'build/Release'
}
[ordered]@{ commit = (git rev-parse HEAD); cuda = '13.3.0'; tensorRT = '11.1.0.106'; onnxRuntime = '1.24.4'; directML = '1.15.4';
    tensorRTSdkSha256 = (Get-FileHash -LiteralPath "$deps/trt.zip" -Algorithm SHA256).Hash.ToLowerInvariant();
    validation = 'Compiled on Windows; GPU playback is tested separately.' } | ConvertTo-Json | Set-Content -LiteralPath 'build/build-info.json'
