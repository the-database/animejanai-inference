#!/usr/bin/env python3
"""Build the shared runtime packs: one release (TAG) that every app installs from, into the
per-user pack store that `aji-runtime` (the-database/aji-runtime) manages.

NVIDIA ships TensorRT as one archive with builder resources for every GPU generation (several GB).
This splits it into packs so a machine downloads only what it needs, and adds the other packs the
apps share:

    trt-runtime-<platform>.7z      nvinfer, the plugin and ONNX parser libraries, trtexec, cudart,
                                   NVIDIA's notices
    trt-<smNN|ptx>-<platform>.7z   the builder resources for one GPU generation (PTX: the fallback
                                   for generations without their own)
    trtrtx-runtime-windows-x64.7z  the TensorRT for RTX runtime (two DLLs, never tensorrt_rtx.exe)
                                   + cudart, which aji_trtrtx.dll also imports, + notices
    dml-runtime-windows-x64.7z     ONNX Runtime + DirectML. Not a store pack ("kind": "private"):
                                   apps keep it in their own folder
    models-*                       the MangaJaNai / IllustrationJaNai model archives, not
                                   re-uploaded: their entries point at the MangaJaNai releases

plus packs-<platform>.json, the index apps pin by SHA-256 (format 1):

    {"format": 1, "tag": "runtime-2", "platform": "windows-x64",
     "packs": [{"name": "trt-runtime", "version": "11.4.0.106-cuda13.4", "kind": "trt",
                "asset": "...", "url": "...", "sha256": "...", "bytes": N, "installed_bytes": M,
                "files": [{"path": "nvinfer_11.dll", "sha256": "...", "bytes": n}, ...]}, ...]}

    kind: trt | trt-builder (+ "sm": "sm120" / "ptx") | trtrtx | models | private
    version: the content version (store folder packs/<name>/<version>), not the release tag

Files sit at each archive's root (models included). Every source is pinned below and checked
against its SHA-256:
    windows  NVIDIA's redistributable TensorRT and TensorRT-RTX zips + the CUDA redistributable
             cudart zip + the ONNX Runtime / DirectML NuGet packages
    linux    NVIDIA's CUDA apt-repo packages (per-file SHA-256 from its Packages index) + the CUDA
             redistributable cudart tarball. Needs dpkg-deb (Debian/Ubuntu).
    models   the MangaJaNai model releases

usage: runtime_packs.py --print-tag
       runtime_packs.py --platform windows|linux --out DIR [--work DIR] [--local-dir DIR ...]
                        [--sevenzip PATH] [--no-models]
  --local-dir  take the TensorRT / TensorRT-RTX / cudart / ORT files from folders that already
               hold them (flat), instead of downloading; repeat for several folders
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

# The release these packs go on. Bump it whenever a pinned version below changes; apps pin it and
# each index's SHA-256. (runtime-1 is MangaJaNaiCLI's older, CLI-only release.)
TAG = "runtime-2"
REPO = "the-database/animejanai-inference"

TRT_VERSION = "11.4.0.106"
TRT_CUDA = "13.4"
CUDART_VERSION = "13.4.49"
TRTRTX_VERSION = "1.7.1.116"
TRTRTX_CUDA = "13.4"

# NVIDIA publishes no checksum beside the Windows zips; recorded from verified downloads (the same
# pins as docs/BUILD-WINDOWS.md).
TRT_ZIP_SHA256 = "b6fd86301ad803492adf7a06de29ce45e3968d29637a1437d0fb0994f412c980"
TRTRTX_ZIP_SHA256 = "2002294a19f81135e4cfd074faddf1f842eb46979f36f7e6db57ff83016ba71d"
# From developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.1.json.
CUDART_SHA256 = {
    "windows": "e6663f3d3e8949eedc2d5ab92c7c5b9fa3f2a222086d91c42bfc5a38bf2b0225",
    "linux": "ec3df9f91560db23a064c6b395d4fde17b2893f5bf954c6bd193dd03a44de522",
}
# From the CUDA apt repo's Packages index (NVIDIA's own checksums).
CUDA_APT = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64"
DEB_SHA256 = {
    "libnvinfer11": "bd24db760728779e980c8852ba87e148a0bf9fb4e69d69bc14f3bdabc0423797",
    "libnvinfer-plugin11": "a99bd3de4fd45585d4a1fce8f6a8d6415e1a7946c86d907f64af0b9b487aee66",
    "libnvonnxparsers11": "d7b0e623e6b8d1c368a000b996938494137e2584b85e2eed8b89f7b34092436c",
    "libnvinfer-bin": "36b0a4f11c3a15bf9f2d4eb5a33ce02b7248582eb384696a1d80a154abf005f8",
}

# ONNX Runtime + DirectML for aji_dml.dll, from NuGet; SHA-256 recorded from a verified download.
ORT_DML = ("Microsoft.ML.OnnxRuntime.DirectML", "1.24.4", "57e9f11b73437bef7a309496135d4c1f96b1a8e9ddba60013fa27bfc1d788681",
           {"runtimes/win-x64/native/onnxruntime.dll": "onnxruntime.dll"})
DIRECTML = ("Microsoft.AI.DirectML", "1.15.4", "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9",
            {"bin/x64-win/DirectML.dll": "DirectML.dll", "LICENSE.txt": "DirectML_LICENSE.txt"})

# The model archives MangaJaNai GUI and MangaJaNaiCLI install: (pack, release tag, asset, sha256).
MODELS_REPO = "the-database/MangaJaNai"
MODELS = [
    ("models-mangajanai", "1.0.0", "MangaJaNai_V1_ONNX_fp16.7z", "2c3809d712e3e96d107e2a59802186c284fac49d91a211435c8c896432ba27ab"),
    ("models-illustrationjanai", "3.0.0", "IllustrationJaNai_V3_ONNX_fp16.7z", "bd7db0632b37b95b06410e99f95347980010b2ccc54730e2d68501f25bae898c"),
    ("models-illustrationjanai-fp32", "3.0.0", "IllustrationJaNai_V3_ONNX_fp32.7z", "8b38bfdc95f194fb18eaa51d7ca869839ebbb1c29bed1e7a943a65b372d090ac"),
    ("models-illustrationjanai-bf16", "3.0.0", "IllustrationJaNai_V3_ONNX_bf16.7z", "48e292c3f92aa2df79f9f83c6318a6c2052c29e6ea9fe2ed23b61cc526900247"),
]

HERE = os.path.dirname(os.path.abspath(__file__))
TRTRTX_NOTICE = os.path.join(HERE, "..", "ci", "windows", "TensorRT-RTX-NOTICE.txt")

WIN_TRT = {"nvinfer_11.dll", "nvinfer_plugin_11.dll", "nvonnxparser_11.dll", "trtexec.exe"}
LINUX_TRT = {"libnvinfer.so.11", "libnvinfer_plugin.so.11", "libnvonnxparser.so.11", "trtexec"}
TRTRTX_DLL = re.compile(r"tensorrt_(onnxparser_)?rtx_\d+_\d+\.dll")
CUDART = re.compile(r"cudart64_\d+\.dll|libcudart\.so\.\d+")
BUILDER = re.compile(r"builder_resource_([a-z0-9]+)[._]")
DML_FILES = {"onnxruntime.dll", "DirectML.dll", "DirectML_LICENSE.txt"}


def log(msg):
    print(msg, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(url, dest, want):
    if os.path.exists(dest) and sha256(dest) == want:
        log(f"cached {os.path.basename(dest)}")
        return dest
    log(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "animejanai-inference-runtime-packs"})
    with urllib.request.urlopen(req) as r, open(dest + ".part", "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    got = sha256(dest + ".part")
    if got != want:
        os.remove(dest + ".part")
        sys.exit(f"SHA-256 mismatch for {url}: got {got}, want {want}")
    os.replace(dest + ".part", dest)
    return dest


def copy_member(zf, info, path):
    with zf.open(info) as src, open(path, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)


# Each stage_* fills {pack name: folder} with that pack's files.

def stage_cudart(work, plat, dest):
    if plat == "windows":
        stem = f"cuda_cudart-windows-x86_64-{CUDART_VERSION}-archive"
        c = fetch(f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/{stem}.zip",
                  os.path.join(work, stem + ".zip"), CUDART_SHA256["windows"])
        with zipfile.ZipFile(c) as zf:
            for info in zf.infolist():
                name = os.path.basename(info.filename)
                if name == "LICENSE":
                    copy_member(zf, info, os.path.join(dest, "CUDA_LICENSE.txt"))
                elif CUDART.fullmatch(name):
                    copy_member(zf, info, os.path.join(dest, name))
    else:
        stem = f"cuda_cudart-linux-x86_64-{CUDART_VERSION}-archive"
        c = fetch(f"https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/{stem}.tar.xz",
                  os.path.join(work, stem + ".tar.xz"), CUDART_SHA256["linux"])
        with tarfile.open(c) as tf:
            members = {os.path.basename(m.name): m for m in tf.getmembers()}
            real = next(m for n, m in members.items() if re.fullmatch(r"libcudart\.so\.\d+\.\d+\.\d+", n))
            with tf.extractfile(real) as src, open(os.path.join(dest, "libcudart.so.13"), "wb") as dst:
                shutil.copyfileobj(src, dst)
            with tf.extractfile(members["LICENSE"]) as src, open(os.path.join(dest, "CUDA_LICENSE.txt"), "wb") as dst:
                shutil.copyfileobj(src, dst)


def builder_dir(stage, name):
    m = BUILDER.search(name)
    d = os.path.join(stage, f"trt-{m.group(1)}")
    os.makedirs(d, exist_ok=True)
    return d


def stage_trt_windows(work, stage):
    rt = os.path.join(stage, "trt-runtime")
    trt_dir = ".".join(TRT_VERSION.split(".")[:3])
    asset = f"TensorRT-Enterprise-{TRT_VERSION}-Windows-amd64-cuda-{TRT_CUDA}-Release-external.zip"
    z = fetch(f"https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/{trt_dir}/zip/{asset}",
              os.path.join(work, asset), TRT_ZIP_SHA256)
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            name = os.path.basename(info.filename)
            if "/bin/" in info.filename and name in WIN_TRT:
                copy_member(zf, info, os.path.join(rt, name))
            elif "/bin/" in info.filename and BUILDER.search(name):
                copy_member(zf, info, os.path.join(builder_dir(stage, name), name))
            elif name.lower() == "acknowledgements.txt":
                copy_member(zf, info, os.path.join(rt, "TensorRT_Acknowledgements.txt"))
    stage_cudart(work, "windows", rt)


def stage_trtrtx(work, stage):
    rt = os.path.join(stage, "trtrtx-runtime")
    asset = f"tensorrt-rtx-{TRTRTX_VERSION}-windows-amd64-cuda-{TRTRTX_CUDA}-release-external.zip"
    ver = ".".join(TRTRTX_VERSION.split(".")[:2])
    z = fetch(f"https://developer.nvidia.com/downloads/trt/rtx_sdk/secure/{ver}/{asset}",
              os.path.join(work, asset), TRTRTX_ZIP_SHA256)
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            name = os.path.basename(info.filename)
            if TRTRTX_DLL.fullmatch(name):
                copy_member(zf, info, os.path.join(rt, name))
            elif info.filename.endswith("/doc/Acknowledgements.txt"):
                copy_member(zf, info, os.path.join(rt, "TensorRT-RTX-Acknowledgements.txt"))
    shutil.copyfile(TRTRTX_NOTICE, os.path.join(rt, "TensorRT-RTX-NOTICE.txt"))
    stage_cudart(work, "windows", rt)


def stage_dml(work, stage):
    rt = os.path.join(stage, "dml-runtime")
    for name, ver, want, files in (ORT_DML, DIRECTML):
        p = fetch(f"https://www.nuget.org/api/v2/package/{name}/{ver}", os.path.join(work, f"{name}.{ver}.nupkg"), want)
        with zipfile.ZipFile(p) as zf:
            for src, dst in files.items():
                with zf.open(src) as s, open(os.path.join(rt, dst), "wb") as d:
                    shutil.copyfileobj(s, d, 1 << 20)


def stage_trt_linux(work, stage):
    rt = os.path.join(stage, "trt-runtime")
    root = os.path.join(work, "debroot")
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root)
    for pkg, want in DEB_SHA256.items():
        deb = f"{pkg}_{TRT_VERSION}-1+cuda{TRT_CUDA}_amd64.deb"
        p = fetch(f"{CUDA_APT}/{deb}", os.path.join(work, deb), want)
        subprocess.run(["dpkg-deb", "-x", p, root], check=True)
    lib = os.path.join(root, "usr", "lib", "x86_64-linux-gnu")
    # Libraries are stored under their sonames (what libaji_trt.so and trtexec ask for); the builder
    # resources keep their file names, which is how libnvinfer opens them.
    for soname in ("libnvinfer.so.11", "libnvinfer_plugin.so.11", "libnvonnxparser.so.11"):
        shutil.copyfile(os.path.realpath(os.path.join(lib, soname)), os.path.join(rt, soname))
    for name in os.listdir(lib):
        if BUILDER.search(name) and not os.path.islink(os.path.join(lib, name)):
            shutil.copyfile(os.path.join(lib, name), os.path.join(builder_dir(stage, name), name))
    # libnvinfer-bin puts trtexec in /usr/bin (11.x) or /usr/src/tensorrt/bin (10.x).
    trtexec = next(p for p in (os.path.join(root, "usr", "bin", "trtexec"),
                               os.path.join(root, "usr", "src", "tensorrt", "bin", "trtexec")) if os.path.exists(p))
    shutil.copyfile(trtexec, os.path.join(rt, "trtexec"))
    copyright_ = os.path.join(root, "usr", "share", "doc", "libnvinfer11", "copyright")
    if os.path.exists(copyright_):
        shutil.copyfile(copyright_, os.path.join(rt, "TensorRT_Acknowledgements.txt"))
    stage_cudart(work, "linux", rt)


def stage_local(dirs, plat, stage):
    """Take the runtime files from folders that already hold them (flat)."""
    for d in dirs:
        for name in sorted(os.listdir(d)):
            src = os.path.join(d, name)
            if not os.path.isfile(src):
                continue
            if name in (WIN_TRT if plat == "windows" else LINUX_TRT) or name == "TensorRT_Acknowledgements.txt":
                dest = os.path.join(stage, "trt-runtime")
            elif BUILDER.search(name):
                dest = builder_dir(stage, name)
            elif plat == "windows" and (TRTRTX_DLL.fullmatch(name) or name.startswith("TensorRT-RTX-")):
                dest = os.path.join(stage, "trtrtx-runtime")
            elif plat == "windows" and name in DML_FILES:
                dest = os.path.join(stage, "dml-runtime")
            elif CUDART.fullmatch(name) or name == "CUDA_LICENSE.txt":
                dest = os.path.join(stage, "trt-runtime")
            else:
                continue
            shutil.copyfile(src, os.path.join(dest, name))
    rtx = os.path.join(stage, "trtrtx-runtime")
    if plat == "windows" and os.listdir(rtx):
        if not os.path.exists(os.path.join(rtx, "TensorRT-RTX-NOTICE.txt")):
            shutil.copyfile(TRTRTX_NOTICE, os.path.join(rtx, "TensorRT-RTX-NOTICE.txt"))
        for f in os.listdir(os.path.join(stage, "trt-runtime")):
            if CUDART.fullmatch(f) or f == "CUDA_LICENSE.txt":
                shutil.copyfile(os.path.join(stage, "trt-runtime", f), os.path.join(rtx, f))


def file_list(folder):
    out = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            p = os.path.join(root, f)
            out.append({"path": os.path.relpath(p, folder).replace("\\", "/"), "sha256": sha256(p),
                        "bytes": os.path.getsize(p)})
    return sorted(out, key=lambda e: e["path"])


def pack_meta(name):
    """(version, kind, extra fields) of a staged pack."""
    if name == "trt-runtime":
        return f"{TRT_VERSION}-cuda{TRT_CUDA}", "trt", {}
    if name == "trtrtx-runtime":
        return f"{TRTRTX_VERSION}-cuda{TRTRTX_CUDA}", "trtrtx", {}
    if name == "dml-runtime":
        return f"ort-{ORT_DML[1]}+dml-{DIRECTML[1]}", "private", {}
    if name.startswith("trt-"):
        return TRT_VERSION, "trt-builder", {"sm": name[len("trt-"):]}
    sys.exit(f"unknown pack {name}")


def model_packs(work, sevenzip):
    out = []
    for name, tag, asset, want in MODELS:
        url = f"https://github.com/{MODELS_REPO}/releases/download/{tag}/{asset}"
        p = fetch(url, os.path.join(work, asset), want)
        x = os.path.join(work, "models-x", name)
        shutil.rmtree(x, ignore_errors=True)
        os.makedirs(x)
        subprocess.run([sevenzip, "x", "-y", "-bso0", "-bsp0", f"-o{x}", p], check=True)
        files = file_list(x)
        out.append({"name": name, "version": tag, "kind": "models", "asset": asset, "url": url,
                    "sha256": want, "bytes": os.path.getsize(p),
                    "installed_bytes": sum(f["bytes"] for f in files), "files": files})
        shutil.rmtree(x, ignore_errors=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-tag", action="store_true", help="print the release tag and exit")
    ap.add_argument("--platform", choices=["windows", "linux"])
    ap.add_argument("--out")
    ap.add_argument("--work", default="runtime-work")
    ap.add_argument("--local-dir", action="append", default=[])
    ap.add_argument("--no-models", action="store_true", help="leave the model packs out (testing)")
    ap.add_argument("--sevenzip", default=shutil.which("7z") or r"C:\Program Files\7-Zip\7z.exe")
    a = ap.parse_args()
    if a.print_tag:
        print(TAG)
        return
    if not a.platform or not a.out:
        ap.error("--platform and --out are required")
    plat = f"{a.platform}-x64"
    os.makedirs(a.work, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    stage = os.path.join(a.work, "stage")
    shutil.rmtree(stage, ignore_errors=True)
    for d in ("trt-runtime",) + (("trtrtx-runtime", "dml-runtime") if a.platform == "windows" else ()):
        os.makedirs(os.path.join(stage, d))
    if a.local_dir:
        stage_local(a.local_dir, a.platform, stage)
    elif a.platform == "windows":
        stage_trt_windows(a.work, stage)
        stage_trtrtx(a.work, stage)
        stage_dml(a.work, stage)
    else:
        stage_trt_linux(a.work, stage)
    if os.name != "nt" and os.path.exists(os.path.join(stage, "trt-runtime", "trtexec")):
        os.chmod(os.path.join(stage, "trt-runtime", "trtexec"), 0o755)

    groups = sorted(d for d in os.listdir(stage) if os.listdir(os.path.join(stage, d)))
    if "trt-runtime" not in groups or not any(g.startswith("trt-") and g != "trt-runtime" for g in groups):
        sys.exit(f"incomplete runtime: {groups}")
    index = {"format": 1, "tag": TAG, "platform": plat, "packs": []}
    for name in groups:
        folder = os.path.join(stage, name)
        files = file_list(folder)
        asset = f"{name}-{plat}.7z"
        dest = os.path.abspath(os.path.join(a.out, asset))
        if os.path.exists(dest):
            os.remove(dest)
        log(f"packing {asset}: {', '.join(f['path'] for f in files)}")
        # Plain LZMA2 without the BCJ/BCJ2 executable filters 7-Zip adds to DLLs by default: what
        # aji-runtime decodes (sevenz-rust2) is LZMA2.
        subprocess.run([a.sevenzip, "a", "-t7z", "-m0=LZMA2", "-mx=7", "-mf=off", "-bso0", "-bsp0", dest]
                       + [f["path"] for f in files], cwd=folder, check=True)
        version, kind, extra = pack_meta(name)
        index["packs"].append({
            "name": name, "version": version, "kind": kind, **extra,
            "asset": asset, "url": f"https://github.com/{REPO}/releases/download/{TAG}/{asset}",
            "sha256": sha256(dest), "bytes": os.path.getsize(dest),
            "installed_bytes": sum(f["bytes"] for f in files), "files": files,
        })
    if not a.no_models:
        index["packs"] += model_packs(a.work, a.sevenzip)
    with open(os.path.join(a.out, f"packs-{plat}.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(index, f, indent=1)
        f.write("\n")
    for p in index["packs"]:
        log(f"{p['name']}/{p['version']} ({p['kind']}): {p['bytes'] / 1e6:.0f} MB "
            f"({p['installed_bytes'] / 1e6:.0f} MB unpacked)")
    log(f"packs-{plat}.json sha256 {sha256(os.path.join(a.out, f'packs-{plat}.json'))}")


if __name__ == "__main__":
    main()
