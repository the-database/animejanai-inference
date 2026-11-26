#!/usr/bin/env python3
"""Qualify libaji's native CUDA backend on whatever NVIDIA GPU this Linux machine has.

Fetches the CUDA 12.9 compiler and TensorRT 11.4 into a work folder, builds the backend for this
GPU's architecture only at each revision asked for, downloads the shipped models the chosen cases
need, and times the builds against each other with sanity/paired_bench.py. Prints one summary
table and saves it with the raw pairs of every comparison. Uses only the Python standard library.

    gpu_check.py --expect-sm 89 --base REVISION             # has anything regressed since REVISION?
    gpu_check.py --pull 153                                 # main against a pull request, video cases
    gpu_check.py --compare head:pr153                       # the same on every case, images included
    gpu_check.py --compare head:all:HD --parity head:all    # is the tuned HD policy worth enabling here?

A build is named base (--base), head (--head, default origin/main), prN (a pull request), or one of
untuned, order, fuse, both, all: head with the HD SPANF3 launch policy forced off, or with the FP16
row order, the gate/projection fusion, both, or both plus the packed FP8 tiles forced on, whatever
the GPU. Those five exist to measure a policy on an architecture that does not have it yet.

Without --compare, --base is timed against head on every case and head against each --pull on the
video cases. Re-running skips finished steps: builds are kept per architecture and commit, timings
per GPU, driver, pair of builds and benchmark arguments. The exit status is nonzero when a requested
timing or check failed or did not run on the native kernels.
Needs Linux x86-64, compute capability 8.0, 8.6, 8.9 or 12.0, a driver for CUDA 12.9, g++, git, curl.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

REPO = 'https://github.com/the-database/animejanai-inference.git'
NVIDIA = 'https://developer.download.nvidia.com/compute/cuda'
CUDA_PARTS = [  # redistrib_12.9.1.json
    ('cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz',
     '7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35'),
    ('cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.9.79-archive.tar.xz',
     '1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99'),
    ('cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.9.27-archive.tar.xz',  # <nv/target>, which cuda_fp16.h includes
     '8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa'),
]
TRT = '11.4.0.106-1+cuda12.9'
TRT_PARTS = [  # the ubuntu2204 package index
    ('libnvinfer11', '734884ec92a3a6a42466c85016b540aef200b87637b061d90ab0d8f8f26b6e13'),
    ('libnvinfer-dev', '2bd7281b4d3acbd88e18697fb69cd8009935a03e8b20c67c0669719c453900a4'),
    ('libnvinfer-headers-dev', 'dbf415605b40fccfb68defa4373434b933210998b7183316dfdc120feab3acdc'),
]
SEVENZIP = ('https://github.com/ip7z/7zip/releases/download/26.05/7z2605-linux-x64.tar.xz',
            '22cf82fcc6eb7474f1d5bb4f02c0f8ebd573a718089d6833dc30f8350a3b232e')
PLAYER = 'https://github.com/the-database/mpv-AnimeJaNai/releases/download/3.12.5'
MANGA = 'https://github.com/the-database/MangaJaNai/releases/download'

HD_BAL = '2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16'
HD_PERF = '2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16'
SD = '2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo'
TFIX = ['temporalfix_s1_v1.1_op18_fp16', 'temporalfix_s2_v1_op18_fp16', 'temporalfix_s3_v1_op18_fp16']
# case name: (configuration code, file); 4.14 is the configuration default
# 4.7 has the two-convolution encoder, 4.12 lite its own widths, 4.20 ensemble the reverse pass and a 384-channel first block
RIFE = {'RIFE 4.26': (426, 'rife_v4.26'), 'RIFE 4.14': (414, 'rife_v4.14'), 'RIFE 4.14 lite': (4141, 'rife_v4.14_lite'),
        'RIFE 4.18': (418, 'rife_v4.18'), 'RIFE 4.22': (422, 'rife_v4.22'), 'RIFE 4.22 lite': (4221, 'rife_v4.22_lite'),
        'RIFE 4.25': (425, 'rife_v4.25'), 'RIFE 4.7': (47, 'rife_v4.7'), 'RIFE 4.12 lite': (4121, 'rife_v4.12_lite'),
        'RIFE 4.20 ensemble': (420, 'rife_v4.20_ensemble')}
RIFE_HEAVY = 'rife_v4.26_heavy'
SUPER_ULTRA = '2x_AnimeJaNai_Strong_V1_SuperUltraCompact_net_g_100000'
ESRGAN = '2x_MangaJaNai_2048p_V1_ESRGAN_95k_fp16.onnx'
ESRGAN_4X = '4x_MangaJaNai_2048p_V1_ESRGAN_70k_fp16.onnx'
FDAT_M = '2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.onnx'
SPAN_S = '2x_IllustrationJaNai_V3denoise_SPAN_S_30k_fp16.onnx'
FDAT_M_4X = '4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.onnx'
FDAT_XL = '4x_IllustrationJaNai_V3denoise_FDAT_XL_32k_fp16.onnx'
HAT_L = '4x_IllustrationJaNai_V3detail_HAT_L_28k_bf16.onnx'
DAT2 = '4x_IllustrationJaNai_V3detail_DAT2_28k_bf16.onnx'
# name: (url, sha256 as published with the release, folder under models/, files)
ARCHIVES = {
    'video': (f'{PLAYER}/mpv-upscale-2x_animejanai-overlay-3.12.5-linux-x64.7z',
              '961ba4a54488d2b8d8d91db556db21fe65b5d806ae09a283f71c7bc7fcc7d28c', 'video', [f'{n}.onnx' for n in (HD_BAL, HD_PERF, SD, *TFIX)]),
    'rife': (f'{PLAYER}/component-rife-linux-x64.7z',
             'e30f50c2671f4020af46a66334d3b4443e4b5c0e3f1310ba7a94ee505b18b684', 'video', [f'{f}.onnx' for f in (*(f for _, f in RIFE.values()), RIFE_HEAVY)]),
    'compact-v1': ('https://github.com/the-database/mpv-AnimeJaNai/releases/download/1.0.0/mpv-upscale-2x_animejanai_v1.zip',
                   'b19e6a782626bbf433da3aba0063b95b0bc18efc2bed52980029b3505000ad78', 'video', [f'{SUPER_ULTRA}.onnx']),  # computed here
    'manga': (f'{MANGA}/1.0.0/MangaJaNai_V1_ONNX_fp16.7z',
              '2c3809d712e3e96d107e2a59802186c284fac49d91a211435c8c896432ba27ab', 'img', [ESRGAN, ESRGAN_4X]),
    'illustration': (f'{MANGA}/3.0.0/IllustrationJaNai_V3_ONNX_fp16.7z',
                     'bd7db0632b37b95b06410e99f95347980010b2ccc54730e2d68501f25bae898c', 'img', [FDAT_M, SPAN_S, FDAT_M_4X, FDAT_XL]),
    'illustration-bf16': (f'{MANGA}/3.0.0/IllustrationJaNai_V3_ONNX_bf16.7z',
                          '48e292c3f92aa2df79f9f83c6318a6c2052c29e6ea9fe2ed23b61cc526900247', 'img', [HAT_L, DAT2]),
}
# FP16 row order, gate/projection fusion, packed FP8 tiles
TUNING = {'untuned': (False, False, False), 'order': (True, False, False), 'fuse': (False, True, False),
          'both': (True, True, False), 'all': (True, True, True)}

ROOT = SDK = LOGS = LIBS = OUT = CUDA = TRT_ROOT = TRT_LIB = VIDEO = IMAGES = HEAVY = None
SCRIPTS = None  # the commit whose sanity/ scripts do the timing
T0 = time.time()


def say(text):
    print(f'[{int(time.time()-T0)//60:3d}:{int(time.time()-T0)%60:02d}] {text}', flush=True)


class Failed(RuntimeError):
    pass


def run(cmd, log, cwd=None, env=None):
    """Run a command with its output appended to logs/<log>.log; raise with the log tail on failure."""
    path = LOGS/f'{log}.log'
    with path.open('a') as f:
        f.write(f'\n$ {" ".join(map(str, cmd))}\n')
        f.flush()
        rc = subprocess.call(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, cwd=cwd, env=env)
    if rc:
        tail = ''.join(path.open(errors='replace').readlines()[-25:])
        raise Failed(f'{log} failed (exit {rc}); end of {path}:\n{tail}')


def capture(cmd, cwd=None):
    return subprocess.check_output(list(map(str, cmd)), text=True, cwd=cwd).strip()


def last_line(error):
    return [l for l in str(error).splitlines() if l.strip()][-1][:150]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, sha256):
    """Download to dest unless a file with the pinned checksum is already there."""
    if dest.exists() and digest(dest) == sha256:
        return dest
    tmp = dest.with_name(dest.name + '.part')
    run(['curl', '-fL', '--retry', '4', '--retry-delay', '3', '-sS', '-o', tmp, url], 'download')
    if digest(tmp) != sha256:
        tmp.unlink()
        raise Failed(f'{url}: checksum mismatch')
    tmp.replace(dest)
    return dest


def gpu():
    """The GPU CUDA will use as device 0: name, driver, compute capability x10, memory, UUID."""
    rows = [[s.strip() for s in line.split(',')] for line in capture(
        ['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,compute_cap,memory.total', '--format=csv,noheader']).splitlines()]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None:
        visible = visible.split(',')[0].strip()
        if not visible:
            raise Failed('CUDA_VISIBLE_DEVICES is set but empty, so CUDA sees no GPU')
        by_index = visible.isdigit()
        if by_index and len(rows) > 1 and os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
            raise Failed('several GPUs: put a GPU UUID in CUDA_VISIBLE_DEVICES, or also set CUDA_DEVICE_ORDER=PCI_BUS_ID')
        rows = [r for r in rows if (r[0] == visible if by_index else r[1].startswith(visible))]
        if len(rows) != 1:
            raise Failed(f'CUDA_VISIBLE_DEVICES={visible} matches {len(rows)} of the GPUs nvidia-smi lists; give one full GPU UUID')
    elif len({tuple(r[2:5]) for r in rows}) > 1:
        raise Failed('several different GPUs: choose one with CUDA_VISIBLE_DEVICES=<GPU UUID>')
    _, uuid, name, driver, cap, mem = rows[0]
    return name, driver, int(cap.replace('.', '')), mem, uuid


def tools():
    for tool in ('g++', 'git', 'curl', 'xz', 'dpkg-deb'):
        if not shutil.which(tool):
            raise Failed(f'{tool} is missing')
    pip = [sys.executable, '-m', 'pip', 'install', '-q']
    version = re.search(r'(\d+)\.(\d+)', capture(['cmake', '--version'])) if shutil.which('cmake') else None
    if not version or (int(version[1]), int(version[2])) < (3, 24):
        run(pip + ['cmake>=3.24'], 'tools')
    if not shutil.which('ninja'):
        run(pip + ['ninja'], 'tools')
    seven = SDK/'7z'/'7zz'
    if not seven.exists():
        seven.parent.mkdir(parents=True, exist_ok=True)
        archive = fetch(SEVENZIP[0], SDK/'7z.tar.xz', SEVENZIP[1])
        run(['tar', '-xf', archive, '-C', seven.parent, '7zz'], 'tools')
        archive.unlink()
    return seven


def toolkits():
    CUDA.mkdir(parents=True, exist_ok=True)
    for rel, sha in CUDA_PARTS:
        done = CUDA/('.' + rel.split('/')[0])
        if not done.exists():
            archive = fetch(f'{NVIDIA}/redist/{rel}', SDK/Path(rel).name, sha)
            run(['tar', '-xf', archive, '-C', CUDA, '--strip-components=1'], 'toolkits')
            archive.unlink()
            done.touch()
    if not (CUDA/'lib64').exists():
        os.symlink('lib', CUDA/'lib64')  # nvcc links from lib64, the redistributable archives use lib
    if not (SDK/'trt'/'.done').exists():
        (SDK/'trt').mkdir(parents=True, exist_ok=True)
        for pkg, sha in TRT_PARTS:
            deb = fetch(f'{NVIDIA}/repos/ubuntu2204/x86_64/{pkg}_{TRT}_amd64.deb', SDK/f'{pkg}.deb', sha)
            if pkg == 'libnvinfer11':  # 1.4 GB, mostly per-GPU builder resources; only libnvinfer.so is used
                run(['bash', '-c', f'set -o pipefail; dpkg-deb --fsys-tarfile "{deb}" | tar -C "{SDK/"trt"}" -x --wildcards '
                     f'"./usr/lib/x86_64-linux-gnu/libnvinfer.so.*"'], 'toolkits')
            else:
                run(['dpkg-deb', '-x', deb, SDK/'trt'], 'toolkits')
            deb.unlink()
        (SDK/'trt'/'.done').touch()


def models(seven, needed):
    for name in sorted(needed):
        url, sha, folder, names = ARCHIVES[name]
        folder = ROOT/'models'/folder
        folder.mkdir(parents=True, exist_ok=True)
        if all((folder/n).exists() for n in names):
            continue
        archive = fetch(url, ROOT/'models'/url.rsplit('/', 1)[1], sha)
        run([seven, 'e', '-y', '-bd', f'-o{folder}', archive, *names, '-r'], 'models')
        archive.unlink()
        missing = [n for n in names if not (folder/n).exists()]
        if missing:
            raise Failed(f'{url} has no {missing}')
    if 'rife' in needed and not (HEAVY/'rife_v4.26.onnx').exists():
        # Releases before the Heavy selector resolve code 4262 to the standard model, so Heavy is timed through
        # code 426 from a folder in which it has the standard model's name.
        HEAVY.mkdir(parents=True, exist_ok=True)
        shutil.copy2(VIDEO/f'{RIFE_HEAVY}.onnx', HEAVY/'rife_v4.26.onnx')


def environment():
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = ':'.join(filter(None, [str(CUDA/'lib'), str(TRT_LIB), env.get('LD_LIBRARY_PATH')]))
    return env


class Builds:
    """libaji_trt.so for this architecture, per named build: libs/sm<N>/<name>-<commit>/libaji_trt.so."""

    def __init__(self, sm, base, head):
        self.sm, self.src, self.libs = sm, ROOT/'src', {}
        if not (self.src/'.git').exists():
            run(['git', 'clone', '-q', REPO, self.src], 'sources')
        run(['git', 'fetch', '-q', '--tags', 'origin'], 'sources', cwd=self.src)
        self.revisions = {'head': self.commit(head)}
        if base:
            self.revisions['base'] = self.commit(base)

    def commit(self, revision):
        try:
            return capture(['git', 'rev-parse', '--verify', '-q', f'{revision}^{{commit}}'], cwd=self.src)
        except subprocess.CalledProcessError:
            raise Failed(f'no commit {revision} in {REPO}')

    def tree(self, kind, sha):
        """A worktree per kind of build, at the commit and with its tracked files as committed."""
        tree = ROOT/kind
        if not tree.exists():
            run(['git', 'worktree', 'add', '-f', '--detach', tree, sha], 'sources', cwd=self.src)
        else:  # never touch a folder that is not a worktree of this script's own clone
            try:
                common = capture(['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], cwd=tree)
            except (subprocess.CalledProcessError, OSError):
                common = ''
            if not common or Path(common).resolve() != (self.src/'.git').resolve():
                raise Failed(f'{tree} exists and is not a worktree of {self.src}; choose another --root')
        # A build that was killed while its source was patched leaves the patch behind. Restore index and files
        # from the commit, then refuse to build anything that still differs from it.
        if capture(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=tree):
            run(['git', 'reset', '-q', '--hard', 'HEAD'], 'sources', cwd=tree)
        if capture(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=tree):
            raise Failed(f'{tree} still differs from its commit after a reset; remove the folder and run again')
        if capture(['git', 'rev-parse', 'HEAD'], cwd=tree) != sha:
            run(['git', 'checkout', '-q', '--detach', sha], 'sources', cwd=tree)
        return tree

    def resolve(self, name):
        """(worktree, commit, tuning flags or None) of a named build."""
        if name in TUNING:
            return 'forced', self.revisions['head'], TUNING[name]  # never patched in the tree head is built from
        if re.fullmatch(r'pr\d+', name):
            if name not in self.revisions:
                run(['git', 'fetch', '-q', 'origin', f'pull/{name[2:]}/head'], 'sources', cwd=self.src)
                self.revisions[name] = capture(['git', 'rev-parse', 'FETCH_HEAD'], cwd=self.src)
            return 'pull', self.revisions[name], None
        if name not in self.revisions:
            raise Failed(f'unknown build "{name}"; use base (with --base), head, prN or one of {", ".join(TUNING)}')
        return name, self.revisions[name], None

    def describe(self, name):
        return f'{name} ({self.resolve(name)[1][:7]})'

    def make(self, tree, log, target):
        bld = tree/f'build-sm{self.sm}'
        if not (bld/'build.ninja').exists():
            run(['cmake', '-S', tree, '-B', bld, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
                 f'-DCMAKE_CUDA_COMPILER={CUDA}/bin/nvcc', f'-DCUDAToolkit_ROOT={CUDA}', f'-DAJI_TRT_ROOT={TRT_ROOT}',
                 f'-DCMAKE_CUDA_ARCHITECTURES={self.sm}-real', f'-DAJI_NATIVE_CUDA_ARCHITECTURES={self.sm}-real'], log)
        run(['cmake', '--build', bld, '--target', target], log)
        return bld

    def lib(self, name):
        if name in self.libs:
            return self.libs[name]
        kind, sha, flags = self.resolve(name)
        target = LIBS/f'sm{self.sm}'/f'{name}-{sha[:7]}'/'libaji_trt.so'
        if not target.exists():
            say(f'building {self.describe(name)} for sm_{self.sm}')
            tree = self.tree(kind, sha)
            source = tree/'src'/'native'/'executor.cpp'
            try:
                if flags:
                    source.write_text(forced(source.read_text(), *flags))
                bld = self.make(tree, f'build-{name}', 'aji_trt')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(bld/'libaji_trt.so', target.with_suffix('.tmp'))
                target.with_suffix('.tmp').replace(target)
            finally:
                if flags:
                    run(['git', 'reset', '-q', '--hard', 'HEAD'], 'sources', cwd=tree)
        self.libs[name] = target
        return target

    def gate_test(self, name):
        """Build and run the repository's own GPU test of the HD kernels at that build's commit: (passed, text)."""
        kind, sha, flags = self.resolve(name)
        if flags:
            raise Failed('--gate-test takes base, head or prN')
        label, log = f'aji_native_gate_reuse_test at {self.describe(name)}', f'gate-test-{name}-run'
        (LOGS/f'{log}.log').unlink(missing_ok=True)
        try:
            bld = self.make(self.tree(kind, sha), f'gate-test-{name}', 'aji_native_gate_reuse_test')
            run([bld/'aji_native_gate_reuse_test', VIDEO/f'{HD_BAL}.onnx', VIDEO/f'{HD_PERF}.onnx', IMAGES/SPAN_S, 'all'],
                log, env=environment())
        except Failed as e:
            return False, f'{label}: FAILED: {last_line(e)}'
        skipped = re.search(r'\[fp8\]: [^\n]*(unavailable|unsupported)[^\n]*', (LOGS/f'{log}.log').read_text(errors='replace'))
        return True, f'{label}: passed' + (f'; FP8 SKIPPED ({skipped[0]})' if skipped else ' in default, fp16 and fp8 modes')


def forced(text, order, fuse, packed):
    """executor.cpp with each use of the device check for the HD launch policy replaced by a constant."""
    check = r'(?:is_sm120|use_hd_span_tuning)\(\)'
    for pattern, on in ((r'(packed_links_ && )' + check, packed), (r'(uses_fp8\(\) \|\| !)' + check, order),
                        (r'(4 \* W_ \|\| !)' + check, fuse)):
        text, count = re.subn(pattern, r'\1' + ('true' if on else 'false'), text)
        if count != 1:
            raise Failed(f'executor.cpp no longer matches {pattern}; update forced() in this script')
    # the helper is now unused; keep the compiler quiet about it
    return re.sub(r'\nbool (is_sm120|use_hd_span_tuning)\(\)', r'\n[[maybe_unused]] bool \1()', text, count=1)


def cases(sm):
    """(name, paired_bench arguments, model folder, group, archives needed, runs FP8, call or setting both builds must have)"""
    (ROOT/'builtins.conf').write_text('[global]\nconfig_version=2\nnative=on\nlogging=yes\n')
    head = '[global]\nconfig_version=2\nbackend=TensorRT\nnative=on\nlogging=yes\n'
    slot = '[slot_{}]\nchain_1_max_fps=120\n'
    (ROOT/'tfix.conf').write_text(head + ''.join(slot.format(i+1) + f'chain_1_temporal_fix_model={m}\n' for i, m in enumerate(TFIX)))
    (ROOT/'rife.conf').write_text(head + ''.join(
        slot.format(i+1) + f'chain_1_rife=yes\nchain_1_rife_model={code}\nchain_1_rife_factor_numerator=2\n'
        'chain_1_rife_factor_denominator=1\nchain_1_rife_before_upscale=yes\n' + ('chain_1_rife_ensemble=yes\n' if 'ensemble' in f else '')
        for i, (code, f) in enumerate(RIFE.values())))
    (ROOT/'compact.conf').write_text(head + slot.format(1) + f'chain_1_model_1_name={SUPER_ULTRA}\n')
    (ROOT/'fp8.conf').write_text(head + ''.join(slot.format(i) + f'chain_1_model_1_name={m}\nnative_span_precision=fp8\n'
                                                 for i, m in ((1, HD_PERF), (2, HD_BAL))))
    video = [('HD Quality', ['--slot', 1001], 'video'), ('HD Balanced', ['--slot', 1002], 'video'), ('HD Performance', ['--slot', 1003], 'video'),
             ('SD Compact', ['--slot', 1002, '--width', 720, '--height', 480], 'video')]
    video += [(f'TemporalFix S{i+1}', ['--case', 'temporal', '--conf', ROOT/'tfix.conf', '--slot', i+1], 'video') for i in range(3)]
    video += [(n, ['--case', 'rife', '--conf', ROOT/'rife.conf', '--slot', i+1], 'rife') for i, n in enumerate(RIFE)]
    video.append(('SuperUltraCompact', ['--conf', ROOT/'compact.conf', '--slot', 1, '--width', 1280, '--height', 720], 'compact-v1'))
    catalog = [(n, a, VIDEO, 'video', {need}, False, None) for n, a, need in video]
    catalog.append(('RIFE 4.26 Heavy', ['--case', 'rife', '--conf', ROOT/'rife.conf', '--slot', 1], HEAVY, 'video', {'rife'}, False, None))
    catalog.append(('RIFE 4.26 ids', ['--case', 'rife-ids', '--conf', ROOT/'rife.conf', '--slot', 1], VIDEO, 'video', {'rife'}, False,
                    'aji_infer_rife_ids'))
    if sm >= 89:  # FP8 needs the SM 89 instruction set
        catalog += [(n, ['--conf', ROOT/'fp8.conf', '--slot', s], VIDEO, 'video', {'video'}, True, 'native_span_precision')
                    for n, s in (('FP8 Performance', 1), ('FP8 Balanced', 2))]
    image = lambda model, size, pairs, tile=None, height=None: [
        '--case', 'image', '--image-model', model, '--width', size, '--height', height or size, '--tile', tile or size, '--pairs', pairs]
    catalog += [(n, a, IMAGES, 'image', {need}, False, None) for n, a, need in (
        ('ESRGAN 1536', image(ESRGAN, 1536, 20), 'manga'), ('ESRGAN 4x 512', image(ESRGAN_4X, 512, 16), 'manga'),
        ('SPAN-S 1536', image(SPAN_S, 1536, 20), 'illustration'), ('FDAT-M 1536', image(FDAT_M, 1536, 20), 'illustration'),
        ('FDAT-M 4x 512', image(FDAT_M_4X, 512, 16), 'illustration'), ('FDAT-XL 4x 512', image(FDAT_XL, 512, 16), 'illustration'),
        ('HAT-L 512', image(HAT_L, 512, 16), 'illustration-bf16'), ('DAT2 512', image(DAT2, 512, 16), 'illustration-bf16'),
        # a page of overlapping, blended tiles, and a page that DAT pads to its 32-pixel attention windows
        ('ESRGAN 256 tiles', image(ESRGAN, 1200, 16, tile=256, height=1000), 'manga'),
        ('DAT2 320x272', image(DAT2, 320, 16, tile=512, height=272), 'illustration-bf16'))]
    return catalog


def bench(identity, label, ref, cand, args, model_dir, fp8, swap):
    """One paired comparison. Raw pairs are kept in results/ with everything that decides a timing, and reused only
    when all of it is the same: GPU, driver, both builds, the timing scripts, the arguments, the configuration file's
    contents, the model folder and the AJI_* variables."""
    settings = {'gpu': identity, 'reference': ref.parent.name, 'candidate': cand.parent.name, 'scripts': SCRIPTS,
                'arguments': list(map(str, args)), 'swapped': swap, 'models': model_dir.name,
                'configurations': [digest(Path(str(x))) for x in args if str(x).endswith('.conf')],
                'environment': {k: v for k, v in sorted(os.environ.items()) if k.startswith('AJI_') and k != 'AJI_ROOT'}}
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    out = OUT/f'{label}{"-swapped" if swap else ""}-{key}.json'
    if out.exists():
        try:
            json.loads(out.read_text())
        except ValueError:
            out.unlink()  # damaged by something outside this script: measure again
    if not out.exists():
        cmd = [sys.executable, ROOT/'tools'/'sanity'/'paired_bench.py', '--reference', ref, '--candidate', cand,
               '--cuda-runtime', CUDA/'lib'/'libcudart.so.12', '--models', model_dir, '--output', out.with_suffix('.tmp'), *args]
        try:
            run(cmd + (['--swap-load-order'] if swap else []), f'bench-{out.stem}', env=environment())
        except Failed as e:
            return {'error': last_line(e)}
        j = json.loads(out.with_suffix('.tmp').read_text())
        j['gpu_check'] = settings
        out.with_suffix('.tmp').write_text(json.dumps(j, indent=1))
        out.with_suffix('.tmp').replace(out)  # the final name only ever holds a complete result
    j = json.loads(out.read_text())
    stats = j.get('gpu') or j['wall']
    problems = []
    for arm in ('reference', 'candidate'):
        log = j['memory'][arm].get('dispatch', '')
        if not re.search(r'[Nn]ative kernels: ', log) or re.search(r'[Nn]ative kernels not used', log):
            problems.append(f'{arm} NOT NATIVE')
        elif fp8 and 'FP8 SPANF3 body' not in log:
            problems.append(f'{arm} NOT FP8')
    return {'ref_ms': stats['reference_ms'], 'cand_ms': stats['candidate_ms'], 'pct': stats['paired_change_pct'],
            'lo': stats['interval95_pct'][0], 'hi': stats['interval95_pct'][1], 'problems': problems,
            'ref_mib': j['memory']['reference']['warm_bytes']/2**20, 'cand_mib': j['memory']['candidate']['warm_bytes']/2**20}


def has(lib, token):
    """Whether a build contains an exported call or a setting name."""
    return token.encode() in lib.read_bytes()


def dispatch(lib, conf, slot):
    """The model log a build prints for a slot, from paired_bench's per-build worker; empty if it could not run."""
    done = subprocess.run(list(map(str, [sys.executable, ROOT/'tools'/'sanity'/'paired_bench.py', '--reference', lib, '--candidate', lib,
                                         '--cuda-runtime', CUDA/'lib'/'libcudart.so.12', '--models', VIDEO, '--conf', conf, '--slot', slot,
                                         '--memory-worker', 'reference'])), capture_output=True, text=True, env=environment())
    try:
        return json.loads(done.stdout.splitlines()[-1])['dispatch']
    except (IndexError, KeyError, ValueError):
        return ''


def compare(identity, title, ref, cand, selected, both_orders, rows):
    """Time every case; a video case that moved by 1% or more is timed again with the load order swapped."""
    say(f'timing: {title}')
    block = []
    for name, args, model_dir, group, _, fp8, call in selected:
        label = f'{ref.parent.name}-vs-{cand.parent.name}-{name}'.replace(' ', '_')
        if call and not has(ref, call):  # an older reference without this call or setting: nothing to compare
            block.append({'case': name, 'skipped': f'{ref.parent.name} has no {call}'})
        elif call and not has(cand, call):  # the candidate lost something the reference has
            block.append({'case': name, 'error': f'{cand.parent.name} has no {call}, which {ref.parent.name} has'})
        if block and block[-1]['case'] == name:
            say('   ' + line(block[-1]))
            continue
        row = bench(identity, label, ref, cand, args, model_dir, fp8, False)
        row['case'] = name
        if 'error' not in row and (both_orders or (group == 'video' and abs(row['pct']) >= 1)):
            row['swapped'] = bench(identity, label, ref, cand, args, model_dir, fp8, True)
        block.append(row)
        say('   ' + line(row))
    rows.append((title, block))


def direction(r):
    """faster or SLOWER: 1% or more with the interval clear of zero. same: the whole interval within 1%.
    not slower: possibly faster, but the interval rules out a 1% slowdown. Anything else is inconclusive."""
    if r['pct'] <= -1 and r['hi'] < 0:
        return 'faster'
    if r['pct'] >= 1 and r['lo'] > 0:
        return 'SLOWER'
    if r['lo'] > -1 and r['hi'] < 1:
        return 'same'
    return 'not slower' if r['hi'] < 1 else 'inconclusive'


def broken(r):
    """Why a row cannot be trusted, or an empty list."""
    if 'skipped' in r:
        return []
    runs = [r] + ([r['swapped']] if 'swapped' in r else [])
    return [x['error'] for x in runs if 'error' in x] + [p for x in runs for p in x.get('problems', [])]


def line(r):
    if 'skipped' in r:
        return f'{r["case"]:<18}SKIPPED: {r["skipped"]}'
    if 'error' in r:
        return f'{r["case"]:<18}FAILED: {r["error"]}'
    change = lambda x: f'{x["pct"]:+6.2f}% [{x["lo"]:+.2f},{x["hi"]:+.2f}]'
    verdict, swapped = direction(r), ''
    if 'swapped' in r:
        s = r['swapped']
        swapped = ' | sw FAILED' if 'error' in s else f' | sw {change(s)}'
        # the two load orders have to agree before a direction is reported
        both = {verdict} if 'error' in s else {verdict, direction(s)}
        verdict = ('inconclusive' if 'error' in s else verdict if len(both) == 1 else
                   'not slower' if both <= {'faster', 'same', 'not slower'} else 'inconclusive')
    notes = ''.join(f' ({p})' for p in dict.fromkeys(broken(r)))
    return (f'{r["case"]:<18}{r["ref_ms"]:8.3f} ->{r["cand_ms"]:8.3f} ms {change(r)}{swapped}'
            f' | {r["ref_mib"]:.0f}->{r["cand_mib"]:.0f} MiB | {verdict}{notes}')


def parity(label, ref, cand, conf, slot):
    """Outputs of one build against another with sanity/direct_graph.py, 90 outputs per slot: (passed, text)."""
    out = OUT/'parity-tmp'
    shutil.rmtree(out, ignore_errors=True)
    log = f'parity-{label}'.replace(' ', '_').replace('>', '')
    (LOGS/f'{log}.log').unlink(missing_ok=True)
    try:
        run([sys.executable, ROOT/'tools'/'sanity'/'direct_graph.py', '--backend', cand, '--reference-backend', ref,
             '--cuda-runtime', CUDA/'lib'/'libcudart.so.12', '--conf', conf, '--models', VIDEO, '--slot', slot,
             '--width', 1920, '--height', 1080, '--output', out], log, env=environment())
    except Failed as e:
        return False, f'{label}: FAILED: {last_line(e)}'
    finally:
        shutil.rmtree(out, ignore_errors=True)
    found = re.findall(r'\d+ outputs matched[^\n]*', (LOGS/f'{log}.log').read_text(errors='replace'))
    return bool(found), f'{label}: {found[-1] if found else "FAILED: ran, but printed no match line"}'


def pair(text, parts):
    fields = text.split(':', parts - 1)
    if len(fields) < 2:
        raise argparse.ArgumentTypeError('expected REFERENCE:CANDIDATE')
    return fields + [''] * (parts - len(fields))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path(os.environ.get('AJI_ROOT', '/content/aji')), help='work folder (default /content/aji)')
    p.add_argument('--expect-sm', type=int, help='stop before downloading anything unless the GPU is this architecture, e.g. 89')
    p.add_argument('--base', help='revision to compare head against')
    p.add_argument('--head', default='origin/main', help='revision under test (default origin/main)')
    p.add_argument('--tools', default='head', help='revision or prN whose sanity/ scripts do the timing (default head)')
    p.add_argument('--pull', type=int, action='append', default=[], help='pull request to time against head on the video cases; repeatable')
    p.add_argument('--compare', type=lambda s: pair(s, 3), action='append', default=[], metavar='REF:CAND[:REGEX]',
                   help='time two named builds on the cases whose name matches REGEX (default every case); replaces the default comparisons')
    p.add_argument('--parity', type=lambda s: pair(s, 2), action='append', default=[], metavar='REF:CAND',
                   help='compare HD outputs of two named builds, FP16 and where available FP8')
    p.add_argument('--gate-test', action='append', default=[], metavar='BUILD', help='build and run aji_native_gate_reuse_test at base, head or prN')
    p.add_argument('--both-orders', action='store_true', help='time every comparison in both load orders')
    a = p.parse_args()

    global ROOT, SDK, LOGS, LIBS, OUT, CUDA, TRT_ROOT, TRT_LIB, VIDEO, IMAGES, HEAVY, SCRIPTS
    ROOT = a.root.resolve()
    SDK, LOGS, LIBS, OUT = ROOT/'sdk', ROOT/'logs', ROOT/'libs', ROOT/'results'
    CUDA, TRT_ROOT = SDK/'cuda', SDK/'trt'/'usr'
    TRT_LIB, VIDEO, IMAGES = TRT_ROOT/'lib'/'x86_64-linux-gnu', ROOT/'models'/'video', ROOT/'models'/'img'
    HEAVY = ROOT/'models'/'heavy'

    name, driver, sm, mem, uuid = gpu()
    say(f'GPU: {name} ({mem}), compute capability {sm/10:.1f}, driver {driver}')
    if a.expect_sm and sm != a.expect_sm:
        raise Failed(f'expected compute capability {a.expect_sm/10:.1f}; this is not the GPU that was asked for')
    if sm not in (80, 86, 89, 120):
        raise Failed(f'the native backend has no kernels for compute capability {sm/10:.1f}')
    for folder in (SDK, LOGS, LIBS, OUT):
        folder.mkdir(parents=True, exist_ok=True)

    catalog = cases(sm)
    comparisons = [(ref, cand, [c for c in catalog if re.search(pattern, c[0])]) for ref, cand, pattern in a.compare]
    if not a.compare:
        comparisons = ([('base', 'head', catalog)] if a.base else []) + [
            ('head', f'pr{n}', [c for c in catalog if c[3] == 'video']) for n in a.pull]
    if not (comparisons or a.parity or a.gate_test):
        raise Failed('nothing to do: give --base, --pull, --compare, --parity or --gate-test')
    empty = [f'{ref}:{cand}' for ref, cand, selected in comparisons if not selected]
    if empty:
        raise Failed(f'no case matches the pattern of --compare {empty[0]}; cases: {", ".join(c[0] for c in catalog)}')
    needed = {archive for _, _, selected in comparisons for c in selected for archive in c[4]}
    needed |= ({'video'} if a.parity or a.gate_test else set()) | ({'illustration'} if a.gate_test else set())

    say('tools, CUDA 12.9 compiler and TensorRT 11.4 headers'); seven = tools(); toolkits()
    say('sources'); builds = Builds(sm, a.base, a.head)
    SCRIPTS = builds.resolve(a.tools)[1] if re.fullmatch(r'pr\d+|head|base', a.tools) else builds.commit(a.tools)
    builds.tree('tools', SCRIPTS)  # sanity/paired_bench.py and sanity/direct_graph.py are run from this revision
    say('models: ' + ', '.join(sorted(needed))); models(seven, needed)
    libs = {n: builds.lib(n) for c in comparisons + a.parity for n in c[:2]}  # build everything before timing anything

    rows, checks, failed = [], [], 0
    for build in a.gate_test:
        say(f'running the GPU test at {build}')
        ok, text = builds.gate_test(build)
        checks.append(text); failed += not ok; say('   ' + text)
    for ref, cand in a.parity:
        say(f'comparing outputs: {ref} -> {cand}')
        slots = [('Balanced FP16', ROOT/'builtins.conf', 1002), ('Performance FP16', ROOT/'builtins.conf', 1003)]
        slots += [('Performance FP8', ROOT/'fp8.conf', 1), ('Balanced FP8', ROOT/'fp8.conf', 2)] if sm >= 89 else []
        for label, conf, slot in slots:
            label, ok, text = f'{ref} -> {cand} {label}', True, ''
            if 'FP8' in label:  # the same outputs prove nothing about FP8 unless both builds selected it
                setting, marker = 'native_span_precision', 'FP8 SPANF3 body'
                if not has(libs[ref], setting):
                    text = f'{label}: SKIPPED, {ref} has no {setting}'
                elif not has(libs[cand], setting):
                    ok, text = False, f'{label}: FAILED: {cand} has no {setting}, which {ref} has'
                else:
                    plain = [n for n in (ref, cand) if marker not in dispatch(libs[n], conf, slot)]
                    if plain:
                        ok, text = False, f'{label}: FAILED: {plain[0]} did not select FP8'
            if not text:
                ok, text = parity(label, libs[ref], libs[cand], conf, slot)
            checks.append(text); failed += not ok; say('   ' + text)
        if sm < 89:
            checks.append(f'{ref} -> {cand} FP8: SKIPPED, this GPU has no FP8 kernels')
    for ref, cand, selected in comparisons:
        compare([uuid, driver, sm], f'{builds.describe(ref)} -> {builds.describe(cand)}', libs[ref], libs[cand], selected, a.both_orders, rows)
    failed += sum(bool(broken(r)) for _, block in rows for r in block)

    text = [f'GPU {name} ({mem}) sm_{sm}, driver {driver} | CUDA 12.9, TensorRT {TRT.split("-")[0]} | {int(time.time()-T0)//60} min',
            'ms per frame (video: GPU time; image: wall time), change with 95% interval,',
            '"sw" = the same comparison with the load order swapped, then warm VRAM.',
            'faster/SLOWER: 1% or more, interval clear of zero. same: interval within 1%. not slower: interval rules out',
            'a 1% slowdown. Both load orders have to agree when both ran; anything else is inconclusive.']
    for title, block in rows:
        text += ['', f'== {title}'] + [line(r) for r in block]
    text += (['', '== checks'] + checks) if checks else []
    text += ['', f'{failed} requested timing(s) or check(s) FAILED'] if failed else []
    (ROOT/'summary.txt').write_text('\n'.join(text) + '\n')
    (ROOT/'summary.json').write_text(json.dumps({'gpu': [name, mem, sm, driver, uuid], 'rows': rows, 'checks': checks, 'failed': failed}, indent=1))
    print('\n' + '=' * 30 + ' SUMMARY ' + '=' * 30 + '\n' + '\n'.join(text), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Failed as e:
        print(f'\nSTOPPED: {e}', flush=True)
        sys.exit(1)
