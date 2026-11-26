#!/usr/bin/env python3
"""Qualify libaji's native CUDA backend on whatever NVIDIA GPU this Linux machine has.

Fetches the CUDA 12.9 compiler and TensorRT 11.4 into a work folder, builds the backend for this
GPU's architecture only at each revision asked for, downloads the shipped models, and times the
builds against each other with sanity/paired_bench.py. Prints one summary table and saves it with
the raw pairs of every comparison. Uses only the Python standard library.

    gpu_check.py --expect-sm 89 --base REVISION             # has anything regressed since REVISION?
    gpu_check.py --pull 153                                 # main against a pull request
    gpu_check.py --compare head:all:HD --parity head:all    # is the tuned HD policy worth enabling here?

A build is named base (--base), head (--head, default origin/main), prN (a pull request), or one of
untuned, order, fuse, both, all: head with the HD SPANF3 launch policy forced off, or with the FP16
row order, the gate/projection fusion, both, or both plus the packed FP8 tiles forced on, whatever
the GPU. Those five exist to measure a policy on an architecture that does not have it yet.

Without --compare, --base is timed against head on every case and head against each --pull on the
video cases. Re-running skips finished steps; results are cached per pair of builds.
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
CUDA_PARTS = [
    ('cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.9.86-archive.tar.xz',
     '7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35'),
    ('cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.9.79-archive.tar.xz',
     '1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99'),
    ('cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.9.27-archive.tar.xz',  # <nv/target>, which cuda_fp16.h includes
     '8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa'),
]
TRT = '11.4.0.106-1+cuda12.9'
SEVENZIP = 'https://github.com/ip7z/7zip/releases/download/26.05/7z2605-linux-x64.tar.xz'
PLAYER = 'https://github.com/the-database/mpv-AnimeJaNai/releases/download/3.12.5'
MANGA = 'https://github.com/the-database/MangaJaNai/releases/download'

HD_BAL = '2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16'
HD_PERF = '2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16'
TFIX = ['temporalfix_s1_v1.1_op18_fp16', 'temporalfix_s2_v1_op18_fp16', 'temporalfix_s3_v1_op18_fp16']
VIDEO_FILES = [HD_BAL, HD_PERF, *TFIX, '2x_AnimeJaNai_SD_V1beta34_Compact_1x3xHxW_dyn-HW_strong_fp16_op21_dynamo']
ESRGAN = '2x_MangaJaNai_2048p_V1_ESRGAN_95k_fp16.onnx'
FDAT_M = '2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.onnx'
SPAN_S = '2x_IllustrationJaNai_V3denoise_SPAN_S_30k_fp16.onnx'
HAT_L = '4x_IllustrationJaNai_V3detail_HAT_L_28k_bf16.onnx'
DAT2 = '4x_IllustrationJaNai_V3detail_DAT2_28k_bf16.onnx'
IMAGE_PACKS = [(f'{MANGA}/1.0.0/MangaJaNai_V1_ONNX_fp16.7z', [ESRGAN]),
               (f'{MANGA}/3.0.0/IllustrationJaNai_V3_ONNX_fp16.7z', [FDAT_M, SPAN_S]),
               (f'{MANGA}/3.0.0/IllustrationJaNai_V3_ONNX_bf16.7z', [HAT_L, DAT2])]
# FP16 row order, gate/projection fusion, packed FP8 tiles
TUNING = {'untuned': (False, False, False), 'order': (True, False, False), 'fuse': (False, True, False),
          'both': (True, True, False), 'all': (True, True, True)}

ROOT = SDK = LOGS = LIBS = OUT = CUDA = TRT_ROOT = TRT_LIB = VIDEO = IMAGES = None
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


def fetch(url, dest, sha256=None):
    if not dest.exists():
        tmp = dest.with_name(dest.name + '.part')
        run(['curl', '-fL', '--retry', '4', '--retry-delay', '3', '-sS', '-o', tmp, url], 'download')
        if sha256:
            h = hashlib.sha256()
            with tmp.open('rb') as f:
                for chunk in iter(lambda: f.read(1 << 22), b''):
                    h.update(chunk)
            if h.hexdigest() != sha256:
                raise Failed(f'{url}: checksum mismatch')
        tmp.rename(dest)
    return dest


def gpu():
    line = capture(['nvidia-smi', '--query-gpu=name,driver_version,compute_cap,memory.total', '--format=csv,noheader']).splitlines()[0]
    name, driver, cap, mem = (s.strip() for s in line.split(','))
    return name, driver, int(cap.replace('.', '')), mem


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
        archive = fetch(SEVENZIP, SDK/'7z.tar.xz')
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
        repo = f'{NVIDIA}/repos/ubuntu2204/x86_64'
        # the runtime package is about 1.9 GB of per-GPU builder resources; only libnvinfer.so is used
        deb = fetch(f'{repo}/libnvinfer11_{TRT}_amd64.deb', SDK/'libnvinfer11.deb')
        run(['bash', '-c', f'set -o pipefail; dpkg-deb --fsys-tarfile "{deb}" | tar -C "{SDK/"trt"}" -x --wildcards '
             f'"./usr/lib/x86_64-linux-gnu/libnvinfer.so.*"'], 'toolkits')
        deb.unlink()
        for pkg in ('libnvinfer-dev', 'libnvinfer-headers-dev'):
            deb = fetch(f'{repo}/{pkg}_{TRT}_amd64.deb', SDK/f'{pkg}.deb')
            run(['dpkg-deb', '-x', deb, SDK/'trt'], 'toolkits')
            deb.unlink()
        (SDK/'trt'/'.done').touch()


def models(seven):
    VIDEO.mkdir(parents=True, exist_ok=True)
    IMAGES.mkdir(parents=True, exist_ok=True)
    jobs = [(f'{PLAYER}/mpv-upscale-2x_animejanai-overlay-3.12.5-linux-x64.7z', VIDEO, [f'{n}.onnx' for n in VIDEO_FILES]),
            (f'{PLAYER}/component-rife-linux-x64.7z', VIDEO, ['rife_v4.26.onnx'])]
    jobs += [(url, IMAGES, names) for url, names in IMAGE_PACKS]
    for url, folder, names in jobs:
        if all((folder/n).exists() for n in names):
            continue
        archive = fetch(url, ROOT/'models'/url.rsplit('/', 1)[1])
        run([seven, 'e', '-y', '-bd', f'-o{folder}', archive, *names, '-r'], 'models')
        archive.unlink()
        missing = [n for n in names if not (folder/n).exists()]
        if missing:
            raise Failed(f'{url} has no {missing}')


def environment():
    env = dict(os.environ)
    env['LD_LIBRARY_PATH'] = ':'.join(filter(None, [str(CUDA/'lib'), str(TRT_LIB), env.get('LD_LIBRARY_PATH')]))
    return env


class Builds:
    """libaji_trt.so for this GPU only, per named build, kept as libs/<name>-<commit>/libaji_trt.so."""

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

    def tree(self, name, sha):
        """A worktree per kind of build, moved to the commit, so later builds of that kind are incremental."""
        tree = ROOT/name
        if not (tree/'.git').exists():
            run(['git', 'worktree', 'add', '-f', '--detach', tree, sha], 'sources', cwd=self.src)
        elif capture(['git', 'rev-parse', 'HEAD'], cwd=tree) != sha:
            run(['git', 'checkout', '-q', '--detach', sha], 'sources', cwd=tree)
        return tree

    def resolve(self, name):
        """(worktree name, commit, tuning flags or None) of a named build."""
        if name in TUNING:
            return 'head', self.revisions['head'], TUNING[name]
        if re.fullmatch(r'pr\d+', name):
            if name not in self.revisions:
                run(['git', 'fetch', '-q', 'origin', f'pull/{name[2:]}/head'], 'sources', cwd=self.src)
                self.revisions[name] = capture(['git', 'rev-parse', 'FETCH_HEAD'], cwd=self.src)
            return 'pull', self.revisions[name], None
        if name not in self.revisions:
            raise Failed(f'unknown build "{name}"; use base (with --base), head, prN or one of {", ".join(TUNING)}')
        return name, self.revisions[name], None

    def make(self, tree, log, target):
        bld = tree/'build'
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
        target = LIBS/f'{name}-{sha[:7]}'/'libaji_trt.so'
        if not target.exists():
            say(f'building {name} ({sha[:7]}) for sm_{self.sm}')
            tree = self.tree(kind, sha)
            source = tree/'src'/'native'/'executor.cpp'
            original = source.read_text()
            try:
                if flags:
                    source.write_text(forced(original, *flags))
                bld = self.make(tree, f'build-{name}', 'aji_trt')
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(bld/'libaji_trt.so', target.with_suffix('.tmp'))
                target.with_suffix('.tmp').rename(target)
            finally:
                if flags:
                    source.write_text(original)
        self.libs[name] = target
        return target

    def gate_test(self, name):
        """Build and run the repository's own GPU test of the HD kernels at that build's commit."""
        kind, sha, flags = self.resolve(name)
        if flags:
            raise Failed('--gate-test takes base, head or prN')
        try:
            bld = self.make(self.tree(kind, sha), f'gate-test-{name}', 'aji_native_gate_reuse_test')
            run([bld/'aji_native_gate_reuse_test', VIDEO/f'{HD_BAL}.onnx', VIDEO/f'{HD_PERF}.onnx', IMAGES/SPAN_S, 'all'],
                f'gate-test-{name}', env=environment())
        except Failed as e:
            return f'aji_native_gate_reuse_test at {name} ({sha[:7]}): FAILED: {last_line(e)}'
        return f'aji_native_gate_reuse_test at {name} ({sha[:7]}): passed (default, fp16 and fp8 modes)'


def forced(text, order, fuse, packed):
    """executor.cpp with each use of the device check for the HD launch policy replaced by a constant."""
    check = r'(?:is_sm120|use_hd_span_tuning)\(\)'
    for pattern, on in ((r'(packed_links_ && )' + check, packed), (r'(uses_fp8\(\) \|\| !)' + check, order),
                        (r'(4 \* W_ \|\| !)' + check, fuse)):
        text, count = re.subn(pattern, r'\1' + ('true' if on else 'false'), text)
        if count != 1:
            raise Failed(f'executor.cpp no longer matches {pattern}; update forced() in this script')
    # the helper is now unused in most variants; keep the compiler quiet about it
    return re.sub(r'\nbool (is_sm120|use_hd_span_tuning)\(\)', r'\n[[maybe_unused]] bool \1()', text, count=1)


def cases(sm):
    """(name, paired_bench arguments, model folder, group)"""
    (ROOT/'builtins.conf').write_text('[global]\nconfig_version=2\nnative=on\nlogging=yes\n')
    head = '[global]\nconfig_version=2\nbackend=TensorRT\nnative=on\nlogging=yes\n'
    slot = '[slot_{}]\nchain_1_max_fps=120\n'
    (ROOT/'tfix.conf').write_text(head + ''.join(slot.format(i+1) + f'chain_1_temporal_fix_model={m}\n' for i, m in enumerate(TFIX)))
    (ROOT/'rife.conf').write_text(head + slot.format(1) + 'chain_1_rife=yes\nchain_1_rife_model=426\nchain_1_rife_factor_numerator=2\n'
                                  'chain_1_rife_factor_denominator=1\nchain_1_rife_before_upscale=yes\n')
    (ROOT/'fp8.conf').write_text(head + ''.join(slot.format(i) + f'chain_1_model_1_name={m}\nnative_span_precision=fp8\n'
                                                 for i, m in ((1, HD_PERF), (2, HD_BAL))))
    video = [('HD Quality', ['--slot', 1001]), ('HD Balanced', ['--slot', 1002]), ('HD Performance', ['--slot', 1003]),
             ('SD Compact', ['--slot', 1002, '--width', 720, '--height', 480])]
    video += [(f'TemporalFix S{i+1}', ['--case', 'temporal', '--conf', ROOT/'tfix.conf', '--slot', i+1]) for i in range(3)]
    video += [('RIFE 4.26', ['--case', 'rife', '--conf', ROOT/'rife.conf', '--slot', 1]),
              ('RIFE 4.26 ids', ['--case', 'rife-ids', '--conf', ROOT/'rife.conf', '--slot', 1])]
    if sm >= 89:  # FP8 needs the SM 89 instruction set
        video += [('FP8 Performance', ['--conf', ROOT/'fp8.conf', '--slot', 1]), ('FP8 Balanced', ['--conf', ROOT/'fp8.conf', '--slot', 2])]
    image = lambda model, size, pairs: ['--case', 'image', '--image-model', model, '--width', size, '--height', size,
                                        '--tile', size, '--pairs', pairs]
    images = [('ESRGAN 1536', image(ESRGAN, 1536, 20)), ('FDAT-M 1536', image(FDAT_M, 1536, 20)),
              ('SPAN-S 1536', image(SPAN_S, 1536, 20)), ('HAT-L 512', image(HAT_L, 512, 16)), ('DAT2 512', image(DAT2, 512, 16))]
    return [(n, a, VIDEO, 'video') for n, a in video] + [(n, a, IMAGES, 'image') for n, a in images]


def bench(label, ref, cand, args, model_dir, swap=False):
    """One paired comparison; returns a result row. Raw pairs are kept in results/<label>.json."""
    label += '-swapped' if swap else ''
    out = OUT/f'{label}.json'
    if not out.exists():
        cmd = [sys.executable, ROOT/'head'/'sanity'/'paired_bench.py', '--reference', ref, '--candidate', cand,
               '--cuda-runtime', CUDA/'lib'/'libcudart.so.12', '--models', model_dir, '--output', out.with_suffix('.tmp'), *args]
        try:
            run(cmd + (['--swap-load-order'] if swap else []), f'bench-{label}', env=environment())
        except Failed as e:
            return {'error': last_line(e)}
        out.with_suffix('.tmp').rename(out)
    j = json.loads(out.read_text())
    stats = j.get('gpu') or j['wall']
    dispatch = j['memory']['candidate'].get('dispatch', '')
    return {'ref_ms': stats['reference_ms'], 'cand_ms': stats['candidate_ms'], 'pct': stats['paired_change_pct'],
            'lo': stats['interval95_pct'][0], 'hi': stats['interval95_pct'][1],
            'ref_mib': j['memory']['reference']['warm_bytes']/2**20, 'cand_mib': j['memory']['candidate']['warm_bytes']/2**20,
            'native': 'native' in dispatch.lower()}


def compare(title, ref, cand, selected, both_orders, rows):
    """Time every case; a video case that moved by 1% or more is timed again with the load order swapped."""
    say(f'timing: {title}')
    block = []
    for name, args, model_dir, group in selected:
        tag = f'{ref.parent.name}-vs-{cand.parent.name}-{name}'.replace(' ', '_')
        row = bench(tag, ref, cand, args, model_dir)
        row['case'] = name
        if 'error' not in row and (both_orders or (group == 'video' and abs(row['pct']) >= 1)):
            row['swapped_pct'] = bench(tag, ref, cand, args, model_dir, swap=True).get('pct')
        block.append(row)
        say('   ' + line(row))
    rows.append((title, block))


def line(r):
    if 'error' in r:
        return f'{r["case"]:<15} FAILED: {r["error"]}'
    verdict = 'SLOWER' if r['pct'] >= 1 and r['lo'] > 0 else 'faster' if r['pct'] <= -1 and r['hi'] < 0 else 'same'
    swapped = f' sw {r["swapped_pct"]:+.2f}%' if r.get('swapped_pct') is not None else ''
    return (f'{r["case"]:<15}{r["ref_ms"]:8.3f} ->{r["cand_ms"]:8.3f} ms {r["pct"]:+6.2f}% [{r["lo"]:+.2f},{r["hi"]:+.2f}]{swapped}'
            f' | {r["ref_mib"]:.0f}->{r["cand_mib"]:.0f} MiB | {verdict}{"" if r["native"] else " (NOT NATIVE)"}')


def parity(label, ref, cand, conf, slot):
    """Outputs of one build against another: sanity/direct_graph.py compares 90 outputs per slot."""
    out = OUT/'parity-tmp'
    shutil.rmtree(out, ignore_errors=True)
    log = f'parity-{label}'.replace(' ', '_')
    (LOGS/f'{log}.log').unlink(missing_ok=True)
    try:
        run([sys.executable, ROOT/'head'/'sanity'/'direct_graph.py', '--backend', cand, '--reference-backend', ref,
             '--cuda-runtime', CUDA/'lib'/'libcudart.so.12', '--conf', conf, '--models', VIDEO, '--slot', slot,
             '--width', 1920, '--height', 1080, '--output', out], log, env=environment())
    except Failed as e:
        return f'{label}: FAILED: {last_line(e)}'
    finally:
        shutil.rmtree(out, ignore_errors=True)
    found = re.findall(r'\d+ outputs matched[^\n]*', (LOGS/f'{log}.log').read_text(errors='replace'))
    return f'{label}: {found[-1] if found else "ran, but printed no match line"}'


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
    p.add_argument('--pull', type=int, action='append', default=[], help='pull request to time against head; repeatable')
    p.add_argument('--compare', type=lambda s: pair(s, 3), action='append', default=[], metavar='REF:CAND[:REGEX]',
                   help='time two named builds on the cases whose name matches REGEX (default all); replaces the default comparisons')
    p.add_argument('--parity', type=lambda s: pair(s, 2), action='append', default=[], metavar='REF:CAND',
                   help='compare HD outputs of two named builds, FP16 and where available FP8')
    p.add_argument('--gate-test', action='append', default=[], metavar='BUILD', help='build and run aji_native_gate_reuse_test at base, head or prN')
    p.add_argument('--both-orders', action='store_true', help='time every comparison in both load orders')
    a = p.parse_args()

    global ROOT, SDK, LOGS, LIBS, OUT, CUDA, TRT_ROOT, TRT_LIB, VIDEO, IMAGES
    ROOT = a.root.resolve()
    SDK, LOGS, LIBS, OUT = ROOT/'sdk', ROOT/'logs', ROOT/'libs', ROOT/'results'
    CUDA, TRT_ROOT = SDK/'cuda', SDK/'trt'/'usr'
    TRT_LIB, VIDEO, IMAGES = TRT_ROOT/'lib'/'x86_64-linux-gnu', ROOT/'models'/'video', ROOT/'models'/'img'

    name, driver, sm, mem = gpu()
    say(f'GPU: {name} ({mem}), compute capability {sm/10:.1f}, driver {driver}')
    if a.expect_sm and sm != a.expect_sm:
        raise Failed(f'expected compute capability {a.expect_sm/10:.1f}; this is not the GPU that was asked for')
    if sm not in (80, 86, 89, 120):
        raise Failed(f'the native backend has no kernels for compute capability {sm/10:.1f}')
    for folder in (SDK, LOGS, LIBS, OUT):
        folder.mkdir(parents=True, exist_ok=True)
    say('tools, CUDA 12.9 compiler and TensorRT 11.4 headers'); seven = tools(); toolkits()
    say('sources'); builds = Builds(sm, a.base, a.head)
    builds.tree('head', builds.revisions['head'])  # the timing and parity scripts are taken from head
    say('models'); models(seven)

    comparisons = a.compare
    if not comparisons:
        comparisons = ([['base', 'head', '']] if a.base else []) + [['head', f'pr{n}', '^(?!.*(1536|512))'] for n in a.pull]
    if not (comparisons or a.parity or a.gate_test):
        raise Failed('nothing to do: give --base, --pull, --compare, --parity or --gate-test')
    libs = {n: builds.lib(n) for c in comparisons + a.parity for n in c[:2]}  # build everything before timing anything
    catalog = cases(sm)
    rows, checks = [], []
    for build in a.gate_test:
        say(f'running the GPU test at {build}'); checks.append(builds.gate_test(build)); say('   ' + checks[-1])
    for ref, cand in a.parity:
        say(f'comparing outputs: {ref} -> {cand}')
        slots = [('Balanced FP16', ROOT/'builtins.conf', 1002), ('Performance FP16', ROOT/'builtins.conf', 1003)]
        slots += [('Performance FP8', ROOT/'fp8.conf', 1), ('Balanced FP8', ROOT/'fp8.conf', 2)] if sm >= 89 else []
        for label, conf, slot in slots:
            checks.append(parity(f'{ref} -> {cand} {label}', libs[ref], libs[cand], conf, slot)); say('   ' + checks[-1])
    describe = lambda n: f'{n} ({builds.resolve(n)[1][:7]})'
    for ref, cand, pattern in comparisons:
        selected = [c for c in catalog if re.search(pattern, c[0])]
        compare(f'{describe(ref)} -> {describe(cand)}', libs[ref], libs[cand], selected, a.both_orders, rows)

    text = [f'GPU {name} ({mem}) sm_{sm}, driver {driver} | CUDA 12.9, TensorRT {TRT.split("-")[0]} | {int(time.time()-T0)//60} min',
            'ms per frame (video: GPU time; image: wall time), change with 95% interval,',
            '"sw" = the same comparison with the load order swapped, then warm VRAM']
    for title, block in rows:
        text += ['', f'== {title}'] + [line(r) for r in block]
    text += (['', '== checks'] + checks) if checks else []
    (ROOT/'summary.txt').write_text('\n'.join(text) + '\n')
    (ROOT/'summary.json').write_text(json.dumps({'gpu': [name, mem, sm, driver], 'rows': rows, 'checks': checks}, indent=1))
    print('\n' + '=' * 30 + ' SUMMARY ' + '=' * 30 + '\n' + '\n'.join(text), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Failed as e:
        print(f'\nSTOPPED: {e}', flush=True)
        sys.exit(1)
