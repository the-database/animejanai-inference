"""Check SPAN fallback, explicit settings, cache invalidation and failed-DLL lifetime.

Run on Windows with a supported CUDA GPU and no concurrent GPU workload.
Uses the existing named_profiles harness, fresh isolated files and stock engines.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

p = argparse.ArgumentParser(description=__doc__)
for key in ['backend', 'runtime', 'models', 'output', 'failed-plugin']:
    p.add_argument('--' + key, type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
repo = Path(__file__).resolve().parents[1]
base_conf = (repo / 'experimental/span-concat/profiles.conf').read_text()
(a.output / 'profiles.conf').write_text(base_conf)
custom = base_conf.replace('--builderOptimizationLevel=5 --optShapes=input:%video_resolution% --skipInference',
    '--builderOptimizationLevel=2 --optShapes=input:%video_resolution% --tilingOptimizationLevel=0 --l2LimitForTiling=33554432 --skipInference')
(a.output / 'custom.conf').write_text(custom)
env = os.environ.copy()
for key in ['AJI_DIRECT_GRAPH', 'AJI_NO_GRAPH', 'AJI_RIFE_WARP']:
    env.pop(key, None)
results = {}

def models_at(name, stock_only=False, engines_from=None):
    dest = a.output / name
    dest.mkdir()
    for f in a.models.glob('*.onnx'):
        if not stock_only or '_SPANF3_' in f.name:
            shutil.copyfile(f, dest / f.name)
    if engines_from:
        for f in engines_from.glob('*.engine'):
            shutil.copyfile(f, dest / f.name)
    return dest

def backend_at(name, plugins=False, failed=False, changed=False):
    dest = a.output / name
    dest.mkdir()
    shutil.copyfile(a.backend, dest / 'aji_trt.dll')
    if plugins:
        for name in ['balanced-next-plugin.dll', 'performance-plugin.dll']:
            shutil.copyfile(a.backend.parent / name, dest / name)
            if changed:
                # A PE overlay changes binary identity while preserving executable code.
                with (dest / name).open('ab') as f:
                    f.write(b'aji-cache-invalidation-regression')
    if failed:
        shutil.copyfile(a.failed_plugin, dest / 'balanced-next-plugin.dll')
    return dest / 'aji_trt.dll'

def run(name, backend, models, size=(640, 480), blocking=False, custom=False, fallback=True, lifetime=False):
    w, h = size
    frame = a.output / f'{w}x{h}.nv12'
    if not frame.exists():
        np.concatenate([np.arange(w * h, dtype=np.uint8), np.full(w * h // 2, 128, np.uint8)]).tofile(frame)
    dest = a.output / name
    script = 'span_plugin_lifetime.py' if lifetime else 'named_profiles.py'
    slots = '2,1' if lifetime else '2,1,4,3' if fallback else '2,4'
    args = [sys.executable, str(repo / 'sanity' / script), '--backend', str(backend.resolve()),
            '--runtime', str(a.runtime.resolve()), '--models', str(models.resolve()),
            '--conf', str((a.output / ('custom.conf' if custom else 'profiles.conf')).resolve()),
            '--frames', str(frame.resolve()), '--width', str(w), '--height', str(h),
            '--slots', slots, '--output', str(dest.resolve())]
    if blocking:
        args += ['--blocking']
    child_env = env.copy()
    if lifetime:
        child_env['AJI_TEST_FAILED_PLUGIN'] = str(backend.parent / 'balanced-next-plugin.dll')
    print('Checking', name, flush=True)
    with (a.output / (name + '.log')).open('w') as log:
        subprocess.run(args, env=child_env, stdout=log, stderr=subprocess.STDOUT, check=True)
    data = json.loads((dest / 'results.json').read_text())
    if fallback:
        hashes = {row['slot']: row['sha256'] for row in data['rows']}
        assert hashes[1] == hashes[2]
        if not lifetime:
            assert hashes[3] == hashes[4]
        for row in data['configure']:
            if row['slot'] in (2, 4):
                assert 'Using stock model for ' in row['configuration'], row
    else:
        assert all('Using stock model for ' not in row['configuration'] for row in data['configure'])
        assert any('.plugin-' in name for name in data['engines'])
    results[name] = dict(passed=True, enginenames=data['engines'], switching_identical=data['switching_identical'])
    (a.output / 'summary.json').write_text(json.dumps(results, indent=2))
    print('Passed', name, flush=True)

small = models_at('models-640')
missing = backend_at('no-plugin')
run('missing-dll-async', missing, small)
run('missing-dll-blocking', missing, small, blocking=True)
stock_only = models_at('stock-only', stock_only=True, engines_from=small)
run('missing-export', a.backend, stock_only)
unusual = models_at('models-1918')
run('unsupported-shape-async', a.backend, unusual, size=(1918, 1078))
run('unsupported-shape-blocking', a.backend, unusual, size=(1918, 1078), blocking=True)
run('custom-settings', a.backend, small, custom=True, fallback=False)
before = {f.name for f in small.glob('*.plugin-*.engine')}
assert len(before) == 2, before
for f in small.glob('*.plugin-*.engine.build.log'):
    log = f.read_text(encoding='utf-8', errors='replace')
    for option in ['--builderOptimizationLevel=2', '--tilingOptimizationLevel=0', '--l2LimitForTiling=33554432']:
        assert option in log, (f, option)
changed = backend_at('changed-plugin', plugins=True, changed=True)
run('changed-binary', changed, small, custom=True, fallback=False)
after = {f.name for f in small.glob('*.plugin-*.engine')}
assert len(after - before) == 2, (before, after)
failed = backend_at('failed-registration', failed=True)
run('failed-registration-lifetime', failed, small, lifetime=True)

# A damaged optional export must not prevent its stock model from running.
# Reversed profile switching in run() also exercises the failed-build latch.
for blocking in [False, True]:
    name = 'failed-build-' + ('blocking' if blocking else 'async')
    broken = models_at(name + '-models', engines_from=stock_only)
    for f in broken.glob('*_Concat_*.onnx'):
        f.write_bytes(b'invalid optional model for fallback regression')
    run(name, a.backend, broken, blocking=blocking)
    assert len(list(broken.glob('*.plugin-*.engine.build.log'))) == 2
print('PASS: stock fallback in blocking/async modes, optional build failure, explicit settings, binary cache identity and DLL lifetime')
