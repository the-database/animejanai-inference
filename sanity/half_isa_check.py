"""Ensure the AVX object cannot provide scalar half.h COMDAT definitions.

Run for optimized and /Od /Ob0 builds; requires Visual Studio dumpbin.
"""
import argparse
import re
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dumpbin', default='dumpbin')
p.add_argument('object')
a = p.parse_args()
symbols = subprocess.check_output([a.dumpbin, '/symbols', a.object], text=True)
for line in symbols.splitlines():
    if 'External' in line and 'UNDEF' not in line and re.search(r'\?(?:f32_to_f16|f16_to_f32)@ajicore@@', line):
        raise SystemExit('FAIL: AVX object defines a scalar helper: ' + line)
if '?pack_f16c@ajicore@@' not in symbols or '?unpack_f16c@ajicore@@' not in symbols:
    raise SystemExit('FAIL: expected the half_f16c object, not an unrelated object')
print('PASS: AVX object defines no scalar half.h helper symbols')
