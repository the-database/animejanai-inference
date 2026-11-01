"""Build the RIFE warp plugin from an x64 MSVC developer prompt."""
import argparse
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cuda', type=Path, required=True)
parser.add_argument('--trt-include', type=Path, required=True)
parser.add_argument('--cccl-include', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--allow-unsupported-compiler', action='store_true')
args = parser.parse_args()
if os.name != 'nt':
    raise RuntimeError('The experimental plugin currently supports Windows only')
args.output.mkdir(parents=True, exist_ok=True)
cmd = [str(args.cuda/'bin/nvcc.exe'), '--shared', '-O3', '-std=c++17', '--fmad=false',
       '-gencode', 'arch=compute_89,code=sm_89', '-gencode', 'arch=compute_120,code=sm_120',
       '-I'+str(args.trt_include.resolve()), '-I'+str(args.cccl_include.resolve()),
       '-Xcompiler', '/MT', '-o', str(args.output.resolve()/'rife-warp-plugin.dll'),
       str(Path(__file__).resolve().parent/'plugin.cu')]
if args.allow_unsupported_compiler:
    cmd.append('--allow-unsupported-compiler')
subprocess.run(cmd, check=True, cwd=args.output)
