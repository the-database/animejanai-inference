"""Build the optional fixed-weight plugin from an x64 MSVC developer prompt.

Requires Python with onnx/numpy, CUDA 13.x and TensorRT 11 headers.
Outputs model + plugin; does not modify an installed AnimeJaNai directory.
"""
import argparse
import os
from pathlib import Path
import subprocess

from prepare import MODELS, prepare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--kind', choices=MODELS, required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--cuda', type=Path, required=True)
parser.add_argument('--trt-include', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--allow-unsupported-compiler', action='store_true')
args = parser.parse_args()
if os.name != 'nt':
    raise RuntimeError('This experimental plugin currently supports Windows only')
source = Path(__file__).resolve().parent / args.kind
output = args.output.resolve()
prepare(args.kind, args.model, output)
nvcc = [str(args.cuda / 'bin/nvcc.exe'), '-O3', '-fatbin',
        '-gencode', 'arch=compute_89,code=sm_89',
        '-gencode', 'arch=compute_120,code=sm_120']
if args.allow_unsupported_compiler:
    nvcc.append('--allow-unsupported-compiler')
subprocess.run([*nvcc, str(source / 'kernels.cu'), '-o', str(output / 'kernels.fatbin')], check=True)
data = (output / 'kernels.fatbin').read_bytes()
(output / 'kernel_ptx.h').write_text('alignas(8) static const unsigned char kPtx[] = {\n' +
    '\n'.join(','.join(str(b) for b in data[i:i+32]) + ',' for i in range(0, len(data), 32)) + '\n};\n')
name = 'balanced-next-plugin' if args.kind == 'balanced' else 'performance-plugin'
subprocess.run(['cl.exe', '/nologo', '/LD', '/O2', '/MT', '/std:c++17', '/EHsc',
                '/I' + str(args.trt_include.resolve()), '/I' + str(args.cuda / 'include'),
                '/I' + str(output), '/Fo' + str(output / (name + '.obj')),
                str(source / 'plugin.cpp'), '/link',
                '/LIBPATH:' + str(args.cuda / 'lib/x64'), 'cuda.lib',
                '/OUT:' + str(output / (name + '.dll'))], check=True, cwd=output)
