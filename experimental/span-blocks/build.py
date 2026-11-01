"""Build the experimental CUTLASS block plugin from an x64 MSVC prompt."""
import argparse
import os
from pathlib import Path
import subprocess

from prepare import prepare_blocks

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--kind', choices=['block-concat', 'native-tail7'], required=True)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--cuda', type=Path, required=True)
parser.add_argument('--trt-include', type=Path, required=True)
parser.add_argument('--cutlass-include', type=Path, required=True)
parser.add_argument('--cccl-include', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--allow-unsupported-compiler', action='store_true')
args = parser.parse_args()
if os.name != 'nt':
    raise RuntimeError('The experimental plugin currently supports Windows only')
output = args.output.resolve()
source = Path(__file__).resolve().parent
prepare_blocks(args.kind, args.model, output)
nvcc = [str(args.cuda/'bin/nvcc.exe'), '-O3', '-gencode', 'arch=compute_89,code=sm_89',
        '-gencode', 'arch=compute_120,code=sm_120']
if args.allow_unsupported_compiler:
    nvcc.append('--allow-unsupported-compiler')
subprocess.run([*nvcc, '-fatbin', str(source.parent/'span-concat/balanced/kernels.cu'),
                '-o', str(output/'kernels.fatbin')], check=True)
data = (output/'kernels.fatbin').read_bytes()
(output/'kernel_ptx.h').write_text('alignas(8) static const unsigned char kPtx[] = {\n' +
    '\n'.join(','.join(str(b) for b in data[i:i+32])+',' for i in range(0,len(data),32)) + '\n};\n')
cmd = [*nvcc, '--shared', '--cudart', 'shared', '-std=c++17', '-DAJ_CONV_720_M256', '-Xcompiler', '/Zc:preprocessor',
       '-I'+str(args.trt_include.resolve()), '-I'+str(args.cutlass_include.resolve()),
       '-I'+str(args.cccl_include.resolve()), '-I'+str(args.cccl_include.resolve()/'cccl'),
       '-I'+str(output), str(source/'plugin.cu'), '-L'+str(args.cuda/'lib/x64'), '-lcuda',
       '-o', str(output/'balanced-blocks-plugin.dll')]
subprocess.run(cmd, check=True, cwd=output)
