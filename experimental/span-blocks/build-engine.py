"""Build one of the measured static engine presets; run without concurrent GPU work."""
import argparse
from pathlib import Path
import subprocess

presets = {
    'balanced-block-4090-1080': ['--optShapes=input:1x3x1080x1920', '--maxAuxStreams=0'],
    'balanced-native7-5090-540': ['--optShapes=input:1x3x540x960', '--tilingOptimizationLevel=3'],
    'performance-cache48-4090-1080': ['--optShapes=input:1x3x1080x1920',
                                    '--tilingOptimizationLevel=3', '--l2LimitForTiling=50331648'],
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--preset', choices=presets, required=True)
parser.add_argument('--trtexec', type=Path, required=True)
parser.add_argument('--onnx', type=Path, required=True)
parser.add_argument('--plugin', type=Path, required=True)
parser.add_argument('--engine', type=Path, required=True)
args = parser.parse_args()
subprocess.run([str(args.trtexec.resolve()), '--onnx='+str(args.onnx.resolve()),
    '--saveEngine='+str(args.engine.resolve()), '--dynamicPlugins='+str(args.plugin.resolve()),
    '--builderOptimizationLevel=5', '--profilingVerbosity=detailed', '--skipInference',
    *presets[args.preset]], check=True)
