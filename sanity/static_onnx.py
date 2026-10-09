"""Regression for dynamic ONNX models used with the Static ONNX preset.

Requires an NVIDIA GPU, a built aji_harness with its runtime DLLs/libraries,
trtexec, and the Python onnx package. Creates tiny Identity models and runs
one frame through each valid configuration; no anime model is required.

python sanity/static_onnx.py --harness /path/to/aji_harness \
    --trtexec /path/to/trtexec --work-dir /path/to/empty/test-directory
"""

import argparse
from pathlib import Path
import subprocess

import onnx
from onnx import TensorProto, helper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--trtexec", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.work_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    raw = root / "frame.nv12"
    raw.write_bytes(bytes([128]) * (16 * 16 * 3 // 2))

    for name, shape in [("dynamic", [1, 3, "H", "W"]),
                        ("fixed", [1, 3, 16, 16])]:
        graph = helper.make_graph(
            [helper.make_node("Identity", ["input"], ["output"])], name,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT16, shape)],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, shape)])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)],
                                  ir_version=10)
        onnx.checker.check_model(model)
        onnx.save(model, root / f"{name}.onnx")

    cases = [
        ("dynamic", "", False),
        ("fixed", "", True),
        ("dynamic", "--optShapes=input:%video_resolution%", True),
        ("dynamic", "--minShapes=input:1x3x8x8 --optShapes=input:1x3x16x16 "
         "--maxShapes=input:1x3x32x32", True),
    ]
    for i, (name, shapes, valid) in enumerate(cases):
        conf = root / f"case-{i}.conf"
        conf.write_text(
            "[global]\nconfig_version=2\nbackend=TensorRT\n"
            f"trt_engine_settings=--builderOptimizationLevel=0 --skipInference {shapes}\n"
            f"[slot_1]\nchain_1_model_1_name={name}\n", encoding="utf-8")
        proc = subprocess.run(
            [str(args.harness.resolve()), "--conf", str(conf), "--model-dir", str(root),
             "--trtexec", str(args.trtexec.resolve()), "--input", str(raw),
             "--width", "16", "--height", "16", "--frames", "1"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            errors="replace", timeout=180)
        (root / f"case-{i}.log").write_text(proc.stdout, encoding="utf-8")
        if valid:
            assert proc.returncode == 0, proc.stdout
        else:
            assert proc.returncode != 0, proc.stdout
            assert "Static ONNX requires a model with fixed input dimensions" in proc.stdout
            assert "select Static or Dynamic" in proc.stdout
            assert not list(root.glob("*.engine*")), "Bad preset started an engine build"
        print(f"PASS: {name}, {shapes or 'Static ONNX'}")


if __name__ == "__main__":
    main()
