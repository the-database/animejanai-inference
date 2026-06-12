#!/usr/bin/env python3
"""Convert the vsmlrt RIFE v1 (video_player) ONNX models to fp16, keeping
the GridSample sampling-grid math in an fp32 island.

The grid lives in normalized [-1, 1] coordinates, where fp16's epsilon
(~1e-3) quantizes sampling positions ~1-2 px coarse: a whole-graph fp16
conversion measured a drop from ~48 dB to ~32 dB against the TensorRT
reference (blacklisting GridSample alone cannot help — the grid is
already fp16 by the time it is cast), while keeping only the convs fp16
drowned the gain in per-conv casts. The island is exactly the ops that
build each grid — Transpose(Add(mesh, Mul(flow, multiplier))) — plus
GridSample itself; flow magnitudes (pixels) and image data are safe in
fp16.

Model IO becomes fp16 (what both aji backends feed). TensorRT's builder
makes the same per-layer precision call implicitly under weak typing;
TRT 11 (strongly typed) follows these ONNX types verbatim, so one model
set serves both backends.

(onnxconverter-common was abandoned: its node_block_list emits
mixed-type bindings at island boundaries in both directions.)

usage: convert_rife_fp16.py SRC_DIR DST_DIR [model.onnx ...]
       (no model list = convert every *.onnx in SRC_DIR)
"""
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

FP16_MAX = 65504.0
GRID_OPS = ("Transpose", "Add", "Mul")   # the grid-builder chain
# float inputs that the spec types independently of T (must stay fp32)
NON_T_FLOAT_INPUTS = {"Resize": {1, 2, 3}}


def island_nodes(graph) -> set[str]:
    prod = {o: n for n in graph.node for o in n.output}
    island: set[str] = set()
    for node in graph.node:
        if node.op_type != "GridSample":
            continue
        island.add(node.name)
        stack = [node.input[1]]
        while stack:
            p = prod.get(stack.pop())
            if p is None or p.name in island or p.op_type not in GRID_OPS:
                continue
            island.add(p.name)
            stack.extend(p.input)
    return island


def tensor_to_fp16(t: TensorProto) -> TensorProto:
    arr = numpy_helper.to_array(t).astype(np.float32)
    arr = np.clip(arr, -FP16_MAX, FP16_MAX).astype(np.float16)
    out = numpy_helper.from_array(arr, t.name)
    return out


def convert_model(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    del g.value_info[:]   # stale entries poison strict EPs

    island = island_nodes(g)
    if not island:
        raise RuntimeError("no GridSample found - not a rife graph?")

    # which tensors stay fp32: outputs of island nodes
    fp32_tensors = {o for n in g.node if n.name in island for o in n.output}

    # initializers: fp16 unless consumed by an island node or at a
    # non-T float position (Resize roi/scales)
    keep_fp32_inits: set[str] = set()
    for n in g.node:
        skip = NON_T_FLOAT_INPUTS.get(n.op_type, set())
        for k, t in enumerate(n.input):
            if n.name in island or k in skip:
                keep_fp32_inits.add(t)
    for i, t in enumerate(g.initializer):
        if t.data_type == TensorProto.FLOAT and t.name not in keep_fp32_inits:
            g.initializer[i].CopyFrom(tensor_to_fp16(t))
    # Constant nodes likewise
    for n in g.node:
        if n.op_type != "Constant" or n.name in island:
            continue
        consumed_fp32 = any(
            (c.name in island and n.output[0] in c.input) or
            n.output[0] in tuple(c.input[k] for k in
                                 NON_T_FLOAT_INPUTS.get(c.op_type, set())
                                 if k < len(c.input))
            for c in g.node)
        if consumed_fp32:
            continue
        for a in n.attribute:
            if a.name == "value" and a.t.data_type == TensorProto.FLOAT:
                a.t.CopyFrom(tensor_to_fp16(a.t))

    # boundary casts (one per tensor, cached)
    prod = {o: n for n in g.node for o in n.output}
    inits = {t.name for t in g.initializer}
    casts: dict[tuple[str, int], str] = {}
    new_nodes = []

    def cast(tensor: str, to: int) -> str:
        key = (tensor, to)
        if key not in casts:
            name = f"{tensor}/cast{16 if to == TensorProto.FLOAT16 else 32}"
            new_nodes.append(helper.make_node("Cast", [tensor], [name],
                                              name=name, to=to))
            casts[key] = name
        return casts[key]

    for n in g.node:
        if n.op_type in ("Constant", "Cast"):
            continue
        skip = NON_T_FLOAT_INPUTS.get(n.op_type, set())
        in_island = n.name in island
        for k, t in enumerate(n.input):
            if not t or k in skip or t in inits:
                continue
            p = prod.get(t)
            src_fp32 = t in fp32_tensors or \
                (p is not None and p.op_type == "Constant" and
                 t in keep_fp32_inits)
            # int tensors (shape math) never flip: their producers are
            # Shape/Gather/Constant-int chains outside both worlds
            if p is not None and p.op_type in ("Shape", "Gather", "Range",
                                               "Squeeze", "Unsqueeze",
                                               "Where", "Equal", "Concat"
                                               ) and k in skip:
                continue
            if in_island and not src_fp32:
                n.input[k] = cast(t, TensorProto.FLOAT)
            elif not in_island and src_fp32:
                n.input[k] = cast(t, TensorProto.FLOAT16)

    g.node.extend(new_nodes)

    # IO: fp16 in and out. The input feeds fp16 consumers directly; the
    # output comes from the fp16 merge math. If a graph output ever
    # comes from the island, give it a cast too.
    g.input[0].type.tensor_type.elem_type = TensorProto.FLOAT16
    out = g.output[0]
    if out.name in fp32_tensors:
        for n in g.node:
            for k, v in enumerate(n.output):
                if v == out.name:
                    n.output[k] = out.name + "_f32"
        g.node.append(helper.make_node("Cast", [out.name + "_f32"],
                                       [out.name], name="aji/cast_output",
                                       to=TensorProto.FLOAT16))
    out.type.tensor_type.elem_type = TensorProto.FLOAT16

    # topo-sort: appended casts must precede their consumers
    order = topo_sort(g)
    del g.node[:]
    g.node.extend(order)
    return model


def topo_sort(g):
    prod = {o: n.name for n in g.node for o in n.output}
    nodes = {n.name: n for n in g.node}
    state: dict[str, int] = {}
    out: list = []

    def visit(name):
        if state.get(name) == 2:
            return
        if state.get(name) == 1:
            raise RuntimeError("cycle at " + name)
        state[name] = 1
        for t in nodes[name].input:
            if t in prod:
                visit(prod[t])
        state[name] = 2
        out.append(nodes[name])

    for n in list(g.node):
        visit(n.name)
    return out


def convert(src: Path, dst: Path) -> tuple[int, int]:
    model = convert_model(onnx.load(str(src)))
    onnx.checker.check_model(model)   # structural check (no shape infer)
    onnx.save(model, str(dst))
    return src.stat().st_size, dst.stat().st_size


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src_dir, dst_dir = Path(sys.argv[1]), Path(sys.argv[2])
    names = sys.argv[3:] or sorted(p.name for p in src_dir.glob("*.onnx"))
    dst_dir.mkdir(parents=True, exist_ok=True)
    total_in = total_out = 0
    for name in names:
        a, b = convert(src_dir / name, dst_dir / name)
        total_in += a
        total_out += b
        print(f"{name}: {a / 1e6:.1f} MB -> {b / 1e6:.1f} MB")
    print(f"total: {total_in / 1e6:.1f} MB -> {total_out / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
