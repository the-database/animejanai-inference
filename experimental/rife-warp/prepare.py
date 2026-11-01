"""Replace the four cast/GridSample/cast sequences in the tested RIFE 4.22 export."""
import argparse
import hashlib
from pathlib import Path

import onnx
from onnx import helper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--model', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True, help='directory for the distinct experimental ONNX')
args = parser.parse_args()
digest = '7fdb67dcf02d23cd573b5269bb41c600b7d4241cfda366099e736938d8f7abc6'
if hashlib.sha256(args.model.read_bytes()).hexdigest() != digest:
    raise ValueError('Expected the tested non-ensemble RIFE 4.22 FP16 model; SHA-256 differs')
model = onnx.load(args.model)
weights = [t.SerializeToString() for t in model.graph.initializer]
producer = {o: n for n in model.graph.node for o in n.output}
consumer = {}
for node in model.graph.node:
    for tensor in node.input:
        consumer.setdefault(tensor, []).append(node)
remove, count = [], 0
for node in model.graph.node:
    if node.op_type != 'GridSample':
        continue
    assert {a.name: helper.get_attribute_value(a) for a in node.attribute} == {
        'align_corners': 1, 'mode': b'bilinear', 'padding_mode': b'border'}
    before, after = producer[node.input[0]], consumer[node.output[0]]
    assert before.op_type == 'Cast' and before.attribute[0].i == onnx.TensorProto.FLOAT
    assert len(after) == 1 and after[0].op_type == 'Cast'
    assert after[0].attribute[0].i == onnx.TensorProto.FLOAT16
    node.input[0], node.output[0] = before.input[0], after[0].output[0]
    node.op_type, node.domain = 'AJRifeWarp', 'com.astra'
    del node.attribute[:]
    node.attribute.extend([helper.make_attribute('plugin_version', '1'),
                           helper.make_attribute('plugin_namespace', '')])
    remove.append(after[0])
    count += 1
for node in remove:
    model.graph.node.remove(node)
needed = {i for node in model.graph.node for i in node.input}
for node in list(model.graph.node):
    if node.op_type == 'Cast' and node.output[0] not in needed:
        model.graph.node.remove(node)
del model.graph.value_info[:]
model.opset_import.append(helper.make_opsetid('com.astra', 1))
assert count == 4 and weights == [t.SerializeToString() for t in model.graph.initializer]
onnx.checker.check_model(model)
args.output.mkdir(parents=True, exist_ok=True)
target = args.output / 'rife_v4.22_fp16_warp_v1.onnx'
onnx.save(model, target)
print(target)
