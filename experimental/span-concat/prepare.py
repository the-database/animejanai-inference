"""Export a known SPAN model and its original projection weights for the plugin.

Requires onnx and numpy. Does not quantize weights or alter the installed model.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

MODELS = {
    'balanced': (64, 'b308e93c3dda3c4f9f968295b2b7b8d1c0583a711bca53a08d51134d9d1cbd5c',
                 'AJBalancedConcatFP16', '2x_AnimeJaNai_HD_V3.1_Balanced_FP16_Concat_Tiled3_v2'),
    'performance': (48, '7b804d87320f37b7269d6e27796046fbd975e95e2aa1f65d5557f52a36282716',
                    'AJPerformanceConcatFP16', '2x_AnimeJaNai_HD_V3.1_Performance_FP16_Concat_v1'),
}


def prepare(kind, source, output):
    channels, digest, creator, name = MODELS[kind]
    if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
        raise ValueError('Expected the original V3.1 non-sharp FP16 model; SHA-256 differs')
    model = onnx.load(source)
    initializers = [t.SerializeToString() for t in model.graph.initializer]
    nodes = {n.name: n for n in model.graph.node}
    concat, conv = nodes['/Concat_3'], nodes['/conv_cat/Conv']
    assert concat.op_type == 'Concat' and conv.op_type == 'Conv'
    assert len(concat.input) == 4 and list(conv.input[:1]) == list(concat.output)
    weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    w, b = weights[conv.input[1]], weights[conv.input[2]]
    assert w.shape == (channels, 4 * channels, 1, 1) and b.shape == (channels,)
    assert w.dtype == b.dtype == np.float16
    output.mkdir(parents=True, exist_ok=True)
    header = ''.join('static const unsigned short ' + label + '[]={' +
                     ','.join(f'0x{x:04x}' for x in values.reshape(-1).view(np.uint16)) + '};\n'
                     for label, values in [('kWeights', w), ('kBias', b)])
    (output / 'parameters.h').write_text(header)
    plugin = helper.make_node(creator, list(concat.input), list(conv.output),
                              name='/' + creator, domain='com.astra',
                              plugin_version='2', plugin_namespace='')
    new_nodes = [plugin if n.name == conv.name else n for n in model.graph.node
                 if n.name != concat.name]
    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.opset_import.append(helper.make_opsetid('com.astra', 1))
    assert initializers == [t.SerializeToString() for t in model.graph.initializer]
    onnx.checker.check_model(model)
    target = output / (name + '.onnx')
    onnx.save(model, target)
    print(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=MODELS, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.kind, args.model, args.output)
