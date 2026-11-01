"""Export the two validated Balanced block graph variants, without changing weights."""
import argparse
import importlib.util
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper

common = Path(__file__).resolve().parent.parent/'span-concat/prepare.py'
spec = importlib.util.spec_from_file_location('span_concat_prepare', common)
concat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(concat)


def prepare_blocks(kind, source, output):
    if kind == 'performance-cache48':
        original = concat.prepare('performance', source, output)
        target = output/'2x_AnimeJaNai_HD_V3.1_Performance_FP16_Concat_Cache48_v1.onnx'
        shutil.copyfile(original, target)
        return target
    concat_model = concat.prepare('balanced', source, output/'concat-source')
    shutil.copyfile(output/'concat-source/parameters.h', output/'parameters.h')
    model = onnx.load(concat_model if kind == 'block-concat' else source)
    initializers = [t.SerializeToString() for t in model.graph.initializer]
    weights = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    byname = {n.name: n for n in model.graph.node}
    consumers = {}
    for node in model.graph.node:
        for tensor in node.input:
            consumers.setdefault(tensor, []).append(node)
    arrays, biases, flags = [], [], []
    replacement, removed, expose = {}, set(), []
    for block in range(1, 9):
        convs = [byname[f'/block_{block}/c{i}_r/eval_conv/Conv'] for i in range(1, 4)]
        for node in convs:
            w, b = weights[node.input[1]], weights[node.input[2]]
            assert w.dtype == b.dtype == np.float16
            assert w.shape == (64, 64, 3, 3) and b.shape == (64,)
            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            assert attrs['pads'] == [1, 1, 1, 1] and attrs.get('strides', [1, 1]) == [1, 1]
            assert attrs.get('group', 1) == 1 and attrs.get('dilations', [1, 1]) == [1, 1]
            arrays.append('{' + ','.join(f'0x{x:04x}' for x in w.transpose(0, 2, 3, 1).copy().reshape(-1).view(np.uint16)) + '}')
            biases.append('{' + ','.join(f'0x{x:04x}' for x in b.view(np.uint16)) + '}')
            followers = consumers[node.output[0]]
            flags.append(len(followers) == 1 and followers[0].op_type == 'Relu')
        relus = [consumers[n.output[0]][0] for n in convs[:2]]
        assert all(n.op_type == 'Relu' for n in relus)
        assert consumers[relus[1].output[0]] == [convs[2]]
        extra = [n for n in consumers[relus[0].output[0]] if n != convs[1]]
        expose.append(bool(extra))
        if kind == 'native-tail7' and block == 8:
            continue
        if kind == 'native-tail7':
            assert not extra
        outputs = ([relus[0].output[0]] if extra else []) + [convs[2].output[0]]
        replacement[convs[0].name] = helper.make_node(f'AJBalancedBlock3x3_{block-1}',
            [convs[0].input[0]], outputs, name=f'/block_{block}/CUTLASS_3Conv',
            domain='com.astra', plugin_version='1', plugin_namespace='aji.span.blocks.v1')
        removed.update(n.name for n in convs[1:] + relus)
    header = 'static const unsigned short kConvWeights[24][36864]={' + ',\n'.join(arrays) + '};\n'
    header += 'static const unsigned short kConvBias[24][64]={' + ',\n'.join(biases) + '};\n'
    header += 'static const bool kConvRelu[24]={' + ','.join(str(x).lower() for x in flags) + '};\n'
    header += 'static const char* kConvNames[24]={' + ','.join(f'"AJBalancedConv3x3_{i}"' for i in range(24)) + '};\n'
    (output/'conv-parameters.h').write_text(header)
    (output/'block-parameters.h').write_text('static const bool kBlockExpose[8]={' +
        ','.join(str(x).lower() for x in expose) + '};\nstatic const char* kBlockNames[8]={' +
        ','.join(f'"AJBalancedBlock3x3_{i}"' for i in range(8)) + '};\n')
    nodes = [replacement.get(n.name, n) for n in model.graph.node if n.name not in removed]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    for node in model.graph.node:
        if node.op_type == 'AJBalancedConcatFP16':
            for attr in node.attribute:
                if attr.name == 'plugin_namespace':
                    attr.s = b'aji.span.blocks.v1'
    if not any(o.domain == 'com.astra' for o in model.opset_import):
        model.opset_import.append(helper.make_opsetid('com.astra', 1))
    assert initializers == [t.SerializeToString() for t in model.graph.initializer]
    onnx.checker.check_model(model)
    name = 'Concat' if kind == 'block-concat' else 'NativeTail7'
    target = output / f'2x_AnimeJaNai_HD_V3.1_Balanced_FP16_Blocks_{name}_v1.onnx'
    onnx.save(model, target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['block-concat', 'native-tail7', 'performance-cache48'], required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(prepare_blocks(args.kind, args.model, args.output))
