import sys, collections, onnx
from onnx import shape_inference, numpy_helper
m = onnx.load(sys.argv[1]); g = m.graph
print("inputs", [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in g.input])
print("ops", collections.Counter(n.op_type for n in g.node).most_common())
init = {i.name: i for i in g.initializer}
# conv shapes; relative spatial scale: track by walking -- approximate via upsample/unshuffle markers
convs = collections.Counter()
for n in g.node:
    if n.op_type == "Conv":
        w = init[n.input[1]]; co, ci, kh, kw = w.dims
        convs[(ci, co, kh)] += 1
for k, v in sorted(convs.items()): print("conv cin %d cout %d k%d x%d" % (k[0], k[1], k[2], v))
for n in g.node:
    if n.op_type in ("Resize", "DepthToSpace", "SpaceToDepth", "Reshape", "Transpose", "Upsample"):
        print(n.op_type, [ (a.name, onnx.helper.get_attribute_value(a)) for a in n.attribute][:3])
print("dtype", g.initializer[0].data_type)
