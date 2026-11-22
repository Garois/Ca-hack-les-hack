"""Build a StegaStamp ONNX that also outputs the decoder's intermediate activations, and extract the dense weights
needed for the per-position message agreement. Writes ~/wm_out/stega/stega_feats.onnx and stega_dense.npz, then checks
output shapes with one CPU run."""
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from common import ROOT

OUTD = Path.home() / "wm_out" / "stega"; OUTD.mkdir(parents=True, exist_ok=True)
FEATS = ["gen_decoder/dec_conv_4/Relu:0", "gen_decoder/dec_conv_5/Relu:0", "gen_decoder/dec_conv_6/Relu:0",
         "gen_decoder/dec_conv_7/Relu:0", "gen_decoder/dec_dense/BiasAdd:0", "gen_decoder/decoder_out/BiasAdd:0"]

m = onnx.load(str(ROOT / "models" / "stegastamp" / "stega_stamp.onnx"))
g = m.graph
inits = {i.name: i for i in g.initializer}
byname = {n.name: n for n in g.node}
producer = {o: n for n in g.node for o in n.output}
# dense weights / biases
d1, d1b = byname["gen_decoder/dec_dense/MatMul"], byname["gen_decoder/dec_dense/BiasAdd"]
d2, d2b = byname["gen_decoder/decoder_out/MatMul"], byname["gen_decoder/decoder_out/BiasAdd"]
get = lambda name: numpy_helper.to_array(inits[name]) if name in inits else None
W1 = get(d1.input[1]); W2 = get(d2.input[1])
b1 = next(get(i) for i in d1b.input if get(i) is not None); b2 = next(get(i) for i in d2b.input if get(i) is not None)
print("W1", W1.shape, "b1", b1.shape, "W2", W2.shape, "b2", b2.shape)
# flatten order check: Flatten input should be the NHWC transpose of dec_conv_7/Relu
fl = producer[d1.input[0]]; tr = producer[fl.input[0]]
print("flatten node:", fl.op_type, fl.name, "<-", tr.op_type, tr.name, [a for a in tr.attribute], "<-", tr.input)
np.savez(OUTD / "stega_dense.npz", W1=W1, b1=b1, W2=W2, b2=b2)
# add outputs
existing = {o.name for o in g.output}
for name in FEATS:
    if name not in existing:
        g.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
onnx.save(m, str(OUTD / "stega_feats.onnx"))
so = ort.SessionOptions(); so.intra_op_num_threads = 8
sess = ort.InferenceSession(str(OUTD / "stega_feats.onnx"), so, providers=["CPUExecutionProvider"])
outs = sess.run(FEATS, {"image": np.random.rand(1, 400, 400, 3).astype(np.float32), "secret": np.zeros((1, 100), np.float32)})
for n, o in zip(FEATS, outs):
    print(f"{n:45s} {o.shape}")
# consistency: logits == W2 relu(W1 flat + b1) + b2 with NHWC flatten of dec_conv_7
c7 = outs[3]; flat = np.transpose(c7, (0, 2, 3, 1)).reshape(1, -1)
pre = flat @ W1 + b1; lg = np.maximum(pre, 0) @ W2 + b2
print("dense pre-act max abs diff:", float(np.abs(pre - outs[4]).max()), " logits max abs diff:", float(np.abs(lg - outs[5]).max()))
