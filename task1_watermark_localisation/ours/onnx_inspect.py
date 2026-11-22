"""List the StegaStamp decoder graph: conv/relu outputs (names, ops) and what feeds the decoder."""
import onnx
from common import ROOT
m = onnx.load(str(ROOT / "models" / "stegastamp" / "stega_stamp_logits.onnx"), load_external_data=False)
g = m.graph
print("inputs:", [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in g.input])
print("outputs:", [o.name for o in g.output])
dec = [n for n in g.node if "decoder" in n.name or any("decoder" in o for o in n.output)]
print("decoder nodes:", len(dec))
for n in dec:
    if n.op_type in ("Conv", "Relu", "MatMul", "Gemm", "Transpose", "Reshape", "GridSample", "Flatten") or "stn" in n.name.lower():
        print(f"{n.op_type:10s} {n.name[:70]:70s} in={[i[:50] for i in n.input[:2]]} out={[o[:60] for o in n.output]}")
# what is the first decoder op fed with?
first = dec[0]
print("first decoder node:", first.op_type, first.name, "inputs:", list(first.input))
