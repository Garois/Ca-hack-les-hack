"""Candidate A (decision rules applied to data v1/v15, before the creative battery is analysed).
base = plain endpoints at T~0.70 with no projection on the family LoRA direction + decoding disguises (T=1.0, top-p);
fp8 detected by the centred fp8 score (z > 9); every other deviating endpoint is its own model, except the two weak
LoRA endpoints of Qwen-1.5B whose estimated strengths agree (alpha ~0.2 both)."""
from write_sub import write

E = lambda *x: [f"ep-{i:02d}" for i in x]
A = {
    "Q15": [E(6, 7, 30, 32), E(21, 26), E(22), E(28), E(3), E(8), E(20)],
    "L1": [E(10, 13, 27, 31), E(15), E(5), E(11), E(9), E(19), E(29), E(16)],
    "Q3": [E(4, 12, 17, 25, 14, 33), E(1), E(18), E(24), E(2), E(23)],
}
write(A, "subs/candA.csv")
print(open("subs/candA.csv").read())
