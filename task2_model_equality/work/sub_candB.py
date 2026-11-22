"""Candidate B: prelim minus the weak-LoRA endpoints identified by the cross-fitted LoRA-direction projection
(Q15: 21, 26; L1: 19; Q3: 01), all as singletons (ladder argument: distinct LoRA strengths).
Candidate B0 (conservative): prelim minus 21 and 26 only."""
from write_sub import write

E = lambda *x: [f"ep-{i:02d}" for i in x]
B = {
    "Q15": [E(6, 7, 8, 20, 30, 32), E(22), E(28), E(3), E(21), E(26)],
    "L1": [E(10, 13, 16, 27, 29, 31), E(15), E(5), E(11), E(9), E(19)],
    "Q3": [E(4, 12, 14, 17, 18, 23, 24, 25, 33), E(1), E(2)],
}
B0 = {
    "Q15": [E(6, 7, 8, 20, 30, 32), E(22), E(28), E(3), E(21), E(26)],
    "L1": [E(10, 13, 16, 19, 27, 29, 31), E(15), E(5), E(11), E(9)],
    "Q3": [E(1, 4, 12, 14, 17, 18, 23, 24, 25, 33), E(2)],
}
write(B, "subs/candB.csv")
write(B0, "subs/candB0.csv")
for f in ("subs/candB.csv", "subs/candB0.csv"):
    print("==", f); print(open(f).read())
