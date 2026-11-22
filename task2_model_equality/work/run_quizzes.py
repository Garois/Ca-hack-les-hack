"""Spend the last budget on near-deterministic quiz prompts: A = exact replicate of the phase-A quiz (all endpoints),
B = 40-item quiz2 (endpoints that still have budget after A)."""
import common
from quiz2 import QUIZ2
from s1_explore import PROMPTS

QUIZ_ORIG = dict(PROMPTS)["quiz"]
b = common.budget()
print("budget before:", {e: b[e] for e in common.EPS}, flush=True)
common.ask_scored([("quizA", QUIZ_ORIG)], "quizA_ep", [e for e in common.EPS if b[e] >= 1])
b = common.budget()
common.ask_scored(QUIZ2, "quiz2_ep", [e for e in common.EPS if b[e] >= 1])
print("budget after:", {e: v for e, v in common.budget().items() if e.startswith("ep")}, flush=True)
