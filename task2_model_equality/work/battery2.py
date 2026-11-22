"""Second battery: 35 new prompts (new topics, same style) to add tokens for every endpoint."""
from battery import TEMPLATES

TOPICS2 = [
    "the solar system", "how electricity is generated", "the history of the Olympic Games", "earthquakes",
    "the life cycle of a star", "how antibiotics work", "the Renaissance", "ocean currents",
    "the invention of the airplane", "how the heart works", "rainforests", "the history of money",
]
PROMPTS2 = []
for j, tpl in enumerate(TEMPLATES):
    for i, t in enumerate(TOPICS2):
        PROMPTS2.append((f"c{j}_{i:02d}", tpl.format(t=t)))
PROMPTS2 = sorted(PROMPTS2, key=lambda p: (int(p[0][3:]), p[0][1]))[:35]
