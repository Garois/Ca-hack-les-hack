"""How many predicted-partial test images per method have a rectangle-search result (union of all result files)."""
import json
from common import CACHE, METHODS
from predict import RECT_THR, load_rects
for m in METHODS:
    names = json.loads((CACHE / f"partial_test_{m}.json").read_text()); r = load_rects(m, "test")
    done = [n for n in names if n in r]
    print(m, f"{len(done)}/{len(names)} searched, {sum(r[n]['acc'] >= RECT_THR[m] for n in done)} rectangles accepted (thr {RECT_THR[m]})")
