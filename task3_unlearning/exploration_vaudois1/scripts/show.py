#!/usr/bin/env python3
"""Tableau compact des runs : python3 scripts/show.py [motif...]  (lit runs/*/metrics.json et eval.json)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
pats = sys.argv[1:] or [""]
for f in sorted(ROOT.glob("runs/*/metrics.json")) + sorted(ROOT.glob("runs/*/eval.json")):
    run = f.parent.name
    if not any(p in run for p in pats):
        continue
    for k, r in json.loads(f.read_text()).items():
        name = (Path(k).parent.name + "/" + Path(k).name) if f.name == "eval.json" else f"a={k}"
        c, an = r["clean"], r["anchor"]
        km = r.get("keep_medium", {})
        print(f"{run:>16} {name:<34} clean {c['ce'][0]:.2f}/{c['ce'][1]:.2f} acc {c['acc']:.2f} "
              f"held {r['heldout']['ce'][0]:.2f} jpeg {r['jpeg']['ce'][0]:.2f} KLtgt {c.get('kl_target', float('nan')):.2f} "
              f"anc {an['kl']:.3f}/{an['agree']:.2f} keepKL {km.get('kl_orig', float('nan')):.3f} "
              f"hoKeepKL {km.get('ho_kl_orig', float('nan')):.3f} hoKL {c.get('ho_kl_orig', float('nan')):.3f}")
