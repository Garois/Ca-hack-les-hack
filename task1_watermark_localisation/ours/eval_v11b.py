"""W3 fix for PLAN_V11 stage 2: SAM refinement evaluated WITH the rectangle override applied last (as in test), using
the rectangle search run on ALL predicted-partial train images (PixelSeal, TrustMark). Updates v11_choice.json."""
import json
from pathlib import Path
import numpy as np
from eval_v10 import guide_of
from eval_v11 import MODES, family_maps, report, stage2
from train_seg import folds
OUT = Path.home() / "wm_out"
choice = json.loads((OUT / "v11_choice.json").read_text())
for m in ("pixelseal", "trustmark"):
    rects = {}
    for f in sorted(OUT.glob(f"rect_{m}_trainfull*.json")):
        rects.update(json.loads(f.read_text()))
    items, fold_of = folds(m); names = [x["image_name"] for x in items]; G = {n: guide_of(n) for n in names}
    z = np.load(OUT / "sam" / f"masks_{m}_train.npz"); SAM = {k: z[k] for k in z.files}
    print(f"== {m}: rect results for {len(rects)} train images, accepted {sum(r['acc'] >= 0.8 for r in rects.values())}", flush=True)
    names, X, forced, auto, chosen = stage2(m, family_maps(m, ["fx", "dfx"]), items, fold_of, G, True, SAM, rects=rects)
    ref = forced[("none",)]
    for md in MODES[1:]:
        report(f"SAM forced {md} +rect", forced[md], ref, names, X)
    lo, mean = report("SAM auto +rect (nested)", auto, ref, names, X)
    print("   chosen partial modes per fold:", [str(chosen[(f, False)][1]) for f in range(5)], flush=True)
    choice[m]["sam"] = bool(lo > 0); print(f"   -> {choice[m]}", flush=True)
(OUT / "v11_choice.json").write_text(json.dumps(choice, indent=1))
