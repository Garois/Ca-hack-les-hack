#!/usr/bin/env python3
"""Sondes de regime SANS entrainement : orig + chirurgie du biais de la tete sur les classes
presumees entierement retirees (classes dont le forget set contient >= --cap images).

head.bias[c] -= D decale exactement le logit de la classe c de -D pour toute entree : les stats
sur le forget set se calculent donc sur les logits en cache (z_clean de orig_stats.npz, calcules
par lemasnedechermont1 en fp16 comme le serveur ; fichier lu, jamais modifie).
D est resolu pour que la CE mediane (images propres) des images forget de ces classes vaille --ce.
C'est la meme construction que R1 (head_rm.py) de lemasnedechermont1, mais autonome et CPU-only.

CPU seulement (~10-20 s). Depuis unlearning/vaudois1 :
    ../.venv/bin/python -I scripts/make_surgery_probe.py --ce 5 10
Ecrit probes/surg_ce<ce>.pt (+ .json : D, classes, stats, sha256, chargement strict OK).
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]  # unlearning/vaudois1
TASK = ROOT.parent  # unlearning/
STATS = TASK / "lemasnedechermont1" / "artifacts" / "orig_stats.npz"
MODEL = TASK / "model.pt"


def ce_rows(z, y):
    m = z.max(1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(z - m).sum(1))
    return lse - z[np.arange(len(z)), y]


def ms(t):
    return [round(float(t.mean()), 3), round(float(t.std()), 3), round(float(np.median(t)), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce", type=float, nargs="+", default=[5.0, 10.0])
    ap.add_argument("--cap", type=int, default=129, help="classe 'retiree' si count >= cap")
    ap.add_argument("--out", default=str(ROOT / "probes"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    S = np.load(STATS)
    y = S["y"].astype(np.int64)
    z = S["z_clean"].astype(np.float64)
    counts = S["counts"]
    cls = np.where(counts >= a.cap)[0]
    rm = counts[y] >= a.cap
    forget_cls = counts > 0
    shift = np.zeros(z.shape[1])
    shift[cls] = 1.0
    print(f"classes presumees retirees {cls.tolist()} ; {int(rm.sum())} images forget sur {len(y)}", flush=True)

    base = torch.load(MODEL, map_location="cpu", weights_only=True)
    try:
        import timm
        net = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)
    except Exception as e:  # timm absent : on saute seulement le test de chargement strict
        net = None
        print("timm indisponible, pas de test strict:", e)

    for t in a.ce:
        lo, hi = 0.0, 80.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if np.median(ce_rows(z[rm] - mid * shift, y[rm])) < t:
                lo = mid
            else:
                hi = mid
        D = (lo + hi) / 2
        zs = z - D * shift
        ce = ce_rows(zs, y)
        pred = zs.argmax(1)

        sd = {k: v.clone() for k, v in base.items()}
        sd["head.bias"][torch.as_tensor(cls)] -= D
        bad = [k for k, v in sd.items() if v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all())]
        assert not bad, bad
        strict = None
        if net is not None:
            net.load_state_dict(sd, strict=True)
            strict = True
        path = out / f"surg_ce{t:g}.pt"
        torch.save(sd, path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        new = Counter(pred[rm].tolist()).most_common(8)
        info = {
            "file": str(path), "sha256": sha, "target_median_ce_removed": t, "D": round(D, 4),
            "removed_classes": cls.tolist(), "n_removed_images": int(rm.sum()),
            "forget_clean_ce_all_mean_std_med": ms(ce),
            "forget_clean_ce_removed_mean_std_med": ms(ce[rm]),
            "forget_clean_ce_kept_mean_std_med": ms(ce[~rm]),
            "acc_removed": round(float((pred[rm] == y[rm]).mean()), 4),
            "acc_kept": round(float((pred[~rm] == y[~rm]).mean()), 4),
            "removed_now_predicted_as_forget_class": round(float(forget_cls[pred[rm]].mean()), 4),
            "top_new_classes_for_removed": [[int(c), int(n), bool(forget_cls[c])] for c, n in new],
            "strict_load_ok": strict,
            "note": "z_clean = logits orig en cache (fp16). Pas de mesure sur ancres/vues alterees ici (GPU requis).",
        }
        (out / f"surg_ce{t:g}.json").write_text(json.dumps(info, indent=2))
        print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
