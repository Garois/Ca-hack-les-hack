#!/usr/bin/env python3
"""Soumission d'un state dict a la tache 31-unlearning + journal automatique dans SCORES.md / scores.jsonl.

Valide d'abord le fichier (memes cles/formes que model.pt, float32, fini, <= 450 Mo, chargement strict timm),
lit la cle API dans ../submission.py (jamais affichee), envoie, affiche la reponse du serveur.
Depuis unlearning/vaudois1 (login node, internet) :
    ../.venv/bin/python -I scripts/submit.py runs/L1/alpha_1.pt --note "mlp8-11 wa3 alpha1"
"""
import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import requests
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT.parent
SERVER = "http://34.147.229.55"
TASK_ID = "31-unlearning"


def validate(path):
    errs = []
    size = path.stat().st_size / 2 ** 20
    if size > 450:
        errs.append(f"taille {size:.0f} Mo")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    ref = torch.load(TASK / "model.pt", map_location="cpu", weights_only=True)
    if set(sd) != set(ref):
        errs.append("cles differentes")
    for k, v in sd.items():
        if k in ref and v.shape != ref[k].shape:
            errs.append(f"forme {k}")
        if v.is_floating_point() and (v.dtype != torch.float32 or not torch.isfinite(v).all()):
            errs.append(f"dtype/fini {k}")
    if not errs:
        import timm
        m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=100, init_values=1e-4)
        m.load_state_dict(sd, strict=True)
    changed = sum(int(not torch.equal(sd[k], ref[k])) for k in ref if k in sd)
    return errs, changed, len(ref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--note", default="")
    ap.add_argument("--who", default="vaudois1")
    ap.add_argument("--dry", action="store_true", help="valide seulement")
    a = ap.parse_args()
    path = Path(a.path).resolve()
    errs, changed, n = validate(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{path.name}: {changed}/{n} tenseurs modifies, sha256 {sha[:12]}, erreurs {errs or 'aucune'}", flush=True)
    if errs or a.dry:
        return 1 if errs else 0
    m = re.search(r'^API_KEY\s*=\s*"([^"]+)"', (TASK / "submission.py").read_text(), re.M)
    if not m or m.group(1) in ("YOUR_API_KEY", "YOUR_API_KEY_HERE"):
        print("cle API introuvable dans ../submission.py", file=sys.stderr)
        return 1
    import time
    for attempt in range(4):  # cooldown equipe (~40 s) : attendre le delai indique puis relancer
        with path.open("rb") as f:
            rep = requests.post(f"{SERVER}/submit/{TASK_ID}", headers={"X-API-Key": m.group(1)},
                                files={"file": (path.name, f)}, timeout=(10, 600))
        if rep.status_code != 429 or attempt == 3:
            break
        try:
            detail = str(rep.json().get("detail", ""))
        except ValueError:
            detail = rep.text
        w = re.search(r"Wait (\d+) seconds", detail)
        wait = int(w.group(1)) + 5 if w else 45
        print(f"429 cooldown : {detail[:120]} -> nouvel essai dans {wait} s", flush=True)
        time.sleep(wait)
    try:
        body = rep.json()
    except ValueError:
        body = {"status": f"HTTP {rep.status_code}", "message": rep.text[:300]}
    if rep.status_code != 200:
        body.setdefault("status", f"HTTP {rep.status_code}")
        body.setdefault("message", str(body.get("detail", rep.text[:300])))
        ra = rep.headers.get("Retry-After")
        if ra:
            body["message"] = f"{body[message]} (Retry-After {ra}s)"
    score = body.get("score")
    print(json.dumps({k: v for k, v in body.items() if "key" not in k.lower()}, indent=1), flush=True)
    now = dt.datetime.now().strftime("%H:%M")
    rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
    with open(ROOT / "scores.jsonl", "a") as f:
        f.write(json.dumps({"time": now, "file": str(rel), "sha256": sha, "note": a.note, "http": rep.status_code,
                            "status": body.get("status"), "score": score, "message": body.get("message"),
                            "previous_best": body.get("previous_best")}) + "\n")
    with open(ROOT / "SCORES.md", "a") as f:
        sc = f"**{score:.4f}**" if isinstance(score, (int, float)) else f"{body.get('status')}: {str(body.get('message'))[:80]}"
        f.write(f"| {now} | {a.who} | `{rel}` | {sha[:8]} | {a.note} | {sc} | |\n")
    return 0 if rep.status_code == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
