#!/usr/bin/env python3
"""Affiche seulement les NOUVEAUX evenements (etat dans .watch_state.json) : etat des jobs vau_*, lignes du runner,
erreurs dans logs/*.log, resultats compacts de runs/*/metrics.json et eval.json."""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".watch_state.json"
st = json.loads(STATE.read_text()) if STATE.exists() else {"seen": [], "jobs": {}, "pos": {}}
seen = set(st["seen"])
out = []

try:
    q = subprocess.run(["squeue", "-u", "vaudois1", "-h", "-o", "%i|%j|%T|%M|%R"], capture_output=True, text=True, timeout=20).stdout
except Exception:
    q = ""
cur = {}
for line in q.strip().splitlines():
    i, name, state, tm, reason = line.split("|")
    if name.startswith("vau_"):
        cur[i] = state
        if st["jobs"].get(i) != state:
            out.append(f"JOB {i} {name}: {state} ({reason})")
for i, old in st["jobs"].items():
    if i not in cur and old != "GONE":
        out.append(f"JOB {i}: termine (n'est plus dans squeue)")
        cur[i] = "GONE"
st["jobs"] = cur

for f in sorted((ROOT / "logs").glob("*")):
    if not f.is_file():
        continue
    pos = st["pos"].get(f.name, 0)
    with open(f, "rb") as fh:
        fh.seek(pos)
        data = fh.read().decode(errors="replace")
        st["pos"][f.name] = fh.tell()
    for line in data.splitlines():
        if f.name.startswith("vau_runner"):
            if re.search(r"start |stop|STOP|idle|runner (start|end)|RUNNING", line):
                out.append(f"RUNNER {line.strip()}")
        elif re.search(r"Traceback|Error|error:|Killed|out of memory|CUDA error|nan", line) and "Warning" not in line:
            out.append(f"ERR {f.name}: {line.strip()[:200]}")
        elif line.startswith("done "):
            out.append(f"DONE {f.name}: {line.strip()}")


def row(r):
    c, mi, he, jp, an = r["clean"], r["mild"], r["heldout"], r["jpeg"], r["anchor"]
    s = (f"clean {c['ce'][0]:.2f}/{c['ce'][1]:.2f} acc{c['acc']:.2f} | mild {mi['ce'][0]:.2f}/{mi['ce'][1]:.2f} | "
         f"held {he['ce'][0]:.2f}/{he['ce'][1]:.2f} | jpeg {jp['ce'][0]:.2f}/{jp['ce'][1]:.2f} | "
         f"anc KL {an['kl']:.3f} agr {an['agree']:.3f} | ret_est {r['retain_est']:.3f}")
    if "ceA" in c:
        s += f" | 9 classes CE {c['ceA'][0]:.2f} / autres {c['ceB'][0]:.2f} (held 9cl {r['heldout']['ceA'][0]:.2f})"
    if "kl_target" in c:
        s += f" | KLtgt {c['kl_target']:.2f}"
    if "keep_medium" in r:
        km = r["keep_medium"]
        s += f" | keepKL {km['kl_orig']:.3f}" + (f" hoKeepKL {km['ho_kl_orig']:.3f}" if "ho_kl_orig" in km else "")
    if "ho_kl_orig" in c:
        s += f" | holdout KL {c['ho_kl_orig']:.3f}"
    return s


for f in sorted((ROOT / "runs").glob("*/metrics.json")) + sorted((ROOT / "runs").glob("*/eval.json")):
    key = str(f.relative_to(ROOT))
    if key in seen:
        continue
    try:
        m = json.loads(f.read_text())
    except Exception:
        continue
    seen.add(key)
    for k, r in m.items():
        name = Path(k).name if f.name == "eval.json" else f"a={k}"
        out.append(f"RES {f.parent.name} {name}: {row(r)}")

st["seen"] = sorted(seen)
STATE.write_text(json.dumps(st))
print("\n".join(out))
