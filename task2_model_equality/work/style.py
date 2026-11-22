"""Surface style markers per endpoint (counts over all responses).
Usage: python style.py DATA [DATA ...] [--fam]"""
import re
import sys

import pandas as pd

import common

args = sys.argv[1:]
use_fam = "--fam" in args
args = [a for a in args if a != "--fam"]
rows = [r for d in args for r in common.load(d)]
fam = {}
if use_fam:
    from fams import EP2FAM
    fam = EP2FAM

agg = {}
for r in rows:
    t = r["response"]
    a = agg.setdefault(r["endpoint"], {"n": 0, "chars": 0, "curly'": 0, "straight'": 0, "curly\"": 0, "straight\"": 0,
                                       "bold": 0, "numlist": 0, "bullets": 0, "heading": 0, "para": 0, "heres": 0,
                                       "emdash": 0, "endash": 0, "ellipsis": 0, "nonascii": 0, "excl": 0, "colon_end": 0})
    a["n"] += 1
    a["chars"] += len(t)
    a["curly'"] += t.count("’") + t.count("‘")
    a["straight'"] += t.count("'")
    a["curly\""] += t.count("“") + t.count("”")
    a["straight\""] += t.count('"')
    a["bold"] += t.count("**")
    a["numlist"] += len(re.findall(r"(?m)^\s*\d+\.", t))
    a["bullets"] += len(re.findall(r"(?m)^\s*[-*•] ", t))
    a["heading"] += len(re.findall(r"(?m)^#+ ", t))
    a["para"] += t.count("\n\n")
    a["heres"] += len(re.findall(r"Here['’]s|Here is", t))
    a["emdash"] += t.count("—")
    a["endash"] += t.count("–")
    a["ellipsis"] += t.count("...") + t.count("…")
    a["nonascii"] += sum(ord(c) > 127 for c in t)
    a["excl"] += t.count("!")
df = pd.DataFrame(agg).T
out = pd.DataFrame(index=df.index)
out["fam"] = [fam.get(e, "") for e in df.index]
out["n"] = df["n"]
out["len"] = (df["chars"] / df["n"]).round(0)
out["curly'%"] = (100 * df["curly'"] / (df["curly'"] + df["straight'"]).clip(lower=1)).round(1)
out["curly\"%"] = (100 * df["curly\""] / (df["curly\""] + df["straight\""]).clip(lower=1)).round(1)
for k in ["bold", "numlist", "bullets", "heading", "para", "heres", "emdash", "endash", "nonascii", "excl"]:
    out[k] = (df[k] / df["n"]).round(2)
out = out.sort_values(["fam"]) if use_fam else out.sort_index()
pd.set_option("display.width", 250)
print(out.to_string())
