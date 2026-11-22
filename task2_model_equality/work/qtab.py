"""Tabulate quant-scan z-scores from data/qs_*.log. Usage: python qtab.py LOGNAME..."""
import re
import sys

import common

SCH = ["fp8_tensor", "fp8_channel", "int8_channel", "int4_g128", "nf4_b64"]
for name in sys.argv[1:]:
    tab, cur = {}, None
    for line in open(common.DATA / name):
        m = re.match(r"(\S+) (\S+) on", line)
        if m:
            cur = m.group(2)
            continue
        m = re.search(r"((?:ref|ep)-\d+): .*z=([+-][\d.]+)", line)
        if m:
            tab.setdefault(m.group(1), {})[cur] = float(m.group(2))
    print("####", name)
    print("        " + " ".join(f"{s[:12]:>12}" for s in SCH))
    for e in sorted(tab):
        print(f"{e:7} " + " ".join(f"{tab[e].get(s, float('nan')):12.2f}" for s in SCH))
