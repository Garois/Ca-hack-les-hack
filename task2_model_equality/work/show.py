"""Print answers of a data file grouped by prompt id then endpoint. Usage: python show.py NAME [id ...] [--chars N]"""
import sys
from collections import defaultdict

import common

args = sys.argv[1:]
name = args.pop(0)
n = 300
if "--chars" in args:
    i = args.index("--chars"); n = int(args[i + 1]); del args[i:i + 2]
rows = common.load(name)
by = defaultdict(lambda: defaultdict(list))
for r in rows:
    by[r["id"]][r["endpoint"]].append(r["response"])
for pid in by:
    if args and pid not in args:
        continue
    print(f"\n######## {pid}")
    for ep in sorted(by[pid]):
        for i, resp in enumerate(by[pid][ep]):
            print(f"[{ep} #{i}] {resp[:n]!r}")
