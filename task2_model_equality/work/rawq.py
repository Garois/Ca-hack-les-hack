"""Send one request to reference endpoints and print the full JSON reply (minus the text) and the latency.
Usage: python rawq.py ref-01,ref-02 ["prompt"]"""
import json
import sys
import time

import common

common.setup()
import task_template as tt

s = tt.make_session()
eps = sys.argv[1].split(",")
prompt = sys.argv[2] if len(sys.argv) > 2 else "Write one sentence about the sea."
assert all(e.startswith("ref-") for e in eps)
for e in eps:
    while True:
        t = time.time()
        r = s.post(f"{tt.BASE_URL}/chat/completions", timeout=180,
                   json={"model": e, "max_tokens": 64, "messages": [{"role": "user", "content": prompt}]})
        dt = time.time() - t
        if r.status_code == 429 and "retry-after" in {k.lower() for k in r.headers}:
            time.sleep(1); continue
        break
    body = r.json()
    for c in body.get("choices", []):
        c.get("message", {})["content"] = c.get("message", {}).get("content", "")[:80]
    print(e, f"{dt:.2f}s", r.status_code, {k: v for k, v in r.headers.items() if k.lower() not in ("set-cookie",)})
    print(json.dumps(body, indent=1)[:1500])
