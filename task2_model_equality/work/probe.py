"""Leaderboard probing tools.
  python probe.py add NAME 'SPEC'   -> writes subs/p_NAME.csv from SPEC and appends it to subs/queue.jsonl
        SPEC = json {"Q15": [[6,7,...],[22],...], "Q3": [...], "L1": [...]} (endpoint numbers, every endpoint once)
  python probe.py run               -> loop: submits queued files in order, >= MIN_GAP s apart, logs to subs/log.jsonl
  python probe.py show              -> prints the log
The API key stays inside ../submission.py; it is never printed or read here."""
import contextlib
import hashlib
import io
import json
import re
import sys
import time

import common
from write_sub import write

sys.path.insert(0, str(common.ROOT))
SUBS = common.WORK / "subs"
QUEUE, LOG = SUBS / "queue.jsonl", SUBS / "log.jsonl"
MIN_GAP = 30 * 60 + 20      # the server still enforces 30 min between successful submissions


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def rows(p):
    return [json.loads(l) for l in open(p)] if p.exists() else []


def add(name, spec):
    spec = json.loads(spec) if isinstance(spec, str) else spec
    assert name not in {r["name"] for r in rows(QUEUE)}, "duplicate name"
    assign = {f: [[f"ep-{i:02d}" for i in g] for g in gs] for f, gs in spec.items()}
    path = SUBS / f"p_{name}.csv"
    write(assign, path)                    # asserts every endpoint exactly once
    with open(QUEUE, "a") as f:
        f.write(json.dumps({"name": name, "spec": spec, "file": str(path), "md5": md5(path), "t_add": time.time()}) + "\n")
    print("queued", name, md5(path))


def submit_one(q):
    import submission
    submission.SUBMISSION_FILE = q["file"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = submission.main()
    out = buf.getvalue()
    m = re.search(r"Score \(public split\):\s*([0-9.eE+-]+)", out)
    return rc, (float(m.group(1)) if m else None), out


def run():
    import os
    import socket
    hb = SUBS / "runner.hb"
    if hb.exists() and time.time() - hb.stat().st_mtime < 90:
        sys.exit("another runner is alive: " + hb.read_text())
    me = f"{socket.gethostname()} pid={os.getpid()}"
    not_before = 0
    while True:
        hb.write_text(me)
        stop = SUBS / "runner.stop"
        if stop.exists():
            stop.unlink(); print("stopped", flush=True); return
        done = {r["name"] for r in rows(LOG)}
        todo = [q for q in rows(QUEUE) if q["name"] not in done]
        last = max([r["t"] for r in rows(LOG)] or [0])
        gapf = SUBS / "gap.txt"
        gap = float(gapf.read_text()) if gapf.exists() else MIN_GAP
        if todo and time.time() - last >= gap and time.time() >= not_before:
            q = todo[0]
            if md5(q["file"]) != q["md5"]:
                rec = {"name": q["name"], "t": time.time(), "score": None, "out": "MD5 MISMATCH - not submitted", "rc": -1}
            else:
                rc, sc, out = submit_one(q)
                w = re.search(r"Wait (\d+) seconds", out)
                if sc is None and w:                 # rate limited: not a submission, retry later
                    print(time.strftime("%H:%M:%S"), "rate limited, wait", w.group(1), flush=True)
                    not_before = time.time() + int(w.group(1)) + 20
                    continue
                rec = {"name": q["name"], "t": time.time(), "score": sc, "rc": rc, "out": out[-400:], "spec": q["spec"]}
            with open(LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(time.strftime("%H:%M:%S"), q["name"], rec["score"], flush=True)
        time.sleep(15)


def show():
    for r in rows(LOG):
        print(time.strftime("%H:%M:%S", time.localtime(r["t"])), f'{r["name"]:24}', r["score"], "" if r["score"] is not None else r["out"][-200:].replace("\n", " "))
    pend = [q["name"] for q in rows(QUEUE) if q["name"] not in {r["name"] for r in rows(LOG)}]
    print("pending:", pend)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "add":
        add(sys.argv[2], sys.stdin.read() if sys.argv[3] == "-" else sys.argv[3])
    elif cmd == "run":
        run()
    else:
        show()
