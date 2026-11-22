"""Shared helpers: API key loading (from ../.env, never printed) + wrappers around the starter kit's query()."""
import json
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
ROOT = WORK.parent
DATA = WORK / "data"
sys.path.insert(0, str(ROOT))

import query_reference as qr  # noqa: E402
import task_template as tt    # noqa: E402

REFS = [f"ref-{i:02d}" for i in range(1, 10)]
EPS = [f"ep-{i:02d}" for i in range(1, 34)]


def api_key() -> str:
    """Reads API_KEY from ROOT/.env. The value is never printed."""
    f = ROOT / ".env"
    if not f.exists():
        sys.exit(f"{f} not found (expected a line API_KEY=...).")
    for line in f.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        if line.startswith("API_KEY") and "=" in line:
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            if v:
                return v
    sys.exit("API_KEY not found in .env")


def setup():
    k = api_key()
    qr.API_KEY = k
    tt.API_KEY = k


def budget() -> dict:
    setup()
    return tt.get_budget(tt.make_session())


def ask(prompts, out_name, endpoints, repeats=1, max_tokens=256):
    """prompts: list of (id, text). Appends answers to work/data/<out_name>.jsonl (resumable, skips done ones).
    Refuses to touch scored endpoints unless explicitly allowed via ask_scored()."""
    assert all(e.startswith("ref-") for e in endpoints), "use ask_scored() for ep- endpoints"
    _ask(prompts, out_name, endpoints, repeats, max_tokens)


def ask_scored(prompts, out_name, endpoints, repeats=1, max_tokens=256):
    """Same as ask() but for the budgeted ep- endpoints. Each answer costs 1 of the 100 per endpoint."""
    _ask(prompts, out_name, endpoints, repeats, max_tokens)


def _ask(prompts, out_name, endpoints, repeats, max_tokens):
    setup()
    DATA.mkdir(exist_ok=True)
    tt.query(prompts, output=DATA / f"{out_name}.jsonl", endpoints=endpoints, repeats=repeats, max_tokens=max_tokens)


def load(out_name):
    p = DATA / f"{out_name}.jsonl"
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []
