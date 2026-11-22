"""Wait for battery 1 (ep) to finish, then battery 3 on ep (35 queries per endpoint), then battery 3 on ref (free).
Usage: python run_battery3.py"""
import subprocess
import time

import common
from battery3 import PROMPTS3

while subprocess.run(["pgrep", "-f", "run_battery.py ep"], capture_output=True).returncode == 0:
    time.sleep(20)
b = common.budget()
print("budget left before resume:", min(b[e] for e in common.EPS), "-", max(b[e] for e in common.EPS), flush=True)
common.ask_scored(PROMPTS3, "bat3_ep", common.EPS)
print("ep done; budget left:", {e: v for e, v in common.budget().items() if e.startswith("ep-")}, flush=True)
common.ask(PROMPTS3, "bat3_ref", common.REFS)
