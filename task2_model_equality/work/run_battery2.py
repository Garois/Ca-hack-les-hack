"""Wait for battery 1 (ep) to finish, then run battery 2 on ep. Usage: python run_battery2.py"""
import subprocess
import time

import common
from battery2 import PROMPTS2

while subprocess.run(["pgrep", "-f", "run_battery.py ep"], capture_output=True).returncode == 0:
    time.sleep(30)
common.ask_scored(PROMPTS2, "bat2_ep", common.EPS)
