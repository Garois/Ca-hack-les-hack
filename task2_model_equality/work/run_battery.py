"""Run the battery. Usage: python run_battery.py ref|ep [N_PROMPTS]"""
import sys

import common
from battery import PROMPTS

which = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else len(PROMPTS)
if which == "ref":
    common.ask(PROMPTS[:n], "bat_ref", common.REFS)
elif which == "ep":
    common.ask_scored(PROMPTS[:n], "bat_ep", common.EPS)
else:
    sys.exit("ref or ep")
