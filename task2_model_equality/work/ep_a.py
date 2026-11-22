"""Phase A on scored endpoints: 3 answers per endpoint (identity, free text, quiz) -> families + data for local scoring."""
import common
from s1_explore import PROMPTS

P = {pid: text for pid, text in PROMPTS}
EP_A = [("who", P["who"]), ("story", P["story"]), ("quiz", P["quiz"])]

if __name__ == "__main__":
    common.ask_scored(EP_A, "ep_a", common.EPS, repeats=1)
