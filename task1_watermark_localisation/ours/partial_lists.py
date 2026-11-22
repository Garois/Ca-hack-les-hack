"""Lists of test images predicted in the partial regime (regime classifier), per method, split in halves."""
import json
from common import CACHE, METHODS, meta
from map_rules import is_full
for m in METHODS:
    names = [x["image_name"] for x in meta("test") if x["method"] == m and not is_full(x)]
    (CACHE / f"partial_test_{m}.json").write_text(json.dumps(names))
    for h in (0, 1):
        (CACHE / f"partial_test_{m}_h{h}.json").write_text(json.dumps(names[h::2]))
    print(m, len(names))
