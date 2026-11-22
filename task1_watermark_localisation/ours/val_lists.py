"""Fold-0 validation images predicted in the partial regime, per method (for honest pipeline evaluation)."""
import json
from common import CACHE, METHODS
from map_rules import is_full
from train_seg import folds
for m in METHODS:
    items, fold_of = folds(m)
    names = [x["image_name"] for x in items if fold_of[x["image_name"]] == 0 and not is_full(x)]
    (CACHE / f"partial_val0_{m}.json").write_text(json.dumps(names)); print(m, len(names))
