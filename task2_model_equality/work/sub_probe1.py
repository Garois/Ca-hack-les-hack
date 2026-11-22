"""Probe 1: families only (every endpoint its own model). Expected public score = 0.1 * MCC_family = 0.100 if the
three families are right."""
from fams import FAMILIES
from write_sub import write

write({f: [[e] for e in eps] for f, (_, eps) in FAMILIES.items()}, "subs/probe1_families_only.csv")
print(open("subs/probe1_families_only.csv").read())
