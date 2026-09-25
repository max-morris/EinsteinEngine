"""Derive an add_eqn order for the Z4c RHS from prioritize_rare_symbols alone.

Generates the recipe unsplit so the ordering function sees the whole RHS as one
kernel, then ranks the 13 author-level add_eqn calls by where their scalar
equations land.

The ordering function is forced to bare prioritize_rare_symbols, NOT the
composition Z4c.py normally bakes with. That composition is

    cartesian_product(insertion_order(exclude_synthetic), prioritize_rare_symbols)

and cartesian_product gives the first function first claim on every symbol. So
insertion_order claims all 13 author equations and prioritize_rare_symbols only
ever orders the CSE temporaries -- at author granularity the composition IS
insertion order, and deriving from it returns the recipe order unchanged
(verified: it produced the identity permutation).

Writes derived_order.json: {"rank": {add_eqn position -> sort key}}.
"""
import json, runpy, statistics, sys
from collections import OrderedDict

from EinsteinEngine.frontend.dsl.dsl_function_frontend import DslFunctionFrontend

REPO = "/home/sbrandt/repos/EinsteinEngine"

#  Record which scalar symbols each author-level add_eqn produces, by
#  snapshotting the equation list around each call. Prefix-matching names would
#  be unreliable here -- "Rt" is a prefix of "Rt_tmp", and tensors expand to
#  component names like RtDD00.
contributions: "OrderedDict[int, list]" = OrderedDict()
_orig = DslFunctionFrontend._add_eqn_now

def traced(self, lhs, rhs):
    if self.name != "z4c_rhs":
        return _orig(self, lhs, rhs)
    before = set(self.eqn_complex.eqn_lists[-1].eqns)
    _orig(self, lhs, rhs)
    after = self.eqn_complex.eqn_lists[-1].eqns
    n = len(contributions) + 1
    contributions[n] = [s for s in after if s not in before]
    return None

DslFunctionFrontend._add_eqn_now = traced

#  Force the bake to use bare prioritize_rare_symbols. The recipe passes its own
#  ordering_fn to bake(), so override at the point bake options are resolved.
from EinsteinEngine.frontend.dsl.dsl_frontend import DslFrontend
from EinsteinEngine.intermediate.eqn_ordering import prioritize_rare_symbols

_orig_fn_opts = DslFrontend._mk_function_bake_options

def forced(self, my_opts):
    opts = _orig_fn_opts(self, my_opts)
    for name, per_fn in opts.items():
        per_fn["ordering_fn"] = prioritize_rare_symbols
    return opts

DslFrontend._mk_function_bake_options = forced

recipe = f"{REPO}/recipes/Cottonmouth/Z4c.py"
sys.argv = [recipe]
g = runpy.run_path(recipe, run_name="__main__")

fun = g["fun_z4c_rhs"]
lists = fun.eqn_complex.eqn_lists
assert len(lists) == 1, f"expected one unsplit loop, got {len(lists)}"
order = list(lists[0].order)
pos = {sym: i for i, sym in enumerate(order)}

print(f"scalar equations in the baked order: {len(order)}")
print(f"author-level add_eqn calls traced:   {len(contributions)}")

rows = []
for n, syms in contributions.items():
    here = [pos[s] for s in syms if s in pos]
    if not here:
        #  Everything this equation produced was optimized away; leave it where
        #  it was rather than inventing a position.
        rows.append((n, float("inf"), 0, "(no surviving scalars)"))
        continue
    rows.append((n, statistics.median(here), len(here), f"{min(here)}..{max(here)}"))

print("\n add_eqn | scalars | positions      | median")
for n, med, cnt, span in rows:
    print(f"   {n:5d} | {cnt:7d} | {span:14s} | {med}")

ranked = sorted(rows, key=lambda r: (r[1], r[0]))
print("\nderived order (add_eqn positions, sorter's preference first):")
print("  " + " -> ".join(str(r[0]) for r in ranked))

#  Emit keys in [0,1); the recipe sorts by these, and the topological pass
#  fixes any dependency the median-position heuristic would have violated.
keys = {str(r[0]): i / len(ranked) for i, r in enumerate(ranked)}
with open("derived_order.json", "w") as fh:
    json.dump({"rank": keys,
               "derived_from": "bare prioritize_rare_symbols, unsplit",
               "order": [r[0] for r in ranked]}, fh, indent=2)
print("\nwrote derived_order.json")
