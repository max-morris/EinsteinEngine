# Equation ordering and split loci

This document explains how the elaborator orders a function's equations while it bakes it, and where the automatic
("auto") split predicates cut the function into loops. It is written for recipe authors and tuner users who know the
engine's basics (recipes, `add_eqn`, `split_loop()` / `soft_split()`, global CSE, temporary promotion, soft split
retainment) but not the ordering and split-locus features.

In short:

- A function's bake has **three points** at which it can be ordered, and each one has its own **ordering function**:
  `early_ordering_fn` (whole `add_eqn` calls), `pre_population_ordering_fn` (scalar equations, before CSE), and
  `ordering_fn` (after CSE, as before).
- The auto split predicates are evaluated at **one** of those points per function, its **split locus**
  (`SplitLocus.Early`, `SplitLocus.PrePopulation` or `SplitLocus.PostPopulation`). The locus decides what a
  "position" is.
- Positions are **0-based**: position `i` means "split after element `i`".
- Every function keeps **two orders**: the recipe order, which is never rewritten, and the pre-population order, which
  the first two ordering functions rewrite. `insertion_order` is gone; use `pre_population_order` instead.
- The early order can be given explicitly (`add_eqn_order([...])`) or derived by the engine from a trial bake
  (`rank_by_post_population(fn)`, section 7), which ranks the `add_eqn` calls by where `fn` puts their equations after
  CSE.

How to run a tuning search is covered in [TUNING-QUICKSTART.md](TUNING-QUICKSTART.md). This document only covers the
parts of tuning that depend on ordering and loci (section 6).

## Contents

1. [The bake pipeline of a function](#1-the-bake-pipeline-of-a-function)
2. [Two orders: recipe order and pre-population order](#2-two-orders-recipe-order-and-pre-population-order)
3. [Groups and the early ordering function](#3-groups-and-the-early-ordering-function)
4. [Split loci](#4-split-loci)
5. [What the engine guarantees at a cut](#5-what-the-engine-guarantees-at-a-cut)
6. [Tuning](#6-tuning)
7. [Deriving an early order automatically: `rank_by_post_population`](#7-deriving-an-early-order-automatically-rank_by_post_population)
8. [Known limitations and pre-existing issues](#8-known-limitations-and-pre-existing-issues)
9. [FAQ](#9-faq)
10. [Glossary](#10-glossary)
11. [Where the code lives](#11-where-the-code-lives)

## Running the examples

Every example below was run against the engine, and its output is shown as printed. Each one assumes this
setup code:

```python
from EinsteinEngine import *
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornDef


def show(fun):
    """Print each loop of a baked function, in its final order."""
    for i, el in enumerate(fun.eqn_complex.eqn_lists):
        print(f"  loop {i}:", [str(lhs) for lhs in el.order])


def show_eqns(fun):
    """Print each loop of a baked function with its equations, and the function's tile temporaries."""
    for i, el in enumerate(fun.eqn_complex.eqn_lists):
        print(f"  loop {i}:", ", ".join(f"{lhs} = {el.eqns[lhs]}" for lhs in el.order))
    print("  tile temporaries:", sorted(str(t) for t in fun.eqn_complex.tile_temporaries))


class Recorder:
    """A split predicate that records the positions it is asked about."""
    def __init__(self, fire):
        self.fire, self.calls = fire, []

    def __call__(self, i):
        self.calls.append(i)
        return self.fire(i)
```

The engine prints progress messages ("Early Baking fn...", "Performing CSE...", and so on) while it bakes. They are
left out of the outputs shown here. Warnings are kept. Run with `PYTHONHASHSEED=0` if you want to reproduce the exact
temporary names.

---

## 1. The bake pipeline of a function

### Background

A recipe describes the program. The frontend flattens it into the *abstract program*: each `add_eqn` call is exploded
into scalar equations (one per tensor component), and each function holds its equations in one or more *equation
lists*, one per loop (kernel). `split_loop()` and `soft_split()` start a new list when the recipe calls them. The
*elaborator* then orders, CSEs, splits and promotes, and the generator turns each list into a loop.

Ordering and splitting happen while `bake()` runs. Before the split loci were introduced, there was one ordering
function, `ordering_fn`, and the auto split predicates were called inside `add_eqn`, right after each call. Now the
predicates are called at bake time, and there are three ordering functions.

### The pipeline

```
 recipe: add_eqn(...), split_loop(), soft_split(), pull_out(...), overwrite(...)
   |      (each add_eqn call records its 0-based index as the origin of its scalar equations)
   v
 DslFrontend.bake(**options)
   |
   |  0. (only with rank_by_post_population) trial bake, section 7
   |
   |  for each function (_early_bake):
   |  +-------------------------------------------------------------------------------+
   |  | 1. early_ordering_fn, if set: order each list's add_eqn groups                |
   |  |    and rewrite the pre-population order                                       |
   |  |    >>> EARLY LOCUS: cut between groups                                        |
   |  | 2. pull-out: each pull_out(expr) becomes a named temporary                    |
   |  | 3. complexity analysis; madd, if do_madd                                      |
   |  | 4. pre-CSE bake, ordered by pre_population_ordering_fn, or by ordering_fn     |
   |  |    when that is None. This fixes the order CSE sees. If                       |
   |  |    pre_population_ordering_fn is set, its result becomes the pre-population   |
   |  |    order.                                                                     |
   |  | 5. every list's ordering_fn is set to ordering_fn (it is sticky; see below)   |
   |  |    >>> PREPOPULATION LOCUS: cut between scalar equations, in the order of 4   |
   |  +-------------------------------------------------------------------------------+
   |
   |  global CSE, if do_cse ("population": CSE populates the lists with temporaries)
   |    then every list is rebaked with ordering_fn
   |    >>> POSTPOPULATION LOCUS: cut in the post-CSE order
   |
   |  merge_soft_splits: soft splits from any source (manual or auto) are merged back
   |    into one loop here, using the soft split retainment strategy
   |  splitmaxxing, if enabled; late bake (recycling temporaries)
   v
 generator: one loop per remaining list
```

**About the word "population".** In this document and in the code, "population" means *global CSE populating the
equation lists with temporaries*. It is not the paper's "populating the abstract program" (which is what the frontend
does while the recipe runs). "Pre-population" therefore means "before CSE has added its temporaries", and
"post-population" means "after".

### The three ordering functions

All three are ordinary `EqnOrderingFn`s: functions `(eqns, eqn_list) -> iterator of LHS symbols` (optionally with
annotations), such as `maximize_symbol_reuse`, `prioritize_rare_symbols`, `recipe_order` or
`pre_population_order`. Whatever order they yield, the engine repairs it so that each equation comes after the
equations it reads.

| Bake option                  | Runs at                                  | Orders                                                      | Default                      |
|------------------------------|------------------------------------------|-------------------------------------------------------------|------------------------------|
| `early_ordering_fn`          | step 1, before pull-out and CSE          | whole `add_eqn` calls ("groups"; see section 3)             | `None`: keep the recipe order |
| `pre_population_ordering_fn` | step 4, the pre-CSE bake                 | scalar equations and pull-out temporaries                   | `None`: use `ordering_fn`     |
| `ordering_fn`                | after CSE, and every later rebake        | everything, CSE temporaries included                        | `maximize_symbol_reuse`       |

Things to know about them:

- **`ordering_fn` runs twice when `pre_population_ordering_fn` is None.** It orders the pre-CSE bake (step 4) and the
  post-CSE rebake. This is exactly what the engine did before the split loci existed, so recipes that set only
  `ordering_fn` generate the same code as before. In this case the pre-population order is *not* rewritten; only the
  baked order of step 4 follows `ordering_fn`.
- **`ordering_fn` is sticky.** Each list stores it, and every later rebake (after CSE, after a cut, after a merge)
  reuses it. That is why step 5 resets every list to `ordering_fn` after the pre-CSE bake. The lists produced by a cut
  inherit it.
- **Bayesian optimization never runs before CSE.** An ordering function whose name (or whose `functools.partial`'s
  function's name) contains `bayesian` is replaced by `prioritize_rare_symbols` for any bake before CSE and for any
  "fast" rebake. This applies to `early_ordering_fn` and `pre_population_ordering_fn` too, so passing
  `bayesian_optimization` there is the same as passing `prioritize_rare_symbols`. (`rank_by_post_population` refuses a
  Bayesian function outright; see section 7.)
- **The early ordering function is not used as an order directly.** It ranks groups (section 3). The other two are
  used as orders.

### Setting them

The options go to `bake()`, either for every function or per function through `functions={name: {...}}`, which
overrides the bake-wide values for that function. `add_eqn_order([...])` refers to one function's `add_eqn` calls, so
pass it per function. (`rank_by_post_population(fn)` can be passed bake-wide: it derives each function's own order.)
Z4c does this for its RHS function (`recipes/Cottonmouth/Z4c.py`):

```python
cottonmouth_Z4c.bake(
    do_cse=True,
    temporary_promotion_strategy=promote_none(),
    # ... other options ...
    ordering_fn=ordering_fn,
    # The tuners' ordering knobs refer to the z4c_rhs add_eqn calls, so they apply to z4c_rhs only.
    functions={"z4c_rhs": {
        "early_ordering_fn": get_optional_tuning_param('early_ordering_fn', None),
        "pre_population_ordering_fn": get_optional_tuning_param('pre_population_ordering_fn', None),
        "ordering_fn": get_optional_tuning_param('ordering_fn', ordering_fn),
    }}
)
```

A name in `functions` that is not a function of the thorn is an error, so a typo cannot silently drop the options:

```python
gf = ThornDef("DOCS", "UNKNOWNFN")
a, src = gf.decl("a", []), gf.decl("src", [])
fun = gf.create_function("my_rhs", ScheduleBin.Evolve)
fun.add_eqn(a, src)
try:
    gf.bake(functions={"my_rsh": {"early_ordering_fn": add_eqn_order([0])}})
except Exception as e:
    print(e)
```

```text
The bake options name unknown functions ['my_rsh']; the functions are ['my_rhs'].
```

---

## 2. Two orders: recipe order and pre-population order

Every equation list keeps two orders of its equations, as dicts from LHS to position:

- **Recipe order** (`EqnList.eqn_recipe_order`): the order in which the equations were added to the function,
  numbered function-wide. It is **never rewritten**. It is the control: whatever the ordering functions do, you can
  always get back to "the order the author wrote".
- **Pre-population order** (`EqnList.eqn_pre_population_order`): starts equal to the recipe order, and is rewritten by
  `early_ordering_fn` (step 1) and by `pre_population_ordering_fn` (step 4). Without them it stays the recipe order.

Neither of them is the order the code is generated in. That is the *baked order*, `EqnList.order`, which each bake
computes by running the list's ordering function (with dependency repair). The two stored orders are inputs that an
ordering function can use.

| Event                                   | Recipe order                            | Pre-population order                                  |
|-----------------------------------------|-----------------------------------------|-------------------------------------------------------|
| `add_eqn`                               | appended                                | appended                                              |
| `early_ordering_fn` (step 1)            | unchanged                               | replaced by the group order                           |
| pull-out, CSE and splitmaxxing temps    | appended, in creation order             | appended, in creation order                           |
| `pre_population_ordering_fn` (step 4)   | unchanged                               | replaced by the pre-CSE baked order                   |
| a cut (any locus)                       | each piece keeps its equations' entries | each piece keeps its equations, in the source's order |
| soft split merge                        | absorbed equations keep their positions | absorbed equations appended, in their own list's pre-population order |

The ordering functions `recipe_order` and `pre_population_order` (both exported from `EinsteinEngine`) yield the
equations in these orders. Both take `exclude_synthetic_symbols=False`. When it is True, synthetic symbols (CSE
temporaries) are not yielded and dependency repair places them just before their first reader. Z4c uses:

```python
ordering_fn = cartesian_product(
    functools.partial(pre_population_order, exclude_synthetic_symbols=True),
    prioritize_rare_symbols
)
```

### `insertion_order` was removed

`insertion_order` (and `EqnList.eqn_insertion_order`) no longer exist. **Replace `insertion_order` with
`pre_population_order`.** Without an early or pre-population ordering function, it gives exactly what
`insertion_order` gave, and the Z4c output is byte-identical.

Do *not* replace it with `recipe_order`. The two differ after a soft split merge: the old insertion order appended the
equations a merge moved into a list, and so does the pre-population order, while the recipe order puts them back at
their recipe positions and lists every temporary after every recipe equation. For example, with one soft split after
call 0, merged with `retain_all()`:

```python
for early in (None, add_eqn_order([0, 2, 1, 3])):
    gf = ThornDef("DOCS", "MERGEORDERS" + ("A" if early is None else "B"))
    a, b, c, d, src = [gf.decl(s, []) for s in ("a", "b", "c", "d", "src")]
    fun = gf.create_function("fn", ScheduleBin.Evolve, auto_soft_split_predicate=lambda i: i == 0)
    fun.add_eqn(a, sin(src) + pull_out(cos(src) * src))   # call 0
    fun.add_eqn(b, a * cos(src))                          # call 1
    fun.add_eqn(c, src * 3 + pull_out(sin(src) * src))    # call 2
    fun.add_eqn(d, b + c)                                 # call 3
    gf.bake(do_cse=True, soft_split_retainment_strategy=retain_all(),
            early_ordering_fn=early, ordering_fn=pre_population_order)
    [el] = fun.eqn_complex.eqn_lists
    print("early_ordering_fn =", early)
    print("  recipe order:        ", [str(k) for k in el.eqn_recipe_order])
    print("  pre-population order:", [str(k) for k in el.eqn_pre_population_order])
```

```text
early_ordering_fn = None
  recipe order:         ['a', 'b', 'c', 'd', 'pull_out_0', 'pull_out_1', 'x0', 'x1', 'x2']
  pre-population order: ['a', 'pull_out_0', 'x0', 'x1', 'x2', 'b', 'c', 'd', 'pull_out_1']
early_ordering_fn = add_eqn_order([0, 2, 1, 3])
  recipe order:         ['a', 'b', 'c', 'd', 'pull_out_0', 'pull_out_1', 'x0', 'x1', 'x2']
  pre-population order: ['a', 'pull_out_0', 'x0', 'x1', 'x2', 'c', 'b', 'd', 'pull_out_1']
```

In the pre-population order, the first loop's equations and temporaries come first, followed by the absorbed second
loop. With the early ordering function, it also keeps `c` before `b`.

---

## 3. Groups and the early ordering function

### What a group is

A **group** is the set of scalar equations created by one author-level `add_eqn` call. `add_eqn(v[li], [...])` with
three components makes one group of three equations. Every equation records the 0-based index of the `add_eqn` call
that created it, its **origin** (`EqnList.eqn_origin`):

- The first `add_eqn` call of a function is call 0, the next call 1, and so on, across all of the function's lists.
- A pull-out temporary inherits the origin of the equation it was pulled out of.
- CSE temporaries have no origin. Equations added by synthetic functions (the ones global CSE creates) have none
  either, and each of them is a group of its own.

The early locus reorders whole groups, and never the equations inside a group. Within a group, the equations stay in
recipe order.

### How the early ordering function orders groups

When `early_ordering_fn` is set, each list is processed on its own (lists are never merged or reordered relative to
each other, so nothing moves across a manual split):

1. **Run the function over the list's scalar equations.** The result is a position for each scalar equation.
   Equations the function does not yield rank last, in recipe order.
2. **Rank each group by the median position of its members.** Ties go to recipe order. A group lands where most of
   its members are, not where its first member is.
3. **Find the dependencies between groups.** Group B depends on group A if an equation of B reads a LHS of A. There is
   one more kind of edge: a group that reads `X` must come before a group that writes `X'` (an `overwrite` of `X`),
   because nothing else connects them (this is the **reader-before-overwriter** edge).
4. **Merge dependency cycles into super-groups.** Two groups can depend on each other even though their scalar
   equations do not form a cycle (group A computes `vD1 = 2*b`, group B computes `b = 3*vD0`). Such groups are
   merged into one **super-group**, ranked by the median over all of its members.
5. **Sort topologically by rank.** Repeatedly emit the lowest-ranked (super-)group whose dependencies have all been
   emitted (Kahn's algorithm on a heap). This is the **dependency repair**: an ordering function can ask for any
   order, but a group never comes before a group it reads.
6. **Rewrite the pre-population order** with the result.

Without `early_ordering_fn`, none of this happens. The groups stay in recipe order, and each `add_eqn` call is its own
group, even when an earlier call reads what a later call writes (section 5, "The backward-read guard").

### `add_eqn_order` and `add_eqn_key_order`

Two helpers make early ordering functions from `add_eqn` call indices (both exported from `EinsteinEngine`):

- `add_eqn_order([i0, i1, ...])`: put call `i0` first, then `i1`, and so on. Indices are 0-based. Calls that are not
  listed keep their recipe order after the listed ones. Repeated or negative indices are rejected when it is created,
  and an index the function does not have is rejected at bake time.
- `add_eqn_key_order(key)`: order the calls by `key(call_index)`, ties by recipe order. For example,
  `add_eqn_key_order(lambda i: -i)` reverses them.

Their `repr` shows the order (`add_eqn_order([8, 1, 0, ...])`), which the tuners record (section 6).

A third option, `rank_by_post_population(fn)`, derives the order from a trial bake instead of taking it from you.
Section 7 explains it.

```python
gf = ThornDef("DOCS", "ORDERRANGE")
a, b, src = gf.decl("a", []), gf.decl("b", []), gf.decl("src", [])
fun = gf.create_function("fn", ScheduleBin.Evolve)
fun.add_eqn(a, src)
fun.add_eqn(b, src * 2)
try:
    gf.bake(early_ordering_fn=add_eqn_order([2, 1, 0]))
except Exception as e:
    print(e)
try:
    add_eqn_order([1, 0, 1])
except Exception as e:
    print(e)
```

```text
add_eqn_order: call index 2 is out of range; the function has 2 add_eqn calls (valid indices are 0..1).
add_eqn_order: the call indices [1] are listed more than once.
```

Steve's derived Z4c order (1-based `9, 2, 1, 3, 4, 6, 11, 8, 5, 10, 13, 7, 12` on feature/eqn-order-instrumentation)
is `add_eqn_order([8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11])` here. `rank_by_post_population(prioritize_rare_symbols)`
derives the same order (section 7).

### Worked example: reordering with dependency repair

Four calls, one of them a vector. Asking for the reverse order cannot put `a` last, because the vector reads it:

```python
def recipe(name, **create_kwargs):
    gf = ThornDef("DOCS", name)
    v = gf.decl("v", [li])
    a, b, d, src = [gf.decl(n, []) for n in ("a", "b", "d", "src")]
    fun = gf.create_function("fn", ScheduleBin.Evolve, **create_kwargs)
    fun.add_eqn(a, src**2 + 1)                    # call 0
    fun.add_eqn(v[li], [a * 2, a * src, a + src])  # call 1: one group of three scalar equations
    fun.add_eqn(b, v[l0] * v[l1] + v[l2])         # call 2
    fun.add_eqn(d, src * 3)                       # call 3
    return gf, fun


gf, fun = recipe("EARLYA")
gf.bake(do_cse=False, ordering_fn=pre_population_order)
print("no early_ordering_fn:")
show(fun)

gf, fun = recipe("EARLYB")
gf.bake(do_cse=False, ordering_fn=pre_population_order,
        early_ordering_fn=add_eqn_order([3, 2, 1, 0]))
el = fun.eqn_complex.eqn_lists[0]
print("add_eqn_order([3, 2, 1, 0]):")
show(fun)
print("  eqn_origin:          ", {str(k): v for k, v in el.eqn_origin.items()})
print("  recipe order:        ", [str(k) for k in el.eqn_recipe_order])
print("  pre-population order:", [str(k) for k in el.eqn_pre_population_order])
```

```text
no early_ordering_fn:
  loop 0: ['a', 'vD0', 'vD1', 'vD2', 'b', 'd']
add_eqn_order([3, 2, 1, 0]):
  loop 0: ['d', 'a', 'vD0', 'vD1', 'vD2', 'b']
  eqn_origin:           {'a': 0, 'vD0': 1, 'vD1': 1, 'vD2': 1, 'b': 2, 'd': 3}
  recipe order:         ['a', 'vD0', 'vD1', 'vD2', 'b', 'd']
  pre-population order: ['d', 'a', 'vD0', 'vD1', 'vD2', 'b']
```

Step by step: the ranks are `d` 0, `b` 1, `v` 2, `a` 3. At the start only `a` and `d` have no unemitted
dependencies, so `d` (rank 0) goes first. Then only `a` is ready, then `v`, then `b`. The baked order follows the
pre-population order here because `ordering_fn=pre_population_order`.

### Worked example: median ranking

An ordering function that interleaves the three components of `w` with `e`. The group `w` sits at positions 0, 2
and 3 (median 2), and `e` at position 1, so `e` goes first:

```python
def interleave(eqns, eqn_list):
    by_name = {str(lhs): lhs for lhs in eqns}
    for name in ("wD0", "e", "wD1", "wD2"):
        yield by_name[name]

gf = ThornDef("DOCS", "MEDIAN")
w = gf.decl("w", [li])
e, src = gf.decl("e", []), gf.decl("src", [])
fun = gf.create_function("fn", ScheduleBin.Evolve)
fun.add_eqn(w[li], [src, src * 2, src * 3])  # call 0: positions 0, 2, 3 -> median 2
fun.add_eqn(e, src * 5)                      # call 1: position 1         -> median 1
gf.bake(do_cse=False, ordering_fn=pre_population_order, early_ordering_fn=interleave)
show(fun)
```

```text
  loop 0: ['e', 'wD0', 'wD1', 'wD2']
```

### Worked example: a super-group

Call 0 computes `vD1 = 2*b`, and call 1 computes `b = 3*vD0`. The two groups depend on each other, so they form one
super-group, which is **one early position**. Three calls give only two positions:

```python
hard = Recorder(lambda i: i == 0)
gf = ThornDef("DOCS", "SUPERGROUP")
v = gf.decl("v", [li])
b, c, src = gf.decl("b", []), gf.decl("c", []), gf.decl("src", [])
fun = gf.create_function("fn", ScheduleBin.Evolve, auto_hard_split_predicate=hard)
fun.add_eqn(v[li], [src, b * 2, src * 3])  # call 0
fun.add_eqn(b, v[l0] * 3)                  # call 1
fun.add_eqn(c, src * 5)                    # call 2
gf.bake(do_cse=False, ordering_fn=pre_population_order, early_ordering_fn=add_eqn_order([2, 1, 0]))
print("positions queried:", hard.calls)
show(fun)
```

```text
positions queried: [0, 1]
  loop 0: ['c']
  loop 1: ['vD0', 'b', 'vD1', 'vD2']
```

The members of a super-group are in recipe order, with dependency repair (`b` has to come between `vD0` and `vD1`).

### Worked example: reader before overwriter

`r` reads `X`, and a later call overwrites `X`. Asking for the overwrite first does not move it before the read:

```python
gf = ThornDef("DOCS", "OVERWRITEEDGE")
X, r, src = gf.decl("X", []), gf.decl("r", []), gf.decl("src", [])
Xp = gf.overwrite(X)
fun = gf.create_function("fn", ScheduleBin.Evolve)
fun.add_eqn(r, X * 2)  # call 0
fun.add_eqn(Xp, src)   # call 1
gf.bake(do_cse=False, ordering_fn=pre_population_order, early_ordering_fn=add_eqn_order([1, 0]))
show(fun)
```

```text
  loop 0: ['r', "X'"]
```

---

## 4. Split loci

### The predicates and the locus

A function's auto split predicates and its locus are arguments of `create_function`:

```python
fun = gf.create_function(
    "my_rhs", ScheduleBin.Evolve,
    auto_hard_split_predicate=hard,   # Callable[[int], bool]
    auto_soft_split_predicate=soft,   # Callable[[int], bool | SoftSplitRetainmentStrategy]
    auto_split_locus=SplitLocus.Early,  # the default
)
```

- The predicates are **no longer called inside `add_eqn`**. They are called once per position at bake time, at the
  function's locus, after the ordering function of that locus has run. So they see the new order.
- At each position, the hard predicate is asked first. The soft predicate is asked only if there is no hard predicate
  or it returned False. **Hard wins over soft.**
- The soft predicate returns False (no split), True (a soft split with the bake's `soft_split_retainment_strategy`),
  or a `SoftSplitRetainmentStrategy` (a soft split with that strategy).
- A function has one locus. Its predicates are evaluated there and nowhere else.
- The predicates are called at every position, in increasing order, even where the engine then refuses the split
  (section 5), so a tuner always sees the same positions.

### What a position is

A **position** is a slot between two consecutive **elements** in the order that exists at the locus. Position `i` is
the slot right after element `i` (counting from 0), so a split at position `i` means "split after element `i`". A
function with `N` elements has positions `0..N-1`, and position `N-1` is after the last element.

| Locus                       | Elements                                                                       | Their order                                                                                 | Z4c `z4c_rhs` N |
|-----------------------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-----------------|
| `SplitLocus.Early` (default) | `add_eqn` calls, or super-groups when `early_ordering_fn` is set              | `early_ordering_fn` over groups; the recipe order without it                                | 13              |
| `SplitLocus.PrePopulation`  | scalar equations and pull-out temporaries                                      | the pre-CSE baked order: `pre_population_ordering_fn`, or `ordering_fn` when that is None   | 64              |
| `SplitLocus.PostPopulation` | scalar equations, pull-out temporaries and CSE temporaries                     | the post-CSE baked order: `ordering_fn`                                                      | 1061            |

The Z4c counts are for the recipe's default ordering functions.

Positions **run across lists**. If manual splits made two lists, the positions of the second list continue where the
first list's ended. A cut is always made within one list: pieces never span two source lists, and no locus moves an
equation from one list to another. So a manual `split_loop()` or `soft_split()` is always kept, and nothing is ever
reordered across it.

**Indexing used to be 1-based.** Previously the predicates were called with the count of `add_eqn` calls made so far
(1 for the first call), and the tuners named their params `split_1..N`. Now position 0 means "after the first
element". See section 6 for converting old checkpoints.

### The same recipe at the three loci

```python
def recipe(name, locus, hard):
    gf = ThornDef("DOCS", name)
    a, b, c, u, v = [gf.decl(n, []) for n in ("a", "b", "c", "u", "v")]
    fun = gf.create_function("f", ScheduleBin.Evolve, auto_split_locus=locus, auto_hard_split_predicate=hard)
    fun.add_eqn(a, pull_out(sin(u) * cos(v)) + (u * v + 1) ** 2)  # call 0
    fun.add_eqn(b, (u * v + 1) ** 3 + a)                          # call 1
    fun.add_eqn(c, (u * v + 1) * b)                               # call 2
    return gf, fun


for locus in (SplitLocus.Early, SplitLocus.PrePopulation, SplitLocus.PostPopulation):
    hard = Recorder(lambda i: False)
    gf, fun = recipe(f"LOCI{locus.name.upper()}", locus, hard)
    gf.bake(do_cse=True, temporary_promotion_strategy=promote_none(), ordering_fn=pre_population_order)
    print(f"{locus.name}: positions {hard.calls}")
show(fun)
```

```text
Early: positions [0, 1, 2]
PrePopulation: positions [0, 1, 2, 3]
PostPopulation: positions [0, 1, 2, 3, 4, 5, 6]
  loop 0: ['x0', 'x1', 'pull_out_0', 'x2', 'a', 'b', 'c']
```

Three `add_eqn` calls give 3 early positions. Pull-out adds `pull_out_0`, which gives 4 pre-population positions. CSE
adds `x0`, `x1` and `x2`, which gives 7 post-population positions.

### Empty pieces are dropped

A cut can produce a piece with no equations: a split after the last element, or a split right before a manual split.
An empty piece computes nothing, so it is dropped, and its boundary is folded into the next piece's:

- Between a hard and a soft boundary, the hard one wins.
- Between two soft boundaries, the later one (the one attached to actual equations) wins, with its retainment
  strategy.
- A custom loop annotation from a manual split survives.

So **a split after the last position does nothing**. Before the split loci existed, a hard split there emitted an
empty loop (an empty kernel launch), and Steve's recipe-order control checkpoint has one. An empty list that manual
splits left behind (e.g. `split_loop()` before the first `add_eqn`) is dropped only when some auto split fires in that
function; otherwise it stays, as it always has.

### Early locus: worked example

The recipe from section 3, with a hard split after early position 0. The position refers to the new order, so the
split comes after `d`:

```python
hard = Recorder(lambda i: i == 0)
gf, fun = recipe("EARLYC", auto_hard_split_predicate=hard)  # recipe() from section 3
gf.bake(do_cse=False, ordering_fn=pre_population_order,
        early_ordering_fn=add_eqn_order([3, 2, 1, 0]))
print("positions queried:", hard.calls)
show(fun)
```

```text
positions queried: [0, 1, 2, 3]
  loop 0: ['d']
  loop 1: ['a', 'vD0', 'vD1', 'vD2', 'b']
```

With a manual split, positions continue across the lists, and each list is reordered on its own:

```python
hard = Recorder(lambda i: i == 2)
gf = ThornDef("DOCS", "MANUAL")
a, b, c, d, src = [gf.decl(s, []) for s in ("a", "b", "c", "d", "src")]
fun = gf.create_function("fn", ScheduleBin.Evolve, auto_hard_split_predicate=hard)
fun.add_eqn(a, src)        # call 0
fun.add_eqn(b, src * 2)    # call 1
fun.split_loop("custom")   # manual hard split
fun.add_eqn(c, src * 3)    # call 2
fun.add_eqn(d, src * 4)    # call 3
gf.bake(do_cse=False, ordering_fn=pre_population_order,
        early_ordering_fn=add_eqn_order([3, 2, 1, 0]))
print("positions queried:", hard.calls)
for i, el in enumerate(fun.eqn_complex.eqn_lists):
    print(f"  loop {i} ({fun.source_annotations.loops.get(i)}):", [str(lhs) for lhs in el.order])
```

```text
positions queried: [0, 1, 2, 3]
  loop 0 (fn loop 0): ['b', 'a']
  loop 1 (custom): ['d']
  loop 2 (fn loop 2): ['c']
```

The early order is `[b a] | [d c]`, so position 2 is after `d`. Default loop annotations are renamed by final loop
index; a custom one is kept.

Without an early ordering function, the early locus is the historical behavior: the elements are the `add_eqn` calls
in recipe order, and the lists produced are the ones the old in-`add_eqn` splitting produced, with two exceptions:
empty pieces are dropped (above), and the cut guards refuse some cuts (section 5, "The cut guards"). Early cuts are
made before the first bake, so, like manual splits, they are not validated (section 5).

### Pre-population locus: worked example

A hard split after every position shows what the elements are, one per loop. Without `pre_population_ordering_fn`,
the elements are in the order `ordering_fn` gives the pre-CSE bake:

```python
n = 0


def build(label, **bake_opts):
    global n
    n += 1
    gf = ThornDef("DOCS", f"PREPOP{n}")
    c, b, a, u, v = [gf.decl(s, []) for s in ("c", "b", "a", "u", "v")]
    fun = gf.create_function("f", ScheduleBin.Evolve, auto_split_locus=SplitLocus.PrePopulation,
                             auto_hard_split_predicate=lambda i: True)
    fun.add_eqn(c, pull_out(sin(u) * cos(v)) + u)  # call 0
    fun.add_eqn(b, u * v)                          # call 1
    fun.add_eqn(a, b + c)                          # call 2
    gf.bake(do_cse=False, **bake_opts)
    print(label)
    show(fun)


build("ordering_fn=recipe_order:", ordering_fn=recipe_order)
build("ordering_fn=lexicographical_order:", ordering_fn=lexicographical_order)
build("pre_population_ordering_fn=recipe_order, ordering_fn=lexicographical_order:",
      pre_population_ordering_fn=recipe_order, ordering_fn=lexicographical_order)
```

```text
ordering_fn=recipe_order:
  loop 0: ['pull_out_0']
  loop 1: ['c']
  loop 2: ['b']
  loop 3: ['a']
ordering_fn=lexicographical_order:
  loop 0: ['b']
  loop 1: ['pull_out_0']
  loop 2: ['c']
  loop 3: ['a']
pre_population_ordering_fn=recipe_order, ordering_fn=lexicographical_order:
  loop 0: ['pull_out_0']
  loop 1: ['c']
  loop 2: ['b']
  loop 3: ['a']
```

`lexicographical_order` asks for `a, b, c, pull_out_0`, and dependency repair turns that into
`b, pull_out_0, c, a`. In the last case, `pre_population_ordering_fn` takes over the pre-CSE bake, so the positions
follow the recipe order even though `ordering_fn` is lexicographical.

After a pre-population cut, CSE runs on the pieces exactly as it runs on lists made by manual splits.

### Post-population locus: worked example

`x0 = u*v + 1` is shared by all three equations. The unsplit order is `x0, a, b, c`, so position 1 is after `a`:

```python
n = 0


def build(hard=None, soft=None, **bake_opts):
    global n
    n += 1
    gf = ThornDef("DOCS", f"POST{n}")
    a, b, c, u, v = [gf.decl(s, []) for s in ("a", "b", "c", "u", "v")]
    fun = gf.create_function("f", ScheduleBin.Evolve, auto_split_locus=SplitLocus.PostPopulation,
                             auto_hard_split_predicate=hard, auto_soft_split_predicate=soft)
    fun.add_eqn(a, (u * v + 1) ** 2)
    fun.add_eqn(b, (u * v + 1) ** 3 + a)
    fun.add_eqn(c, (u * v + 1) * b)
    gf.bake(do_cse=True, **bake_opts)
    return fun


print("unsplit:")
show_eqns(build())
print("hard cut after position 1, promote_none():")
show_eqns(build(hard=lambda i: i == 1, temporary_promotion_strategy=promote_none()))
print("hard cut after position 1, promote_all():")
show_eqns(build(hard=lambda i: i == 1, temporary_promotion_strategy=promote_all()))
print("soft cut after position 1, retain_none():")
show_eqns(build(soft=lambda i: i == 1, soft_split_retainment_strategy=retain_none()))
print("soft cut after position 1, retain_all():")
show_eqns(build(soft=lambda i: i == 1, soft_split_retainment_strategy=retain_all()))
print("hard cut after position 3, the last one:")
show_eqns(build(hard=lambda i: i == 3))
```

```text
unsplit:
  loop 0: x0 = u*v + 1, a = x0**2, b = a + x0**3, c = b*x0
  tile temporaries: []
hard cut after position 1, promote_none():
  loop 0: x0 = u*v + 1, a = x0**2
  loop 1: x0 = u*v + 1, b = a + x0**3, c = b*x0
  tile temporaries: ['a']
hard cut after position 1, promote_all():
  loop 0: x0 = u*v + 1, a = x0**2
  loop 1: b = a + x0**3, c = b*x0
  tile temporaries: ['a', 'x0']
soft cut after position 1, retain_none():
  loop 0: x0 = u*v + 1, x0_ss0 = u*v + 1, a = x0**2, b = a + x0_ss0**3, c = b*x0_ss0
  tile temporaries: []
soft cut after position 1, retain_all():
  loop 0: x0 = u*v + 1, a = x0**2, b = a + x0**3, c = b*x0
  tile temporaries: []
hard cut after position 3, the last one:
  loop 0: x0 = u*v + 1, a = x0**2, b = a + x0**3, c = b*x0
  tile temporaries: []
```

What happened in each case:

- **Hard cut, `promote_none()`**: `x0` is needed on both sides, so it is recomputed in each loop. `a` crosses the
  hard cut, so it becomes a tile temporary, as it would with a manual split.
- **Hard cut, `promote_all()`**: the promotion strategy allows promoting `x0`, so it is computed once, in the first
  loop that needs it, and carried to the second loop as a tile temporary.
- **Soft cut, `retain_none()`**: the soft cut is merged back into one loop. The retainment strategy forgets `x0` at
  the cut, so the merge recomputes it as `x0_ss0` for the equations after the cut. No tile temporaries.
- **Soft cut, `retain_all()`**: the merge keeps `x0`, so the result is the unsplit loop.
- **After the last position**: the trailing piece is empty and is dropped.

At the post-population locus, positions are in the post-CSE order, before `merge_soft_splits`. A merged loop is
rebaked with `ordering_fn`, so after a soft cut the final order need not be the order that was cut. If CSE is off
(`do_cse=False`), the post-population locus cuts the pre-CSE baked order instead, and there is nothing to promote.

---

## 5. What the engine guarantees at a cut

All cuts, at every locus, go through one primitive, `EqnComplex.refine`. It splits each existing list into ordered
pieces and builds a new list from each piece. The guarantees below hold for every locus unless a locus is named.

### Every locus

- **Pieces keep the order that was cut.** Each new list is rebaked with an order-preserving function over its piece
  (with dependency repair, which never has to move anything when the cut order already respects dependencies). So the
  order the predicates saw is the order CSE sees (pre-population locus) or the final order (post-population locus,
  unless a soft cut is merged).
- **No backward dependencies.** No auto hard cut leaves a piece reading something that a later piece computes (see
  "The backward-read guard" below).
- **Parameters are repartitioned.** Each piece gets the grid spacing parameters, the parameters its equations read,
  and the parameters its recipe equations registered.
- **Manual splits are kept**, and their custom loop annotations too (section 4).
- **Each list keeps its ordering functions.** A new list inherits its source list's `ordering_fn`.

### Early and pre-population loci

These cuts happen before CSE. CSE then runs on the new lists exactly as on lists made by manual splits, so temporary
promotion, tile temporaries and soft split merging work as they always have.

A pull-out temporary that a pre-population hard cut separates from its readers becomes a tile temporary. In a Cactus
thorn it gets a centering inferred from its right-hand side, the same way CSE temporaries get one.

### Post-population locus: closure assignment

After CSE, a list already contains its Local CSE temporaries, and one temporary may be read on both sides of a cut.
The list is therefore not simply sliced. Each piece is assigned by **closure**:

- The *roots* of a piece are the equations in its slice that are not Local CSE temporaries (or existing tile
  temporaries).
- The piece gets its roots, plus every Local CSE temporary that the roots need, transitively.

The consequences:

- A Local temporary needed only after a cut **moves** there.
- A Local temporary needed on both sides of a cut is **recomputed** in each loop that needs it (duplicated).
- A temporary that no piece needs is dropped, so dead copies never appear.
- A borrowed temporary is placed just before its first reader in the piece.

**Only hard cuts promote.** This follows the paper's rule that a soft split generates no tile temporaries. A Local
temporary needed by pieces separated only by soft cuts is always duplicated, and `merge_soft_splits` then decides,
through the retainment strategy, whether the merged loop keeps one copy or forgets it and recomputes it. When the
pieces that need a temporary span a hard cut, the temporary promotion strategy that global CSE used decides: if it
classifies the temporary as Tile (e.g. `promote_all()`), the temporary is computed once, in the first piece that needs
it, and becomes a tile temporary. Under `promote_none()` it is duplicated. A Local temporary that CSE placed in several
lists of the function is never promoted.

**Existing tile temporaries move to their first reader.** A temporary that CSE already made a tile temporary is never
duplicated. It moves to the first piece of its list that reads it, so no loop computes a tile temporary that nothing in
it reads.

**Named symbols stay put.** Recipe equations and pull-out temporaries stay in their slices and are never duplicated.
One that crosses a hard cut becomes a tile temporary, as with manual splits.

### The cut guards

Before making the cuts at any locus, the engine computes the **cut hazards** of every position and refuses the cuts
that would change what the code computes. There are two kinds:

| Hazard        | A cut at position p would ...                                                          | Refuses     | Warning says                                               |
|---------------|-----------------------------------------------------------------------------------------|-------------|------------------------------------------------------------|
| overwrite     | separate a write of `X'` (at or before p) from a later read of `X` (after p)            | hard and soft | `which would separate X' from a later read of X`         |
| backward read | leave an element at or before p reading a symbol that an element after p writes        | hard only   | `which would put a read of b before its write` (or `reads of a2, b ... their writes`) |

The predicates are still asked at refused positions, so probing and tuning are unaffected, but a split they request
there is ignored with one warning per function. When both kinds apply, the warning joins both parts with "and".

#### The overwrite guard

`X' = overwrite(X)` writes a new value into the storage of `X`. If a cut put the write of `X'` in one loop and a later
read of `X` in a later loop, the later loop would read the new value instead of the old one. The engine therefore
**refuses** any position at which a cut would separate a writer of `X'` (or `X''`, ...) from a later reader of an
earlier version. At the post-population locus, a reader also counts as reading `X` through any Local temporary it
reads, since the closure assignment may move or duplicate that temporary (e.g. a CSE temporary `x0 = X`).

For example, with a hard split requested at every position:

```python
for locus in (SplitLocus.Early, SplitLocus.PrePopulation, SplitLocus.PostPopulation):
    gf = ThornDef("DOCS", f"GUARD{locus.name.upper()}")
    X, Y, a, u = [gf.decl(s, []) for s in ("X", "Y", "a", "u")]
    Xp, Yp = gf.overwrite(X), gf.overwrite(Y)
    fun = gf.create_function("enforce", ScheduleBin.Evolve, auto_split_locus=locus,
                             auto_hard_split_predicate=lambda i: True)  # ask for a cut everywhere
    fun.add_eqn(a, u * 2 + 1)      # call 0
    fun.add_eqn(Xp, X / (X + Y))   # call 1
    fun.add_eqn(Yp, Y / (X + Y))   # call 2
    gf.bake(do_cse=True, ordering_fn=pre_population_order)
    print(locus.name)
    show(fun)
```

```text
Warning: enforce: ignoring the auto splits at positions [1] of the Early locus, which would separate X' from a later read of X.
Early
  loop 0: ['a']
  loop 1: ['x0', 'x1', 'x2', "X'", "Y'"]
Warning: enforce: ignoring the auto splits at positions [1] of the PrePopulation locus, which would separate X' from a later read of X.
PrePopulation
  loop 0: ['a']
  loop 1: ['x0', 'x1', 'x2', "X'", "Y'"]
Warning: enforce: ignoring the auto splits at positions [4] of the PostPopulation locus, which would separate X' from a later read of X.
PostPopulation
  loop 0: ['a']
  loop 1: ['x0', 'x1', 'x2', "X'", "Y'"]
```

After CSE the loop is `x0 = X, x1 = Y, x2 = 1/(x0 + x1), X' = x0*x2, Y' = x1*x2`. At the early and pre-population
loci, position 1 is between `X'` and `Y'`, and position 2 is after the last element, so it does nothing. At the
post-population locus, the order is `a, x0, x1, x2, X', Y'`: position 4 (between `X'` and `Y'`) is refused, because
`Y'` reads `X` through `x2` and `x0`. The cuts at positions 1 to 3 are made, but the Local temporaries `x0`, `x1` and
`x2` move to their reader `X'`, so those pieces end up empty and are dropped.

At the early locus without an early ordering function, this refusal is new: the old in-`add_eqn` splitting made such
cuts. For a hard cut, that generated code which read the already overwritten variable. For a soft cut (which the
merge rejoins) the refusal is merely conservative.

#### The backward-read guard

At the early locus without an early ordering function, the elements are the `add_eqn` calls in recipe order, and
nothing forces that order to respect the dependencies between calls. A recipe may compute `a = 2*b` in call 0 and
`b` in call 1 (the bake's dependency repair puts `b` first within a loop), and two calls may depend on each other
(call 2 computes `vD1 = 2*a2`, call 3 computes `a2 = 3*vD0 + a`). A **hard** cut between such calls would put the read
in an earlier loop than the write, so it is refused. The old in-`add_eqn` splitting crashed there with an internal
`AssertionError`.

A **soft** cut there is still made: `merge_soft_splits` rejoins the loops and orders the merged loop with dependency
repair, so nothing is read before it is written, and the generated code is the same as before the split loci existed.
With an early ordering function the question never arises, because the calls are ordered topologically and mutually
dependent calls form one super-group (one position). The other loci cut a baked order, which already respects
dependencies.

```python
def build(name, early_ordering_fn=None, **create_kwargs):
    gf = ThornDef("DOCS", name)
    v = gf.decl("v", [li])
    a, a2, b, src = [gf.decl(s, []) for s in ("a", "a2", "b", "src")]
    fun = gf.create_function("fn", ScheduleBin.Evolve, **create_kwargs)
    fun.add_eqn(a, b * 2)                       # call 0: reads b, which call 1 writes
    fun.add_eqn(b, src * 3)                     # call 1
    fun.add_eqn(v[li], [src, a2 * 2, src * 3])  # call 2: reads a2, which call 3 writes
    fun.add_eqn(a2, v[l0] * 3 + a)              # call 3: reads vD0, which call 2 writes
    gf.bake(do_cse=False, ordering_fn=pre_population_order, early_ordering_fn=early_ordering_fn)
    return fun


hard = Recorder(lambda i: True)
fun = build("BACKHARD", auto_hard_split_predicate=hard)
print("hard cuts everywhere, no early_ordering_fn: positions", hard.calls)
show(fun)
fun = build("BACKSOFT", auto_soft_split_predicate=lambda i: i == 0)
print("soft cut after position 0, no early_ordering_fn:")
show(fun)
hard = Recorder(lambda i: True)
fun = build("BACKORDERED", add_eqn_order([0, 1, 2, 3]), auto_hard_split_predicate=hard)
print("hard cuts everywhere, early_ordering_fn=add_eqn_order([0, 1, 2, 3]): positions", hard.calls)
show(fun)
```

```text
Warning: fn: ignoring the auto splits at positions [0, 2] of the Early locus, which would put reads of a2, b before their writes.
hard cuts everywhere, no early_ordering_fn: positions [0, 1, 2, 3]
  loop 0: ['b', 'a']
  loop 1: ['vD0', 'a2', 'vD1', 'vD2']
soft cut after position 0, no early_ordering_fn:
  loop 0: ['b', 'a', 'vD0', 'a2', 'vD1', 'vD2']
hard cuts everywhere, early_ordering_fn=add_eqn_order([0, 1, 2, 3]): positions [0, 1, 2]
  loop 0: ['b']
  loop 1: ['a']
  loop 2: ['vD0', 'a2', 'vD1', 'vD2']
```

In the first case, position 0 (between `a` and the `b` it reads) and position 2 (between the two mutually dependent
calls) are refused, position 1 is cut, and position 3 is after the last call, so it does nothing. In the second case
the soft cut is made and then merged away. In the third case, the early ordering function puts call 1 before call 0
and makes calls 2 and 3 one super-group, so there are only three positions, and every hard cut is made.

A manual `split_loop()` between a read and its write is still the author's responsibility, but it now fails with a
clear `DslException` instead of the `AssertionError` (see "Validation errors" below). A manual `soft_split()` there
works, as it always has.

### Validation errors

After a pre-population or post-population cut, the engine checks that the cut did not change the meaning of the
function and raises a `DslException` if it did. (Early cuts are made before the first bake, like manual splits, so
there is nothing to compare them with and they are not checked.) The last row of the table is different: it comes
from the check every bake makes on its loops, whatever made the split. In a tuning run, a trial that raises is scored
`-inf` and the search moves away from it.

| Message                                                                                                                 | Meaning                                                                                                                                                                                                                                                              |
|-------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `After splitting, the inferred write region of 'X' in loop i (cut from loop s) changed from A to B.`                    | A piece would write the output `X` over a different region (Interior, Boundary, Everywhere) than the unsplit list did. The engine seeds each piece with what it knew was analytic to prevent this, so this points to an engine problem. Report it, and avoid the position. |
| `After splitting, loop i, which runs over R1, reads temporary 't', which loop j only computes over R2.`                 | The cut made `t` cross into another loop, as a tile temporary, but the loop that computes it runs over a smaller region than a loop that reads it and writes grid variables. The reading loop would use values that were never computed. Choose another position.         |
| `After splitting, loop i reads 'X', which loop j (cut from the same loop s) has already overwritten.`                   | A backstop for the overwrite guard. Auto splits never cut there, so it points to an engine problem.                                                                                                                                                                  |
| `'b' is written in loop 1 after it is read in loop(s) [0]: a loop cannot read a value that a later loop computes, ...` | Raised for any split, but in practice by a manual `split_loop()` placed between an equation and a later one it reads (auto hard cuts are refused there). Move the manual split, or make it a `soft_split()`. |

The region check is a heuristic that predicts the loop regions the generator will infer. Here is a cut it rejects:
`tmp = sin(x)` is analytic, so the unsplit function writes it (and `b`, which reads it) everywhere. A hard cut after
`a` moves the read of `tmp` into a loop that writes `b` everywhere, but `tmp` is now a tile temporary computed only in
a loop that runs over the interior:

```python
for locus in (SplitLocus.PrePopulation, SplitLocus.Early):
    gf = ThornDef("DOCS", f"VALIDTILE{locus.name.upper()}")
    x = mk_symbol("x")
    t, a, b, src = gf.decl("tmp", []), gf.decl("a", []), gf.decl("b", []), gf.decl("src", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve, auto_split_locus=locus,
                             auto_hard_split_predicate=lambda i: i == 1)
    fun.add_eqn(t, sin(x))
    fun.add_eqn(a, src * 2)
    fun.add_eqn(b, t * 2)
    try:
        gf.bake(do_cse=False, ordering_fn=recipe_order)
        print(locus.name, "accepted:")
        for i, el in enumerate(fun.eqn_complex.eqn_lists):
            print(f"  loop {i} writes", {str(k): r.name for k, r in el.write_decls.items()})
    except Exception as e:
        print(locus.name, "refused:", e)
```

```text
PrePopulation refused: After splitting, loop 1, which runs over Everywhere, reads temporary 'tmp', which loop 0 only computes over Interior.
Early accepted:
  loop 0 writes {'a': 'Interior', 'tmp': 'Everywhere'}
  loop 1 writes {'b': 'Interior'}
```

The same cut at the early locus is accepted, because early cuts are not validated: `b` is then written only over the
interior, exactly as it would be with a manual `split_loop()` after `a`.

A manual `split_loop()` that puts a read before its write:

```python
gf = ThornDef("DOCS", "MANUALBACK")
a, b, src = gf.decl("a", []), gf.decl("b", []), gf.decl("src", [])
fun = gf.create_function("fn", ScheduleBin.Evolve)
fun.add_eqn(a, b * 2)
fun.split_loop()
fun.add_eqn(b, src * 3)
try:
    gf.bake()
except Exception as e:
    print(type(e).__name__, e)
```

```text
DslException 'b' is written in loop 1 after it is read in loop(s) [0]: a loop cannot read a value that a later loop computes, so no split (e.g., split_loop()) may separate the read from the equation that writes 'b'.
```

### Other errors you may see

- **`Equation for 'a' is already defined in this loop. ...`** A LHS may be assigned only once between manual splits.
  Previously an auto split predicate could separate two assignments to the same LHS as the equations were
  added. Predicates are now evaluated at bake time, so they cannot; use a manual `split_loop()` there.

  ```python
  gf = ThornDef("DOCS", "DUPLHS")
  a, src = gf.decl("a", []), gf.decl("src", [])
  fun = gf.create_function("fn", ScheduleBin.Evolve, auto_hard_split_predicate=lambda i: i == 0)
  fun.add_eqn(a, src)
  try:
      fun.add_eqn(a, src * 2)
  except Exception as e:
      print(e)
  ```

  ```text
  Equation for 'a' is already defined in this loop. A LHS may be assigned only once between split_loop()/soft_split() calls; note that auto split predicates are evaluated at bake time, not as equations are added, so they cannot separate two assignments to the same LHS.
  ```

- **`add_eqn_order: ...`** and **`The bake options name unknown functions ...`**: see sections 3 and 1.

---

## 6. Tuning

[TUNING-QUICKSTART.md](TUNING-QUICKSTART.md), section 5, has the full walkthrough of the split tuners. What matters
here:

- The split tuners (`CombinatorialSplitTuner`, `CutPositionSplitTuner`) provide the out-params
  `auto_hard_split_predicate`, `auto_soft_split_predicate` and `auto_split_locus`, plus whichever of
  `early_ordering_fn`, `pre_population_ordering_fn` and `ordering_fn` you give them. The recipe must read each of them
  with `get_tuning_param` / `get_optional_tuning_param` and pass it to `create_function` or `bake`, as Z4c does
  (section 1).
- **The number of positions `N` is probed.** Before the search, the recipe runs once with placeholder predicates that
  never split, and the engine reports how many positions each predicate is asked about. `N` depends on the recipe,
  the locus, and the ordering functions (an `early_ordering_fn` can merge groups into super-groups), which is why the
  locus and the ordering functions are provided while probing.
- `CombinatorialSplitTuner` has two params per position, so it is practical only at the early locus. Use
  `CutPositionSplitTuner(n_cuts, locus=...)` at the other two.

You can probe a recipe directly, e.g. to see `N` at each locus. Here is the three-call recipe of section 4 as a
recipe file that reads the knobs the way Z4c does:

```python
# probe_recipe_demo.py
from EinsteinEngine import *
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornDef

gf = ThornDef("DOCS", "PROBEDEMO")
a, b, c, u, v = [gf.decl(s, []) for s in ("a", "b", "c", "u", "v")]
fun = gf.create_function(
    "f", ScheduleBin.Evolve,
    auto_hard_split_predicate=get_tuning_param('auto_hard_split_predicate', None),
    auto_soft_split_predicate=get_tuning_param('auto_soft_split_predicate', None),
    auto_split_locus=get_optional_tuning_param('auto_split_locus', SplitLocus.Early),
)
fun.add_eqn(a, pull_out(sin(u) * cos(v)) + (u * v + 1) ** 2)
fun.add_eqn(b, (u * v + 1) ** 3 + a)
fun.add_eqn(c, (u * v + 1) * b)
gf.bake(do_cse=True, functions={"f": {
    "early_ordering_fn": get_optional_tuning_param('early_ordering_fn', None),
}})
```

Probing it at each locus:

```python
from EinsteinEngine.tuning.probe import probe_recipe

for locus in SplitLocus:
    probe = probe_recipe("probe_recipe_demo.py", ["auto_hard_split_predicate"], {"auto_split_locus": locus})
    print(locus.name, dict(probe))
```

```text
Early {'auto_hard_split_predicate': 3}
PrePopulation {'auto_hard_split_predicate': 4}
PostPopulation {'auto_hard_split_predicate': 7}
```

A probe fails if the recipe never reads one of the names it was given (a recipe that ignored the locus or an ordering
function would be tuned against the wrong positions). The message names the unread out-params and tells you to read
them with `get_optional_tuning_param`.

### The probe file (sidecar) and its fingerprint

The probe result is saved next to the checkpoint as `<checkpoint>.probe.json`. It holds the counts, plus a stable
description (a *fingerprint*) of the locus and ordering functions the tuner provided, such as
`"early_ordering_fn": "add_eqn_order([8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11])"`. Resuming a search, or running
`generate_best`, probes again and refuses to continue if the count or a description changed, because the checkpoint's
coordinates only mean something for the positions they were recorded against. What the fingerprint can and cannot
see is described in TUNING-QUICKSTART.md, "The probe file". With `rank_by_post_population`, the sidecar also records
the order the engine derived, and a changed derived order is refused too (section 7).

### Old checkpoints

A checkpoint recorded before the split loci existed has 1-based split params (`split_1..N`), and possibly more
positions than the recipe has (the old Z4c tuner declared 15; the Z4c RHS has 13 `add_eqn` calls). It is refused with
a message that says so, rather than replayed wrongly. Convert it once into a new file with
`scripts/shift_checkpoint_indices.py`:

```bash
python scripts/shift_checkpoint_indices.py old.jsonl new.jsonl
python scripts/shift_checkpoint_indices.py old.jsonl new.jsonl --n-positions 13   # also drop positions >= 13
```

For a three-position `CombinatorialSplitTuner`, a 1-based entry and its conversion look like this:

```text
old.jsonl: {"target": -5.0, "params": {"split_1": 2, "split_2": 0, "split_3": 1, "soft_retain_percentile_3": 0.25}}
  refused: old.jsonl:1 has params the experiment does not declare: split_3, soft_retain_percentile_3. It was probably recorded against a different tuner or recipe. ...
new.jsonl: {"target": -5.0, "params": {"split_0": 2, "split_1": 0, "split_2": 1, "soft_retain_percentile_2": 0.25}}
```

---

## 7. Deriving an early order automatically: `rank_by_post_population`

`add_eqn_order([...])` needs you to know a good order of the `add_eqn` calls. `rank_by_post_population(fn)` has the
engine work one out: it orders the calls by where `fn` places their equations **after CSE**. On Z4c,
`rank_by_post_population(prioritize_rare_symbols)` derives exactly the order Steve derived offline and hardcoded,
`[8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11]`, and the tuned build generated with it is byte-identical to his.

```python
gf.bake(..., functions={"my_rhs": {"early_ordering_fn": rank_by_post_population(prioritize_rare_symbols)}})
```

### The chicken-and-egg problem

The order worth imitating is the post-CSE order that a good ordering function gives. That is the order that
matters for register pressure, and it exists only once CSE has added its temporaries and rewritten the right-hand
sides in terms of them. But the early locus runs **first**, before pull-out, before the pre-CSE bake, and before CSE:

```
   early locus ---> pull-out ---> pre-CSE bake ---> global CSE ---> post-CSE order
       ^                                                                  |
       |                                                                  |
       +------- the ranking the early locus needs is computed here -------+
```

The post-CSE order depends on CSE, CSE depends on the order before it, and that order is what the early locus is
about to decide. So the engine bakes twice: first a **trial bake** that it throws away, just to learn the post-CSE
order, then the real bake with an early order derived from it.

### The pipeline with a trial bake

```
 DslFrontend.bake(**options)
   |
   |  _derive_early_orders: only if some function has early_ordering_fn=rank_by_post_population(fn)
   |  +---------------------------------------------------------------------------------+
   |  | 1. TRIAL BAKE, on throwaway copies of EVERY function of the thorn               |
   |  |    a. snapshot, for each ranked function, which equations each add_eqn call     |
   |  |       produced (eqn_origin before the bake)                                     |
   |  |    b. early-bake every function with its real bake options, except that         |
   |  |         - a ranked function has early_ordering_fn=None,                         |
   |  |           pre_population_ordering_fn=None and ordering_fn=fn                    |
   |  |         - no function has auto split predicates (manual splits are kept)        |
   |  |    c. global CSE over all functions (if do_cse); rebake the ranked functions    |
   |  |       after CSE                                                                 |
   |  |    d. read each ranked list's order  ==> the post-CSE positions                 |
   |  |    e. restore everything (also if the trial raised)                             |
   |  | 2. RANK: in each list, order the add_eqn calls by the median post-CSE position  |
   |  |    of their own equations  ==> the derived order, e.g. [8, 1, 0, 2, ...]        |
   |  |    (printed in one line, and reported to the tuning probe)                      |
   |  | 3. replace the option: early_ordering_fn := add_eqn_order(<derived order>)      |
   |  +---------------------------------------------------------------------------------+
   |
   |  REAL BAKE: the pipeline of section 1, exactly as with that explicit add_eqn_order:
   |    step 1 (early locus): order_groups ranks the groups by it, repairs dependencies,
   |    merges cycles into super-groups, and the early auto splits run; then pull-out,
   |    the pre-CSE bake, CSE, the other loci, merges, ...
   v
 generator
```

### Step 1: the trial bake

The trial bakes **the whole thorn**, every function, up to and including global CSE, not just the function being
ranked. That is because CSE is global: it finds common subexpressions across all functions, so what a function's
post-CSE list looks like depends on the other functions. Baking Z4c's `z4c_rhs` alone gives 759 equations after CSE
instead of 1068, and ranks the calls as `[1, 8, 2, 0, 3, 5, 10, 4, 7, 9, 11, 12, 6]`, which is not Steve's order.
Baking the whole thorn gives his order.

What the trial does and does not do:

- **Each ranked function** is baked as a plain `ordering_fn=fn` bake: no early ordering function, no pre-population
  ordering function, and `fn` orders both the pre-CSE bake and the post-CSE rebake.
- **Every other function** is baked with its own bake options, as the real bake will bake it. (Steve's offline script
  forced `prioritize_rare_symbols` on every function instead. On Z4c, the trial's `z4c_rhs` list has 1068 equations
  after CSE, the offline script's 1061; the derived order is the same.)
- **No auto splits.** No function has its auto split predicates during the trial, so the trial is unsplit (manual
  `split_loop()` / `soft_split()` calls still apply).
- **The CSE side effects are skipped.** The trial does not declare global temporaries, create their synthetic
  functions, or infer centerings, and it rebakes only the ranked functions after CSE. None of that can change a ranked
  function's post-CSE order.
- **The positions** are each list's order right after the post-CSE rebake: before post-population cuts (there are
  none in the trial), soft split merges, splitmaxxing and temporary recycling. For an unsplit function without those,
  that is its final order.

### Step 2: the ranking

For each list of a ranked function, the engine takes the positions of the equations **each `add_eqn` call itself
produced**, in that list's post-CSE order, and ranks the calls by the median of those positions:

- **Only the call's own equations count**: its scalar components, snapshotted before the trial bake. The pull-out
  temporaries made from them later do not count (even though they inherit the call's origin), and neither do CSE
  temporaries (which have none). They still take up positions in the order; they are just not averaged in. This is
  what Steve's offline script did, and it matters in the pull-out example below.
- **Ties go to the lower call index.**
- **A call none of whose equations is in the order** goes last in its list, in call-index order. A call that
  produced no equation at all is sorted in with those of the last list, as in Steve's script, which gave all of them
  the median `inf`. The result is always a permutation of `0..(number of add_eqn calls - 1)`.
- **Lists are ranked separately** and concatenated. Only the order of the calls within a list matters, since the
  early locus never moves anything across a manual split.

### Step 3: the real bake

The option is replaced by `add_eqn_order(<derived order>)`, and from there **nothing differs from passing that list
yourself**. In particular, the derived order is a *ranking*, not necessarily the final group order: the early locus
still repairs dependencies (a group never comes before a group it reads, and groups in a dependency cycle merge into a
super-group, section 3). The real bake uses the function's real `ordering_fn`, not the `fn` the trial ranked with, for
everything after the early locus.

### Worked example: the trial, the ranking and the real bake

Three calls, baked without CSE (so "post-CSE" is just the baked order), ranked by `lexicographical_order`. You can
reproduce the trial by hand: it is a plain bake with `ordering_fn=lexicographical_order`.

```python
def build(name, **bake_opts):
    gf = ThornDef("DOCS", name)
    v = gf.decl("v", [li])
    u, a, src = gf.decl("u", []), gf.decl("a", []), gf.decl("src", [])
    fun = gf.create_function("fn", ScheduleBin.Evolve)
    fun.add_eqn(u, src)                              # call 0
    fun.add_eqn(v[li], [src * 2, src * 3, src * 4])  # call 1
    fun.add_eqn(a, u + 1)                            # call 2
    gf.bake(do_cse=False, **bake_opts)
    return fun


print("the trial, reproduced by hand:")
show(build("RANKBYHAND", ordering_fn=lexicographical_order))
print("rank_by_post_population(lexicographical_order):")
show(build("RANKED", ordering_fn=pre_population_order,
           early_ordering_fn=rank_by_post_population(lexicographical_order)))
print("add_eqn_order([0, 2, 1]):")
show(build("EXPLICIT", ordering_fn=pre_population_order, early_ordering_fn=add_eqn_order([0, 2, 1])))
```

```text
the trial, reproduced by hand:
  loop 0: ['u', 'a', 'vD0', 'vD1', 'vD2']
rank_by_post_population(lexicographical_order):
fn: rank_by_post_population(EinsteinEngine.intermediate.eqn_ordering.lexicographical_order) derived the early order [0, 2, 1] from a trial bake (0.0 s).
  loop 0: ['u', 'a', 'vD0', 'vD1', 'vD2']
add_eqn_order([0, 2, 1]):
  loop 0: ['u', 'a', 'vD0', 'vD1', 'vD2']
```

In the trial order `u, a, vD0, vD1, vD2`, call 0 (`u`) is at position 0, call 2 (`a`) at position 1, and call 1 (`vD0`,
`vD1`, `vD2`) at positions 2, 3 and 4, median 3. So the derived order is `[0, 2, 1]`, and the real bake is exactly
the explicit `add_eqn_order([0, 2, 1])` bake. (The real bake orders with `pre_population_order` here, so its final
order shows the group order.)

### Worked example: the ranking is not the final order

`a` reads `wD0`. In the lexicographic trial `a` lands right after `wD0`, at position 1, while the median of `w` is
position 2, so `a`'s call ranks first. The real bake still cannot put `a` before the group it reads:

```python
gf = ThornDef("DOCS", "RANKREPAIR")
w = gf.decl("w", [li])
a, src = gf.decl("a", []), gf.decl("src", [])
fun = gf.create_function("fn", ScheduleBin.Evolve)
fun.add_eqn(w[li], [src, src * 2, src * 3])  # call 0
fun.add_eqn(a, w[l0] * 2)                    # call 1
gf.bake(do_cse=False, ordering_fn=pre_population_order,
        early_ordering_fn=rank_by_post_population(lexicographical_order))
show(fun)
```

```text
fn: rank_by_post_population(EinsteinEngine.intermediate.eqn_ordering.lexicographical_order) derived the early order [1, 0] from a trial bake (0.0 s).
  loop 0: ['wD0', 'wD1', 'wD2', 'a']
```

### Worked example: pull-out temporaries do not count

The trial order is `pull_out_0, qa, qb, qc` (lexicographic; `qc` reads `pull_out_0` anyway). Counting only the
equation each call produced, the medians are 3 (call 0, `qc`), 2 (call 1, `qb`) and 1 (call 2, `qa`), so the order is
`[2, 1, 0]`. If the pull-out temporary counted for call 0, its median would be 1.5 and the order `[2, 0, 1]`:

```python
gf = ThornDef("DOCS", "PULLOUTCOUNT")
qa, qb, qc, src = gf.decl("qa", []), gf.decl("qb", []), gf.decl("qc", []), gf.decl("src", [])
fun = gf.create_function("fn", ScheduleBin.Evolve)
fun.add_eqn(qc, pull_out(src * src) * 2)  # call 0
fun.add_eqn(qb, src + 1)                  # call 1
fun.add_eqn(qa, src * 3)                  # call 2
gf.bake(do_cse=False, ordering_fn=pre_population_order,
        early_ordering_fn=rank_by_post_population(lexicographical_order))
show(fun)
```

```text
fn: rank_by_post_population(EinsteinEngine.intermediate.eqn_ordering.lexicographical_order) derived the early order [2, 1, 0] from a trial bake (0.0 s).
  loop 0: ['qa', 'qb', 'pull_out_0', 'qc']
```

### Isolation: the trial leaves no trace

The real bake is identical to a bake with the explicit `add_eqn_order`, **temporary names included**:

- The trial works on copies of the functions' equation lists, annotations and flags, so the real lists (their
  origins, orders and sticky ordering functions) are untouched.
- The thorn's automatic name counter is restored, so the real pull-out temporaries get the same names. (CSE names its
  temporaries from `x0` afresh in every CSE call.)
- The CSE results (tile and global temporaries, temporary kinds, the promotion predicate) and the function table are
  restored, and the skipped CSE hooks are the ones that would have changed declarations and centerings.
- **No auto split predicate is called during the trial**, so a `Recorder` or a tuner's predicate sees exactly the
  positions of the real bake, and the probe gets no split-position report from the trial.
- Everything is restored even if the trial raises; the error then propagates.
- The trial's progress messages (and its verbose output, with `EINSTEINENGINE_VERBOSE`) are silenced. It prints one
  line per ranked function (the "derived the early order" line above). Warnings are not silenced, so a warning the
  trial triggers would appear twice; none does on Z4c.
- **Your callables run twice per bake.** The ordering functions and the temporary promotion strategy you pass run in
  the trial and again in the real bake. This matters only for callables with side effects (e.g. one that counts its
  calls).
- **One trial per bake serves every function that uses `rank_by_post_population`.** You can pass it bake-wide
  (`bake(early_ordering_fn=rank_by_post_population(fn))`): unlike `add_eqn_order`, which names one function's calls,
  it derives each function's own order.

### Restrictions

- **`fn` must be deterministic.** Bayesian optimization (a function or partial whose name contains `bayesian`) is
  refused: its result changes from run to run, which would break the tuner's check that the order is unchanged, and it
  would run on the whole post-CSE list in an extra bake. `rank_by_post_population` cannot be nested either.
- **It is only valid as `early_ordering_fn`.** It has the signature of an ordering function so that it type-checks
  there (bake options, tuner out-params), but it cannot order anything itself: calling it, passing it as `ordering_fn`
  or `pre_population_ordering_fn`, or passing it to `_early_bake` directly raises a `DslException`.
- Its `repr` names the rule and the function, e.g.
  `rank_by_post_population(EinsteinEngine.intermediate.eqn_ordering.prioritize_rare_symbols)`.

```python
for bad in (lambda: rank_by_post_population(bayesian_optimization),
            lambda: rank_by_post_population(rank_by_post_population(prioritize_rare_symbols))):
    try:
        bad()
    except Exception as e:
        print(e)
```

```text
rank_by_post_population(EinsteinEngine.intermediate.eqn_ordering.bayesian_optimization): Bayesian optimization cannot rank the add_eqn calls. It is nondeterministic, so the derived order (which tuners record and compare) would change from run to run, and it would run on the whole post-CSE list in an extra bake. Rank with a deterministic function (e.g. prioritize_rare_symbols) instead.
rank_by_post_population cannot be nested: the trial bake needs a plain ordering function.
```

### Cost

The trial is about one extra bake of the thorn up to CSE, and it is paid **on every bake**: every generation of the
recipe, every tuning trial, the probe at the start of a tuning run, and `generate_best` (which probes and then
generates, so it pays twice). On Z4c the trial takes about 16 to 18 s. One unsplit Z4c generation took about 62 s with
the explicit order and 79 s with the derived rule, about 28% more (indicative timings on a shared machine). There is no
cache across runs, because the order depends on the recipe. If you want the speed, derive the order once, then pin it
with `add_eqn_order([...])`.

### Tuning: the derived order in the probe and the sidecar

With `early_ordering_fn=rank_by_post_population(fn)` in a split tuner, the probe records which order was derived,
because the tuner's positions only mean something in that order: a split coordinate means "cut after the i-th group
*in this order*". The fingerprint of the tuner's `early_ordering_fn` names only the rule, and the order the rule
yields depends on the recipe (its equations, its other functions, its bake options), so the order is recorded
separately:

```json
{
  "counts": {"auto_hard_split_predicate": 13},
  "probe_params": {"auto_split_locus": "SplitLocus.Early",
                   "early_ordering_fn": "rank_by_post_population(EinsteinEngine.intermediate.eqn_ordering.prioritize_rare_symbols)"},
  "derived_orders": {"z4c_rhs": [8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11]}
}
```

`probe_recipe` returns it as `derived_orders`. Here it is for the demo recipe of section 6, whose calls form a chain,
so any ordering function ranks them `[0, 1, 2]` (dependency repair keeps a chain in order):

```python
from EinsteinEngine.tuning.probe import probe_recipe

probe = probe_recipe("probe_recipe_demo.py", ["auto_hard_split_predicate"],
                     {"auto_split_locus": SplitLocus.Early,
                      "early_ordering_fn": rank_by_post_population(prioritize_rare_symbols)})
print(dict(probe), dict(probe.derived_orders))
```

```text
f: rank_by_post_population(EinsteinEngine.intermediate.eqn_ordering.prioritize_rare_symbols) derived the early order [0, 1, 2] from a trial bake (0.0 s).
{'auto_hard_split_predicate': 3} {'f': (0, 1, 2)}
```

The mismatch rule is the same as for the counts: when the recipe changes so that the derived order changes, resuming a
checkpoint that already has entries (or running `generate_best` on it) is refused, with a message naming the old and
new order. An empty checkpoint just gets a new sidecar.

**Switching a checkpoint between the explicit list and the derived rule is refused**, even when the order is the same,
because the two fingerprints differ. Here is what the refusal lists, for a checkpoint recorded with the explicit list
and resumed with the derived rule:

```python
from EinsteinEngine.tuning.probe import ProbeResult
from EinsteinEngine.tuning.tuning import ProbeRecord

recorded = ProbeRecord(ProbeResult({"auto_hard_split_predicate": 13}),
                       {"early_ordering_fn": "add_eqn_order([8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11])"})
current = ProbeRecord(ProbeResult({"auto_hard_split_predicate": 13}),
                      {"early_ordering_fn": "rank_by_post_population(EinsteinEngine.intermediate.eqn_ordering.prioritize_rare_symbols)"},
                      {"z4c_rhs": (8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11)})
for difference in recorded.differences(current):
    print(difference)
```

```text
the probe param 'early_ordering_fn' was 'add_eqn_order([8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11])' and is now 'rank_by_post_population(EinsteinEngine.intermediate.eqn_ordering.prioritize_rare_symbols)'
the add_eqn order rank_by_post_population derived for function 'z4c_rhs' was absent and is now [8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11]
```

To switch, start a new checkpoint file (or delete the sidecar and lose the check).

### Z4c

`tuning/z4c_derived_order/tuner.py` now uses the rule:

```python
def get_tuner() -> Tuner:
    return CombinatorialSplitTuner(early_ordering_fn=rank_by_post_population(prioritize_rare_symbols))
```

- For today's recipe it derives `[8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11]`, the order Steve derived offline. The
  engine prints `z4c_rhs: rank_by_post_population(EinsteinEngine.intermediate.eqn_ordering.prioritize_rare_symbols)
  derived the early order [8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11] from a trial bake (16.2 s).`
- With Steve's converted checkpoint, `generate_best` through the derived rule produces code byte-identical to his
  tuned build, and so does the explicit list. That build ran in 4.933 s versus 5.332 s for the recipe order at 128³
  on his runs.
- `get_explicit_order_tuner()` in the same file pins the order with `add_eqn_order(Z4C_STEVE_ORDER)`: no trial bake,
  but it does not follow changes to the recipe, and its sidecar does not match the rule's (see above).

Which one to use: the rule follows the recipe (if you change `z4c_rhs` or the rest of the thorn, the order is derived
again, and a checkpoint recorded under the old order is refused rather than misread). The explicit list is about 17 s
faster per generation and never changes on its own.

---

## 8. Known limitations and pre-existing issues

These predate the split loci, or are deliberate; none of them is caused by the loci unless stated.

- **One loop's order does not respect reader-before-overwriter.** Only the early grouping (section 3) adds the edge
  that keeps a read of `X` before the write of `X'`. The ordering functions that order the equations *within* a loop do
  not, so a loop can compute `X'` before a later read of `X` in the same loop. This affects unsplit functions too. A
  related case: a soft cut that respects the guard, merged with `retain_none()`, recomputes a forgotten `x = X` for the
  later piece, and the rebake of the merged loop can then place `X'` before that load. A manual `soft_split()` there
  does the same.
- **The vanilla F90 generator does not support `overwrite()`.** It declares the overwritten variable both as an input
  and as an output instead of `INOUT`.
- **Positions the backward-read guard refuses still count toward `N`** (early locus without an early ordering
  function; section 5), so a tuner's hard split there has no effect.
- **Early cuts are not validated** (section 5). They behave exactly like manual splits, including write regions that
  change silently.
- **A grid function that crosses a hard cut is carried as a tile temporary of the same name** (e.g. `a` in the
  post-population example of section 4), as with manual splits.
- **`merge_soft_splits` resets custom loop annotations** to their defaults.
- **Bayesian `ordering_fn` at the post-population locus.** If a function has (manual) soft splits, its post-CSE
  rebake is a "fast" one (a merge follows), so a Bayesian `ordering_fn` is replaced by `prioritize_rare_symbols` there.
  The post-population positions are then in that order. (The `SplitLocus` docstring says so too.)
- **`rank_by_post_population` costs a trial bake on every run** (about 17 s for Z4c, section 7), and ranks by the
  order right after the post-CSE rebake, i.e. before soft split merges, splitmaxxing and temporary recycling. Warnings
  raised during the trial are not silenced, so they would be printed twice.
- **Memory across tuning trials.** Class-level caches keep references to every baked function, so memory grows over
  many in-process trials.
- **Whether post-population cuts pay off is untested.** Steve's experiments found that cutting a globally sorted list
  was harmful, and post-population cutting is the most complex code path. The early locus is the one with a measured
  speedup.

---

## 9. FAQ

### Why did my split not happen?

Check, in this order:

1. **Is the predicate asked at the position you think?** Positions are 0-based and mean "after element `i`". A
   predicate written for the old 1-based convention splits one element late. Use a `Recorder` (see "Running the
   examples") to see which positions are queried.
2. **Is the locus the one you think?** The default is `SplitLocus.Early`. A recipe that reads the locus from a tuner
   must pass it to `create_function(..., auto_split_locus=...)`.
3. **Does the order at that locus put the element where you expect?** The positions refer to the order after that
   locus's ordering function has run (and after dependency repair), not to the recipe order.
4. **Did hard win?** The soft predicate is asked only where the hard one declined.
5. **Was it a soft split?** Soft splits never create a separate loop in the generated code: `merge_soft_splits` merges
   them back, and the retainment strategy only decides what is recomputed after the cut. With `retain_all()` a soft
   split leaves no visible trace.
6. **Was the piece empty?** A split after the last position, right before a manual split, or (at the post-population
   locus) between Local temporaries and their only reader produces an empty piece, which is dropped.
7. **Was it refused?** Look for "ignoring the auto splits at positions ..." in the output; the rest of the message
   names the guard (section 5, "The cut guards").
8. **Is the position past `N`?** A predicate is never asked about a position at or beyond the number of elements, so
   a checkpoint with more positions than the recipe has simply never uses the extra ones.

### Why does the tuner say an out-param was "never read"?

While probing, the tuner provides the predicates, the locus and any ordering functions you gave it. If the recipe
never reads one of them, the probe would count the wrong positions, so it stops. Read every provided name in the
recipe with `get_optional_tuning_param(name, default)` and pass it on, as in section 1, to `create_function` (the
predicates and `auto_split_locus`) or to `bake(functions={...})` (the ordering functions). Only reads made before the
probe stops the recipe (at the end of the first bake in which every placeholder predicate was asked) count, so a name
read only for a later thorn counts as never read.

### Why is my old checkpoint refused?

One of:

- It uses 1-based keys (`split_1..N`). Convert it with `scripts/shift_checkpoint_indices.py` (section 6).
- It has more positions than the recipe now has. Add `--n-positions N`.
- It predates the soft retain percentile params. Such a checkpoint cannot be converted.
- The probe file next to it no longer matches: the number of positions changed, the tuner's locus or ordering
  functions changed, or the order `rank_by_post_population` derived changed (because the recipe changed). The message
  lists each difference. Start a new checkpoint, or restore the recipe and tuner.
- It was recorded with `add_eqn_order([...])` and you now use `rank_by_post_population(...)`, or the other way
  around. The fingerprints differ even when the order is the same (section 7), so start a new checkpoint.

### Which locus should I tune?

- **Early** has the fewest positions (13 for Z4c), works with `CombinatorialSplitTuner`, and is the only locus with a
  measured speedup. Combine it with an `early_ordering_fn` to search splits in a different group order (for Z4c,
  `rank_by_post_population(prioritize_rare_symbols)` gives the order of the measured speedup). Start here.
- **PrePopulation** (64 for Z4c) can cut inside a tensor equation and between pull-out temporaries and their readers,
  before CSE decides what to share. Use `CutPositionSplitTuner`.
- **PostPopulation** (1061 for Z4c) cuts the final order, CSE temporaries included, and recomputes or promotes shared
  temporaries at each cut. It is the most flexible and the least proven. Use `CutPositionSplitTuner` with a small
  number of cuts.

### How do I get a good early order?

Three ways, from cheapest to most automatic:

- **Keep the recipe order** (no `early_ordering_fn`). This is the historical behavior.
- **Give it explicitly** with `add_eqn_order([...])`, e.g. an order you derived once. No extra cost per bake, but it
  does not follow changes to the recipe, and its indices must be kept in sync with the `add_eqn` calls by hand.
- **Derive it** with `rank_by_post_population(prioritize_rare_symbols)` (section 7). It ranks the calls by where a
  good ordering function puts their equations after CSE, which is how Steve found the Z4c order that gave the
  speedup. It costs a trial bake on every run (about 17 s for Z4c). A cheap middle ground is to run it once, copy the
  order it prints ("... derived the early order [...]"), and pin that with `add_eqn_order`.

A plain ordering function as `early_ordering_fn` (e.g. `prioritize_rare_symbols` itself) ranks the groups by where it
puts their equations *before* CSE, which is a different order: on Z4c it gives `[8, 7, 1, 2, 0, 9, 12, 3, 4, 5, 10, 6,
11]`, whose speed has not been measured.

### Do I need to change my recipe?

Only if it uses `insertion_order` (replace it with `pre_population_order`), or relies on auto predicates being called
inside `add_eqn` (for example, to separate two assignments to the same LHS; use a manual `split_loop()`). A recipe that
sets none of the new options generates the same code as before, except that a split after the last position no
longer emits an empty loop, and the cut guards refuse some splits (section 5, "The cut guards").

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Recipe** | The Python file that describes the thorn: declarations, functions, `add_eqn` calls, splits, bake options. The source of truth. |
| **Abstract program** | The elaborator's representation of the program: a mapping of scalar symbols to expressions (SSA), organized into functions and equation lists. |
| **Elaborator** | The middle end: it orders equations, runs CSE, splits loops and promotes temporaries. |
| **Equation list** (`EqnList`) | The equations of one loop of a function. A function's lists form its `EqnComplex`. |
| **Group** | The scalar equations of one `add_eqn` call. |
| **Origin** | The 0-based index of the `add_eqn` call that created an equation (`EqnList.eqn_origin`). |
| **Super-group** | Groups that depend on each other in a cycle, merged into one element of the early locus. |
| **Recipe order** | The order in which equations were added, numbered function-wide. Never rewritten. |
| **Pre-population order** | Starts as the recipe order; rewritten by `early_ordering_fn` and `pre_population_ordering_fn`. Replaces the old insertion order. |
| **Baked order** | `EqnList.order`: the order a bake computed with the list's ordering function and dependency repair. The code is generated in this order. |
| **Dependency repair** | Moving an equation (or group) after everything it reads, whatever order the ordering function asked for. |
| **Trial bake** | The throwaway bake of the whole thorn (through global CSE, without auto splits) that `rank_by_post_population` runs before the real bake to learn the post-CSE order. |
| **Ranked function** | A function whose `early_ordering_fn` is `rank_by_post_population(fn)`. The trial bake orders it with `fn`; one trial serves every ranked function of a thorn. |
| **Derived order** | The 0-based `add_eqn` order that `rank_by_post_population` computes from the trial bake: the calls ranked by the median post-CSE position of their own equations. The real bake uses it as `add_eqn_order(<derived order>)`. |
| **Population** | Global CSE populating the equation lists with temporaries. (Not the paper's "populating the abstract program".) |
| **Locus** (`SplitLocus`) | The point in the bake where a function's auto split predicates are evaluated: `Early`, `PrePopulation` or `PostPopulation`. |
| **Element** | What the positions at a locus separate: groups or super-groups (Early), or scalar equations (the other two). |
| **Position** | A 0-based slot after an element; position `i` means "split after element `i`". |
| **Cut** | A hard or soft split at one position. |
| **Hard split** | Splits a loop into two kernels. Values carried across it become tile temporaries (or are recomputed). |
| **Soft split** | Keeps one kernel, but marks a point after which some temporaries are recomputed instead of kept. Merged back by `merge_soft_splits`. Never creates tile temporaries. |
| **Retainment strategy** | For a soft split: which temporaries are carried forward (retained) and which are forgotten (recomputed). `retain_all()`, `retain_none()`, `retain_rank(n)`, ... |
| **Temporary promotion** | Deciding which kind a CSE temporary becomes; the *temporary promotion strategy* (`promote_none()`, `promote_all()`, ...) decides. |
| **Local temporary** | A CSE temporary that is a local variable of one loop. Recomputed in each loop that needs it. |
| **Tile temporary** | A temporary stored in a grid-shaped buffer so that later loops of the same function can read it. |
| **Global temporary** | A temporary promoted to a full grid function, readable by other functions. |
| **Pull-out** | `pull_out(expr)`: makes `expr` a named temporary so that the ordering can move it on its own. |
| **Overwrite** | `X' = overwrite(X)`: a new version of `X` written into the same storage. |
| **Refinement** | The primitive (`EqnComplex.refine`) that every cut goes through: split each list into ordered pieces. |
| **Piece** | One part of a cut list; it becomes a new list, unless it is empty. |
| **Cut hazard** | A reason not to cut at a position (`CutHazard`): an overwrite (a cut would separate a write of `X'` from a later read of `X`) or a backward read (a cut would put a read before its write). |
| **Cut guard** | The check that refuses the auto splits at hazardous positions, with one warning per function: the overwrite guard refuses hard and soft cuts, the backward-read guard hard cuts only. The predicates are still asked there. |
| **Closure assignment** | How post-population cuts assign temporaries: each piece gets its roots plus the Local temporaries they need. |
| **Probe** | Running the recipe once with placeholder predicates to count the positions at a locus. |
| **Sidecar** | `<checkpoint>.probe.json`: the probe counts and the fingerprint, stored next to a checkpoint. |
| **Fingerprint** | Stable descriptions of the locus and ordering functions a tuner provided, used to detect a changed configuration. |

---

## 11. Where the code lives

| File | What it has |
|---|---|
| `EinsteinEngine/intermediate/split_locus.py` | `SplitLocus`. |
| `EinsteinEngine/intermediate/eqn_grouping.py` | Groups, median ranking, super-groups, dependency repair, the reader-before-overwriter edge, `cut_hazards` / `CutHazard` (the cut guards), `rank_add_eqn_calls`. |
| `EinsteinEngine/intermediate/eqn_ordering.py` | `recipe_order`, `pre_population_order`, `add_eqn_order`, `add_eqn_key_order`, `rank_by_post_population` (its docstring has the full design), `pre_cse_stand_in`. |
| `EinsteinEngine/intermediate/eqnlist.py` | The two orders, `eqn_origin`, `EqnComplex.refine`, `rebake_refined` and the validation, `merge_soft_splits`. |
| `EinsteinEngine/intermediate/post_population_split.py` | Closure assignment for post-population cuts. |
| `EinsteinEngine/intermediate/loop_region.py` | Loop region inference used by the validation. |
| `EinsteinEngine/frontend/dsl/dsl_function_frontend.py` | `_early_bake` (early and pre-population loci), predicate queries (`_query_auto_splits`), the cut guards (`_cut_hazards`), post-population cuts. |
| `EinsteinEngine/frontend/dsl/dsl_frontend.py` | `bake()` (the pipeline), the bake options, the post-population hook. |
| `EinsteinEngine/frontend/dsl/dsl_frontend.py`, `_derive_early_orders` | The trial bake of `rank_by_post_population`. `eqn_grouping.rank_add_eqn_calls` ranks the calls. |
| `EinsteinEngine/tuning/probe.py`, `tuning.py`, `tune_splitting.py` | Probing, the sidecar (including derived orders), the split tuners. |
| `unit_tests/test_split_loci.py`, `unit_tests/test_post_population_split.py`, `unit_tests/test_rank_by_post_population.py` | Small, readable examples of every rule above. |
