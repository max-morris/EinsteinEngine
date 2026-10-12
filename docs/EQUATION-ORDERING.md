# Equation ordering and split loci

This document explains how the elaborator orders a function's equations while it bakes it, and where the automatic
("auto") split predicates cut the function into loops. It assumes you know the engine's basics (recipes, `add_eqn`,
`split_loop()` / `soft_split()`, global CSE, temporary promotion, soft split retainment).

In short:

- A function's bake has **three points** at which it can be ordered, each with its own **ordering function**:
  `early_ordering_fn` (whole `add_eqn` calls), `pre_population_ordering_fn` (scalar equations, before CSE), and
  `ordering_fn` (after CSE).
- The auto split predicates are evaluated at **one** of those points per function, its **split locus**
  (`SplitLocus.Early`, `SplitLocus.PrePopulation` or `SplitLocus.PostPopulation`). The locus decides what a
  "position" is.
- Positions are **0-based**: position `i` means "split after element `i`".
- Every function keeps **two orders**: the recipe order, which is never rewritten, and the pre-population order, which
  the first two ordering functions rewrite.
- The early order can be given explicitly (`add_eqn_order([...])`) or, by a tuner, derived once from a run of the
  recipe (`rank_by_post_population(fn)`, section 7).

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
8. [Known limitations](#8-known-limitations)
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

The engine's progress messages ("Early Baking fn...", "Performing CSE...", and so on) are left out of the outputs;
warnings are kept. Run with `PYTHONHASHSEED=0` to reproduce the exact temporary names.

---

## 1. The bake pipeline of a function

The frontend flattens the recipe into the *abstract program*: each `add_eqn` call is exploded into scalar equations
(one per tensor component), and each function holds its equations in one or more *equation lists*, one per loop
(kernel). `split_loop()` and `soft_split()` start a new list. The *elaborator* then orders, CSEs, splits and promotes
while `bake()` runs, and the generator turns each list into a loop.

```
 recipe: add_eqn(...), split_loop(), soft_split(), pull_out(...), overwrite(...)
   |      (each add_eqn call records its 0-based index as the origin of its scalar equations)
   v
 DslFrontend.bake(**options)
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
   |    (without CSE, the lists are rebaked with ordering_fn if pre_population_ordering_fn is set)
   |    >>> POSTPOPULATION LOCUS: cut in the post-CSE order
   |
   |  merge_soft_splits: soft splits from any source (manual or auto) are merged back
   |    into one loop here, using the soft split retainment strategy
   |  splitmaxxing, if enabled; late bake (recycling temporaries)
   v
 generator: one loop per remaining list
```

**"Population"** here means *global CSE populating the equation lists with temporaries*, not the frontend
populating the abstract program while the recipe runs. "Pre-population" means "before CSE has added its
temporaries", and "post-population" means "after".

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

- **When `pre_population_ordering_fn` is None, `ordering_fn` orders both the pre-CSE bake (step 4) and the post-CSE
  rebake.** The pre-population order is then *not* rewritten; only the baked order of step 4 follows `ordering_fn`.
- **`ordering_fn` is sticky.** Each list stores it, and every later rebake (after CSE, after a cut, after a merge)
  reuses it, which is why step 5 resets every list to `ordering_fn`. The lists produced by a cut inherit it.
- **Bayesian optimization never runs before CSE.** An ordering function whose name (or whose `functools.partial`'s
  function's name) contains `bayesian` is replaced by `prioritize_rare_symbols` for any bake before CSE and for any
  "fast" rebake. This applies to `early_ordering_fn` and `pre_population_ordering_fn` too.
- **The early ordering function is not used as an order directly.** It ranks groups (section 3).

### Setting them

The options go to `bake()`, either for every function or per function through `functions={name: {...}}`, which
overrides the bake-wide values for that function. `add_eqn_order([...])` refers to one function's `add_eqn` calls, so
pass it per function, as Z4c does for its RHS function (`recipes/Cottonmouth/Z4c.py`):

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

A name in `functions` that is not a function of the thorn is an error, so a typo cannot silently drop the options
(`The bake options name unknown functions ['my_rsh']; the functions are ['my_rhs'].`).

---

## 2. Two orders: recipe order and pre-population order

Every equation list keeps two orders of its equations, as dicts from LHS to position:

- **Recipe order** (`EqnList.eqn_recipe_order`): the order in which the equations were added to the function,
  numbered function-wide. It is **never rewritten**, so you can always get back to "the order the author wrote".
- **Pre-population order** (`EqnList.eqn_pre_population_order`): starts equal to the recipe order, and is rewritten by
  `early_ordering_fn` (step 1) and by `pre_population_ordering_fn` (step 4). Without them it stays the recipe order.

Neither is the order the code is generated in. That is the *baked order*, `EqnList.order`, which each bake computes
by running the list's ordering function (with dependency repair). The two stored orders are inputs an ordering
function can use.

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

`pre_population_order` is the replacement for the removed `insertion_order`: without an early or pre-population
ordering function it gives the same order. `recipe_order` is not a replacement, because the two differ after a soft
split merge: the pre-population order appends the equations a merge moved into a list, while the recipe order puts
them back at their recipe positions and lists every temporary after every recipe equation. For example, with one soft
split after call 0, merged with `retain_all()`:

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
loop, which keeps `c` before `b` when the early ordering function asked for it.

---

## 3. Groups and the early ordering function

### What a group is

A **group** is the set of scalar equations created by one `add_eqn` call. `add_eqn(v[li], [...])` with three
components makes one group of three equations. Every equation records the 0-based index of the `add_eqn` call that
created it, its **origin** (`EqnList.eqn_origin`):

- The first `add_eqn` call of a function is call 0, the next call 1, and so on, across all of the function's lists.
- A pull-out temporary inherits the origin of the equation it was pulled out of.
- CSE temporaries have no origin. Equations added by synthetic functions (the ones global CSE creates) have none
  either, and each of them is a group of its own.

The early locus reorders whole groups, never the equations inside a group, which stay in recipe order.

### How the early ordering function orders groups

When `early_ordering_fn` is set, each list is processed on its own (lists are never merged or reordered relative to
each other, so nothing moves across a manual split):

1. **Run the function over the list's scalar equations.** The result is a position for each scalar equation.
   Equations the function does not yield rank last, in recipe order.
2. **Rank each group by the median position of its members.** Ties go to recipe order. A group lands where most of
   its members are, not where its first member is.
3. **Find the dependencies between groups.** Group B depends on group A if an equation of B reads a LHS of A. In
   addition, a group that reads `X` must come before a group that writes `X'` (an `overwrite` of `X`): the
   **reader-before-overwriter** edge.
4. **Merge dependency cycles into super-groups.** Two groups can depend on each other even though their scalar
   equations do not form a cycle (group A computes `vD1 = 2*b`, group B computes `b = 3*vD0`). Such groups are
   merged into one **super-group**, ranked by the median over all of its members.
5. **Sort topologically by rank** (**dependency repair**). Repeatedly emit the lowest-ranked (super-)group whose
   dependencies have all been emitted (Kahn's algorithm on a heap), so a group never comes before a group it reads.
6. **Rewrite the pre-population order** with the result.

Without `early_ordering_fn`, none of this happens: each `add_eqn` call is its own group, in recipe order, even when an
earlier call reads what a later call writes (section 5, "The backward-read guard").

### `add_eqn_order` and `add_eqn_key_order`

Two helpers make early ordering functions from `add_eqn` call indices (both exported from `EinsteinEngine`):

- `add_eqn_order([i0, i1, ...])`: put call `i0` first, then `i1`, and so on. Indices are 0-based. Calls that are not
  listed keep their recipe order after the listed ones. Repeated or negative indices are rejected when it is created
  (`add_eqn_order: the call indices [1] are listed more than once.`), and an index the function does not have is
  rejected at bake time (`add_eqn_order: call index 2 is out of range; the function has 2 add_eqn calls (valid indices
  are 0..1).`).
- `add_eqn_key_order(key)`: order the calls by `key(call_index)`, ties by recipe order. For example,
  `add_eqn_key_order(lambda i: -i)` reverses them.

Their `repr` shows the order (`add_eqn_order([8, 1, 0, ...])`), which the tuners record (section 6). A tuner can also
derive the order from a run of the recipe with `rank_by_post_population(fn)` (section 7).

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

The ranks are `d` 0, `b` 1, `v` 2, `a` 3. At the start only `a` and `d` have no unemitted dependencies, so `d`
(rank 0) goes first, then `a`, `v` and `b`. The baked order follows the pre-population order because
`ordering_fn=pre_population_order`.

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

Call 0 computes `vD1 = 2*b`, and call 1 computes `b = 3*vD0`. The two groups form one super-group, which is **one
early position**, so three calls give only two positions:

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

- The predicates are called once per position at bake time, at the function's locus (and nowhere else), after the
  ordering function of that locus has run, so they see its order. They are not called inside `add_eqn`.
- At each position, the hard predicate is asked first, and the soft predicate only if there is no hard predicate or
  it returned False. **Hard wins over soft.**
- The soft predicate returns False (no split), True (a soft split with the bake's `soft_split_retainment_strategy`),
  or a `SoftSplitRetainmentStrategy` (a soft split with that strategy).
- The predicates are called at every position, in increasing order, even where the engine then refuses the split
  (section 5), so a tuner always sees the same positions.

### What a position is

A **position** is a slot between two consecutive **elements** in the order that exists at the locus. Position `i` is
the slot right after element `i` (counting from 0). A function with `N` elements has positions `0..N-1`, and position
`N-1` is after the last element.

| Locus                       | Elements                                                                       | Their order                                                                                 | Z4c `z4c_rhs` N |
|-----------------------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-----------------|
| `SplitLocus.Early` (default) | `add_eqn` calls, or super-groups when `early_ordering_fn` is set              | `early_ordering_fn` over groups; the recipe order without it                                | 13              |
| `SplitLocus.PrePopulation`  | scalar equations and pull-out temporaries                                      | the pre-CSE baked order: `pre_population_ordering_fn`, or `ordering_fn` when that is None   | 64              |
| `SplitLocus.PostPopulation` | scalar equations, pull-out temporaries and CSE temporaries                     | the post-CSE baked order: `ordering_fn`                                                      | 1061            |

The Z4c counts are for the recipe's default ordering functions.

Positions **run across lists**: if manual splits made two lists, the positions of the second list continue where the
first list's ended. A cut is always made within one list, and no locus moves an equation from one list to another, so
a manual `split_loop()` or `soft_split()` is always kept and nothing is reordered across it.

### The same recipe at the three loci

```python
def loci_recipe(name, locus, hard):
    gf = ThornDef("DOCS", name)
    a, b, c, u, v = [gf.decl(n, []) for n in ("a", "b", "c", "u", "v")]
    fun = gf.create_function("f", ScheduleBin.Evolve, auto_split_locus=locus, auto_hard_split_predicate=hard)
    fun.add_eqn(a, pull_out(sin(u) * cos(v)) + (u * v + 1) ** 2)  # call 0
    fun.add_eqn(b, (u * v + 1) ** 3 + a)                          # call 1
    fun.add_eqn(c, (u * v + 1) * b)                               # call 2
    return gf, fun


for locus in (SplitLocus.Early, SplitLocus.PrePopulation, SplitLocus.PostPopulation):
    hard = Recorder(lambda i: False)
    gf, fun = loci_recipe(f"LOCI{locus.name.upper()}", locus, hard)
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

Three `add_eqn` calls give 3 early positions. Pull-out adds `pull_out_0` (4 pre-population positions), and CSE adds
`x0`, `x1` and `x2` (7 post-population positions).

### Empty pieces are dropped

A cut can produce a piece with no equations: a split after the last element, or a split right before a manual split.
An empty piece is dropped, and its boundary is folded into the next piece's:

- Between a hard and a soft boundary, the hard one wins.
- Between two soft boundaries, the later one (the one attached to actual equations) wins, with its retainment
  strategy.
- A custom loop annotation from a manual split survives.

So **a split after the last position does nothing**. An empty list that manual splits left behind (e.g.
`split_loop()` before the first `add_eqn`) is dropped only when some auto split fires in that function; otherwise it
stays.

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

Without an early ordering function, the elements are the `add_eqn` calls in recipe order. Early cuts are made before
the first bake, so, like manual splits, they are not validated (section 5).

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

- **Hard cut, `promote_none()`**: `x0` is needed on both sides, so it is recomputed in each loop. `a` crosses the
  hard cut, so it becomes a tile temporary, as it would with a manual split.
- **Hard cut, `promote_all()`**: `x0` may be promoted, so it is computed once, in the first loop that needs it, and
  carried to the second loop as a tile temporary.
- **Soft cut, `retain_none()`**: the soft cut is merged back into one loop. The retainment strategy forgets `x0` at
  the cut, so the merge recomputes it as `x0_ss0` for the equations after the cut.
- **Soft cut, `retain_all()`**: the merge keeps `x0`, so the result is the unsplit loop.
- **After the last position**: the trailing piece is empty and is dropped.

Positions are in the post-CSE order, before `merge_soft_splits`. A merged loop is rebaked with `ordering_fn`, so
after a soft cut the final order need not be the order that was cut. If CSE is off (`do_cse=False`), the
post-population locus cuts the order `ordering_fn` gives (there are no CSE temporaries), and there is nothing to
promote.

---

## 5. What the engine guarantees at a cut

All cuts, at every locus, go through one primitive, `EqnComplex.refine`, which splits each existing list into ordered
pieces and builds a new list from each piece.

### Every locus

- **Pieces keep the order that was cut.** Each new list is rebaked with an order-preserving function over its piece
  (with dependency repair, which never moves anything when the cut order already respects dependencies). So the order
  the predicates saw is the order CSE sees (pre-population locus) or the final order (post-population locus, unless a
  soft cut is merged).
- **No backward dependencies.** No auto hard cut leaves a piece reading something that a later piece computes (see
  "The backward-read guard" below).
- **Parameters are repartitioned.** Each piece gets the grid spacing parameters, the parameters its equations read,
  and the parameters its recipe equations registered.
- **Manual splits are kept**, and their custom loop annotations too (section 4).
- **A new list inherits its source list's `ordering_fn`.**

### Early and pre-population loci

These cuts happen before CSE, which then runs on the new lists exactly as on lists made by manual splits, so temporary
promotion, tile temporaries and soft split merging work as usual.

A pull-out temporary that a pre-population hard cut separates from its readers becomes a tile temporary. In a Cactus
thorn it gets a centering inferred from its right-hand side, the same way CSE temporaries get one.

### Post-population locus: closure assignment

After CSE, a list already contains its Local CSE temporaries, and one temporary may be read on both sides of a cut,
so the list is not simply sliced. Each piece is assigned by **closure**:

- The *roots* of a piece are the equations in its slice that are not Local CSE temporaries (or existing tile
  temporaries).
- The piece gets its roots, plus every Local CSE temporary that the roots need, transitively.

So:

- A Local temporary needed only after a cut **moves** there.
- A Local temporary needed on both sides of a cut is **recomputed** in each loop that needs it.
- A temporary that no piece needs is dropped.
- A borrowed temporary is placed just before its first reader in the piece.

**Only hard cuts promote**, since a soft split generates no tile temporaries. A Local temporary needed by pieces
separated only by soft cuts is always duplicated, and `merge_soft_splits` then decides, through the retainment
strategy, whether the merged loop keeps one copy or recomputes it. When the pieces that need a temporary span a hard
cut, the temporary promotion strategy that global CSE used decides: if it classifies the temporary as Tile (e.g.
`promote_all()`), the temporary is computed once, in the first piece that needs it, and becomes a tile temporary.
Under `promote_none()` it is duplicated. A Local temporary that CSE placed in several lists of the function is never
promoted.

**Existing tile temporaries move to their first reader.** A temporary that CSE already made a tile temporary is never
duplicated. It moves to the first piece of its list that reads it.

**Named symbols stay put.** Recipe equations and pull-out temporaries stay in their slices and are never duplicated.
One that crosses a hard cut becomes a tile temporary, as with manual splits.

### The cut guards

Before making the cuts at any locus, the engine computes the **cut hazards** of every position and refuses the cuts
that would change what the code computes:

| Hazard        | A cut at position p would ...                                                          | Refuses     | Warning says                                               |
|---------------|-----------------------------------------------------------------------------------------|-------------|------------------------------------------------------------|
| overwrite     | separate a write of `X'` (at or before p) from a later read of `X` (after p)            | hard and soft | `which would separate X' from a later read of X`         |
| backward read | leave an element at or before p reading a symbol that an element after p writes        | hard only   | `which would put a read of b before its write` (or `reads of a2, b ... their writes`) |

The predicates are still asked at refused positions, so the probe counts are unaffected. Outside tuning, a split
they request there is ignored with one warning per function (when both kinds apply, the warning joins both parts
with "and"). In a tuning trial, any refused cut fails the trial (section 6).

#### The overwrite guard

`X' = overwrite(X)` writes a new value into the storage of `X`. If a cut put the write of `X'` in one loop and a later
read of `X` in a later loop, the later loop would read the new value instead of the old one. The engine therefore
refuses any position at which a cut would separate a writer of `X'` (or `X''`, ...) from a later reader of an
earlier version. At the post-population locus, a reader also counts as reading `X` through any Local temporary it
reads, since the closure assignment may move or duplicate that temporary (e.g. a CSE temporary `x0 = X`). For a soft
cut, which the merge rejoins, the refusal is conservative.

With a hard split requested at every position:

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
loci, position 1 is between `X'` and `Y'`, and position 2 is after the last element. At the post-population locus, the
order is `a, x0, x1, x2, X', Y'`: position 4 (between `X'` and `Y'`) is refused, because `Y'` reads `X` through `x2`
and `x0`. The cuts at positions 1 to 3 are made, but `x0`, `x1` and `x2` move to their reader `X'`, so those pieces
end up empty and are dropped.

#### The backward-read guard

At the early locus without an early ordering function, the elements are the `add_eqn` calls in recipe order, and
nothing forces that order to respect the dependencies between calls. A recipe may compute `a = 2*b` in call 0 and
`b` in call 1 (the bake's dependency repair puts `b` first within a loop), and two calls may depend on each other
(call 2 computes `vD1 = 2*a2`, call 3 computes `a2 = 3*vD0 + a`). A **hard** cut between such calls would put the read
in an earlier loop than the write, so it is refused.

A **soft** cut there is made: `merge_soft_splits` rejoins the loops and orders the merged loop with dependency
repair. With an early ordering function the question never arises, because the calls are ordered topologically and
mutually dependent calls form one super-group. The other loci cut a baked order, which already respects dependencies.

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
calls) are refused, position 1 is cut, and position 3 is after the last call. In the second case the soft cut is made
and then merged away. In the third case, the early ordering function puts call 1 before call 0 and makes calls 2 and
3 one super-group, so there are only three positions, and every hard cut is made.

A manual `split_loop()` between a read and its write is the author's responsibility and fails with a `DslException`
(see "Validation errors" below). A manual `soft_split()` there works.

### Validation errors

After a pre-population or post-population cut, the engine checks that the cut did not change the meaning of the
function and raises a `DslException` if it did. Early cuts are made before the first bake, like manual splits, so
there is nothing to compare them with and they are not checked. The last row of the table comes from the check every
bake makes on its loops, whatever made the split. In a tuning run, a trial that raises fails (section 6).

| Message                                                                                                                 | Meaning                                                                                                                                                                                                                                                              |
|-------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `After splitting, the inferred write region of 'X' in loop i (cut from loop s) changed from A to B.`                    | A piece would write the output `X` over a different region (Interior, Boundary, Everywhere) than the unsplit list did. The engine seeds each piece with what it knew was analytic to prevent this, so this points to an engine problem. Report it, and avoid the position. |
| `After splitting, loop i, which runs over R1, reads temporary 't', which loop j only computes over R2.`                 | The cut made `t` cross into another loop, as a tile temporary, but the loop that computes it runs over a smaller region than a loop that reads it and writes grid variables. The reading loop would use values that were never computed. Choose another position.         |
| `After splitting, loop i reads 'X', which loop j (cut from the same loop s) has already overwritten.`                   | A backstop for the overwrite guard. Auto splits never cut there, so it points to an engine problem.                                                                                                                                                                  |
| `'b' is written in loop 1 after it is read in loop(s) [0]: a loop cannot read a value that a later loop computes.` | Raised for any split, but in practice by a manual `split_loop()` placed between an equation and a later one it reads (auto hard cuts are refused there). Move the manual split, or make it a `soft_split()`. |

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

### Other errors you may see

- **`Equation for 'a' is already defined in this loop. ...`** A LHS may be assigned only once between manual splits.
  Auto split predicates are evaluated at bake time, so they cannot separate two assignments to the same LHS; use a
  manual `split_loop()` there.
- **`add_eqn_order: ...`** and **`The bake options name unknown functions ...`**: see sections 3 and 1.

---

## 6. Tuning

[TUNING-QUICKSTART.md](TUNING-QUICKSTART.md), section 5, has the full walkthrough of the split tuners, the probe and
the probe file. What matters here:

- The split tuners (`CombinatorialSplitTuner`, `CutPositionSplitTuner`) provide the out-params
  `auto_hard_split_predicate`, `auto_soft_split_predicate` and `auto_split_locus`, plus whichever of
  `early_ordering_fn`, `pre_population_ordering_fn` and `ordering_fn` you give them. The recipe must read each of them
  with `get_tuning_param` / `get_optional_tuning_param` and pass it to `create_function` or `bake`, as Z4c does
  (section 1).
- **The number of positions `N` is probed.** Before the search, the recipe runs once with placeholder predicates that
  never split, and the engine reports how many positions each predicate is asked about. `N` depends on the recipe,
  the locus, and the ordering functions (an `early_ordering_fn` can merge groups into super-groups), so the locus and
  the ordering functions are provided while probing.
- `CombinatorialSplitTuner` has two params per position, so it is practical only at the early locus. Use
  `CutPositionSplitTuner(n_cuts, locus=...)` at the other two.
- **A trial whose cuts the engine refuses fails.** If a cut guard (section 5) refuses any split a trial asked for,
  the trial fails (`remote_feedback.TrialFailed`) before its code is synced or built, so a configuration whose cuts
  were dropped is never scored as if they had been made, and can never become the best. A validation error (section 5)
  also fails the trial.

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

A probe fails if the recipe never reads one of the names it was given, since a recipe that ignored the locus or an
ordering function would be tuned against the wrong positions.

The probe result is saved next to the checkpoint as `<checkpoint>.probe.json`, with a stable description (a
*fingerprint*) of the locus and ordering functions the tuner provided, such as
`"early_ordering_fn": "add_eqn_order([8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11])"`, and any order derived for
`rank_by_post_population` (section 7). Resuming a search, or running `generate_best`, refuses to continue if the count
or a description changed, because the checkpoint's coordinates only mean something for the positions they were
recorded against. See TUNING-QUICKSTART.md, "The probe file", for what the fingerprint can and cannot see, and "Old
checkpoints" for converting checkpoints with 1-based split params (`split_1..N`).

---

## 7. Deriving an early order automatically: `rank_by_post_population`

`add_eqn_order([...])` needs you to know a good order of the `add_eqn` calls. A tuner can instead take
`early_ordering_fn=rank_by_post_population(fn)` (from `EinsteinEngine.tuning.probe`; it can only be the
`early_ordering_fn`), which orders the calls by where `fn` places their equations **after CSE**. On Z4c,
`rank_by_post_population(prioritize_rare_symbols)` derives `[8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11]`.

```python
def get_tuner() -> Tuner:
    return CombinatorialSplitTuner(early_ordering_fn=rank_by_post_population(prioritize_rare_symbols))
```

### Why it needs a separate run

The order worth imitating is the post-CSE order that a good ordering function gives, but the early locus runs
**first**:

```
   early locus ---> pull-out ---> pre-CSE bake ---> global CSE ---> post-CSE order
       ^                                                                  |
       +------- the ranking the early locus needs is computed here -------+
```

So the order is derived from a separate run of the recipe, and the trials then use it as an ordinary
`add_eqn_order`.

### How it is resolved

`tuning.build_experiment` resolves it once per checkpoint:

1. If the checkpoint's probe file already records an order for it (with the same description), that order is used.
2. Otherwise `probe.derive_add_eqn_order` runs the recipe once, with `fn` as the `ordering_fn` out-param, no early or
   pre-population ordering function, and no auto split predicates. The engine calls the probe at the start of the
   bake (`probe.bake_started`, which snapshots each call's equations before pull-out temporaries inherit their
   origin) and after global CSE (`probe.cse_finished`, which reads the post-CSE order of the function baked with
   `fn`). The whole thorn is baked, because global CSE works across functions.
3. `eqn_grouping.rank_add_eqn_calls` ranks the calls of each list by the median post-CSE position of the equations
   each call produced (not its pull-out temporaries, nor CSE temporaries). Ties go to the lower call index, and a
   call none of whose equations survive goes last.
4. The order is printed and recorded in the probe file as `"derived_orders": {"early_ordering_fn": [...]}`. The probe,
   every trial, a resumed search, `generate_best` and `plot_tuning` use `add_eqn_order(<that order>)`.

The recipe must read `early_ordering_fn` and `ordering_fn` with `get_optional_tuning_param` and pass both to the same
function, and exactly one function may be baked with `fn`. An order derived once is never derived again for that
checkpoint, so a later change to the engine or the recipe cannot silently change what the checkpoint's split
coordinates mean. To derive it again, start a new checkpoint file.

### Worked example

```python
# derive_demo.py
from EinsteinEngine import *
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornDef
from EinsteinEngine.tuning.tuning import get_optional_tuning_param

gf = ThornDef("DOCS", "RANKED")
v = gf.decl("v", [li])
u, a, src = gf.decl("u", []), gf.decl("a", []), gf.decl("src", [])
fun = gf.create_function("fn", ScheduleBin.Evolve)
fun.add_eqn(u, src)                              # call 0
fun.add_eqn(v[li], [src * 2, src * 3, src * 4])  # call 1
fun.add_eqn(a, u + 1)                            # call 2
gf.bake(do_cse=False,
        ordering_fn=get_optional_tuning_param("ordering_fn", pre_population_order),
        early_ordering_fn=get_optional_tuning_param("early_ordering_fn", None))
```

```python
from EinsteinEngine.tuning.probe import derive_add_eqn_order
print(derive_add_eqn_order("derive_demo.py", lexicographical_order))
```

```text
('fn', (0, 2, 1))
```

Baked without CSE, "post-CSE" is just the baked order, `u, a, vD0, vD1, vD2` (lexicographic, with `u` moved in front
of `a`, which reads it). Call 0 (`u`) is at position 0, call 2 (`a`) at position 1, and call 1 at positions 2, 3 and 4,
median 3. So the order is `[0, 2, 1]`. As with any `add_eqn_order`, the early locus still repairs dependencies: a call
that ranks before a group it reads is moved after it.

## 8. Known limitations

- **One loop's order does not respect reader-before-overwriter.** Only the early grouping (section 3) adds the edge
  that keeps a read of `X` before the write of `X'`. The ordering functions that order the equations *within* a loop do
  not, so a loop can compute `X'` before a later read of `X` in the same loop. This affects unsplit functions too. A
  related case: a soft cut that respects the guard, merged with `retain_none()`, recomputes a forgotten `x = X` for the
  later piece, and the rebake of the merged loop can then place `X'` before that load. A manual `soft_split()` there
  does the same.
- **The vanilla F90 generator does not support `overwrite()`.** It declares the overwritten variable both as an input
  and as an output instead of `INOUT`.
- **Positions the backward-read guard refuses still count toward `N`** (early locus without an early ordering
  function; section 5), so a tuner can ask for a hard split there, and that trial fails.
- **Early cuts are not validated** (section 5). They behave exactly like manual splits, including write regions that
  change silently.
- **A grid function that crosses a hard cut is carried as a tile temporary of the same name** (e.g. `a` in the
  post-population example of section 4), as with manual splits.
- **`merge_soft_splits` resets custom loop annotations** to their defaults.
- **Bayesian `ordering_fn` at the post-population locus.** If a function has (manual) soft splits, its post-CSE
  rebake is a "fast" one (a merge follows), so a Bayesian `ordering_fn` is replaced by `prioritize_rare_symbols` there,
  and the post-population positions are in that order.
- **`rank_by_post_population` ranks by the order right after the post-CSE rebake**, i.e. before soft split merges,
  splitmaxxing and temporary recycling (section 7).
- **Memory across tuning trials.** Class-level caches keep references to every baked function, so memory grows over
  many in-process trials.
- **Post-population cutting is the most complex code path** and the least tested.

---

## 9. FAQ

### Why did my split not happen?

Check, in this order:

1. **Is the predicate asked at the position you think?** Positions are 0-based and mean "after element `i`". Use a
   `Recorder` (see "Running the examples") to see which positions are queried.
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
   a checkpoint with more positions than the recipe has never uses the extra ones.

### Why does the tuner say an out-param was "never read"?

The recipe never read one of the predicates, the locus or the ordering functions the tuner provides while probing,
so the probe would count the wrong positions. Read every provided name with `get_optional_tuning_param(name,
default)` and pass it on, as in section 1. Only reads made before the probe stops the recipe count; see
TUNING-QUICKSTART.md, "Probing".

### Why is my old checkpoint refused?

One of:

- It uses 1-based keys (`split_1..N`), or has more positions than the recipe has. Convert it with
  `scripts/shift_checkpoint_indices.py` (TUNING-QUICKSTART.md, "Old checkpoints").
- It predates the soft retain percentile params. Such a checkpoint cannot be converted.
- The probe file next to it no longer matches: the number of positions changed, or the tuner's locus or ordering
  functions changed. The message lists each difference. Start a new checkpoint, or restore the recipe and tuner.
- It was recorded with `add_eqn_order([...])` and you now use `rank_by_post_population(...)`, or the other way
  around. The fingerprints differ even when the order is the same, so start a new checkpoint.

### Which locus should I tune?

- **Early** has the fewest positions (13 for Z4c) and works with `CombinatorialSplitTuner`. Combine it with an
  `early_ordering_fn` to search splits in a different group order. Start here.
- **PrePopulation** (64 for Z4c) can cut inside a tensor equation and between pull-out temporaries and their readers,
  before CSE decides what to share. Use `CutPositionSplitTuner`.
- **PostPopulation** (1061 for Z4c) cuts the final order, CSE temporaries included, and recomputes or promotes shared
  temporaries at each cut. It is the most flexible and the least tested. Use `CutPositionSplitTuner` with a small
  number of cuts.

### How do I get a good early order?

- **Keep the recipe order** (no `early_ordering_fn`).
- **Give it explicitly** with `add_eqn_order([...])`, e.g. an order you derived once. Its indices must be kept in sync
  with the `add_eqn` calls by hand.
- **Derive it** in a tuner with `rank_by_post_population(prioritize_rare_symbols)` (section 7). It costs one extra
  run of the recipe per new checkpoint, and the order is recorded in the probe file (copy it into an `add_eqn_order`
  to use it in the recipe itself).

A plain ordering function as `early_ordering_fn` (e.g. `prioritize_rare_symbols` itself) ranks the groups by where it
puts their equations *before* CSE, which generally gives a different order.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Recipe** | The Python file that describes the thorn: declarations, functions, `add_eqn` calls, splits, bake options. |
| **Abstract program** | The elaborator's representation of the program: a mapping of scalar symbols to expressions (SSA), organized into functions and equation lists. |
| **Elaborator** | The middle end: it orders equations, runs CSE, splits loops and promotes temporaries. |
| **Equation list** (`EqnList`) | The equations of one loop of a function. A function's lists form its `EqnComplex`. |
| **Group** | The scalar equations of one `add_eqn` call. |
| **Origin** | The 0-based index of the `add_eqn` call that created an equation (`EqnList.eqn_origin`). |
| **Super-group** | Groups that depend on each other in a cycle, merged into one element of the early locus. |
| **Recipe order** | The order in which equations were added, numbered function-wide. Never rewritten. |
| **Pre-population order** | Starts as the recipe order; rewritten by `early_ordering_fn` and `pre_population_ordering_fn`. |
| **Baked order** | `EqnList.order`: the order a bake computed with the list's ordering function and dependency repair. The code is generated in this order. |
| **Dependency repair** | Moving an equation (or group) after everything it reads, whatever order the ordering function asked for. |
| **Derived order** | The `add_eqn` order that `rank_by_post_population` is resolved to (section 7), recorded in the probe file. |
| **Population** | Global CSE populating the equation lists with temporaries. |
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
| **Cut hazard** | A reason not to cut at a position (`CutHazard`): an overwrite or a backward read (section 5). |
| **Cut guard** | The check that refuses the auto splits at hazardous positions: the overwrite guard refuses hard and soft cuts, the backward-read guard hard cuts only. |
| **Closure assignment** | How post-population cuts assign temporaries: each piece gets its roots plus the Local temporaries they need. |
| **Probe** | Running the recipe once with placeholder predicates to count the positions at a locus. |
| **Sidecar** | `<checkpoint>.probe.json`: the probe counts, the fingerprint and any derived orders, stored next to a checkpoint. |
| **Fingerprint** | Stable descriptions of the locus and ordering functions a tuner provided, used to detect a changed configuration. |

---

## 11. Where the code lives

| File | What it has |
|---|---|
| `EinsteinEngine/intermediate/split_locus.py` | `SplitLocus`. |
| `EinsteinEngine/intermediate/eqn_grouping.py` | Groups, median ranking, super-groups, dependency repair, the reader-before-overwriter edge, `cut_hazards` / `CutHazard` (the cut guards), `rank_add_eqn_calls`. |
| `EinsteinEngine/intermediate/eqn_ordering.py` | `recipe_order`, `pre_population_order`, `add_eqn_order`, `add_eqn_key_order`, `pre_cse_stand_in`. |
| `EinsteinEngine/intermediate/eqnlist.py` | The two orders, `eqn_origin`, `EqnComplex.refine`, `rebake_refined` and the validation, `merge_soft_splits`. |
| `EinsteinEngine/intermediate/post_population_split.py` | Closure assignment for post-population cuts. |
| `EinsteinEngine/intermediate/loop_region.py` | Loop region inference used by the validation. |
| `EinsteinEngine/frontend/dsl/dsl_function_frontend.py` | `_early_bake` (early and pre-population loci), predicate queries (`_query_auto_splits`), the cut guards (`_cut_hazards`), post-population cuts. |
| `EinsteinEngine/frontend/dsl/dsl_frontend.py` | `bake()` (the pipeline), the bake options, the post-population hook. |
| `EinsteinEngine/tuning/probe.py`, `tuning.py`, `tune_splitting.py` | Probing, `rank_by_post_population` and `derive_add_eqn_order`, the sidecar (including derived orders), refused-cut collection, the split tuners. |
| `EinsteinEngine/tuning/remote_feedback.py` | One trial: generate, check for refused cuts, sync, build, run, time; `TrialFailed`. |
| `unit_tests/test_split_loci.py`, `unit_tests/test_post_population_split.py`, `unit_tests/test_rank_by_post_population.py` | Small, readable examples of every rule above. |
