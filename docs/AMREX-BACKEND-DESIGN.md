# A shared AMReX backend: design

Status: design version 3.2 (2026-10-20), signed off by both reviewers (round 3), for Max's approval. Version 3.2 is
3.1 plus reviewer B's round-3 notes. Branch `foreign/amrex`, from master `8e45a5b7`. Nothing is
implemented on this branch yet. Versions 1 (`0861b89e`) and 2 (`d0e6ae61`) were blocked by both reviewers; section
12 maps every finding to its resolution.

Max's decision (2026-10-20): "I want a shared, subclassable/extensible amrex backend, as anything else requires a lot
of repeated work."

## 1. Purpose and scope

EE generates kernels for two AMReX codes, GRTeclyn and MHDuet, and both need `AMREX_GPU_DEVICE` code on
`amrex::Array4`. They differ in nearly every interface detail: how the code is invoked, where fields live, how
coordinates and parameters are spelled, and where results go. This backend implements once what they share:
- the DSL-to-C++ lowering, classification and checks;
- temporaries, stencil reads and splits;
- correct, deterministic printing.

Each code gets a subclass for the rest.

**Non-goals.**
- No change to EE's DSL or intermediate layer. Every requirement is met at the frontend, generator or emit level,
  and the backend uses only existing hooks (bake options, CSE optimizations).
- No timing instrumentation, MPI, ghost filling or boundary conditions: the host code owns those.

## 2. What exists

- **EE master (`8e45a5b7`): no AMReX code.** The generic C++ and F90 printers print floats with 15 digits.
  MHDuet's separate master PR (`ee-floatfix`) changes that; this backend prints its own literals in any case.
- **`foreign/grteclyn` (`0bd5e1cc`): a validated, twin-reviewed AMReX backend specific to GRTeclyn**
  (`frontend/dsl/amrex/`, `generators/amrex_generator.py`, `emit/code/amrex/`, `wizards/amrex_wizard.py`,
  `unit_tests/test_amrex_backend.py`).
  - It emits a class with one `AMREX_GPU_DEVICE` function per loop, and a `compute()` that launches them.
  - It supports hard splits, keeping tile temporaries in a stream-ordered scratch MultiFab.
  - Its CCZ4 header has md5 `27b43325`.
  - It lacks 6 master commits, including PR #112 (CSE factoring), and it imports `subst_tensor_grchombo` from the
    GRChombo backend, which master lacks.
- **`foreign/mhduet` (`499982ba`): a recipe only.** Its backend waits for this design.

## 3. Requirements

| # | requirement | GRTeclyn | MHDuet |
|---|---|---|---|
| R1 | invocation | header-only class; `compute(MultiFab& rhs, const MultiFab& state)` launches one `ParallelFor` per loop, in order; host synchronizes | header-only `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE` free function per DSL function, in a namespace, called inside a host-written fused `ParallelFor`; no launch, scratch, shared memory, atomics |
| R2 | fields | state `Array4<Real const>` (components `c_<name>`); rhs `Array4<Real>` (same enum) | evolved fields in one `Array4` parameter `un` (per RK stage), components `simflowny_vars::<name>`; primitives in a second `Array4` with its own enum; read-only matter fields (`Df`, `Sfd`, `vfd`, `pf`, ...) |
| R3 | stencil reads | offsets up to 3; per-component pointers with int strides | `arr(i+a, j+b, k+c, comp)`, offsets up to 3 |
| R4 | outputs | stores to the rhs array; a later loop may read and update (overwrite) an rhs component an earlier loop wrote | 22 values in registers (`Real&` out-parameters or a returned struct), each assigned exactly once, in a fixed documented order; no stores |
| R5 | coordinates | cell centers `(i + 1/2) dx` | nodes `prob_lo[d] + i*dx[d]` |
| R6 | parameters | class members filled in the constructor from GRTeclyn's parameter structs | members of a struct passed by const reference (`parameters.<name>`) |
| R7 | functions | three DSL functions = GRTeclyn's three kernels | two: `rhs` (no KO) and `ko` |
| R8 | splits | hard splits into several kernels with scratch temporaries (tuning) | later: sequential inline functions |
| R9 | printing | exact literals, `Abs`, expanded integer powers | the same; π exact; Rationals exact; `pow` with rational exponents documented |
| R10 | guards | ternary floors | a guarded term `factor > 0 ? factor*S : 0` never computes `factor * 0` or `0 * Inf` |
| R11 | flags | double; nvcc with `--use_fast_math` (contraction on) | double; nvcc without fast math (contraction on by default), and g++ |
| R12 | provenance | md5-frozen, deterministic header; EE commit recorded beside it (`PROVENANCE`) | the same |
| R13 | stencil width | settable | settable (asserts 5) |
| R14 | CPU | compiles in AMReX CPU builds | `HOST_DEVICE`, callable from a CPU harness |

## 4. Architecture

EE's four-layer pattern, with one base class per layer and a few explicit, required strategies:

```
frontend/dsl/amrex/
  amrex_frontend.py    AMReXModule, AMReXFunction; field, output and parameter declarations
  layout.py            ArrayParam, AccessScheme (StridedPointer, Array4Call)
  schemes.py           CoordinateScheme, ParamScheme
  naming.py            component naming rules (moved here from the GRChombo backend: no cross-backend import)
generators/amrex/
  analysis.py          LoopAnalysis: classification and checks, shared by both modes
  modes.py             KernelLaunchMode, PointFunctionMode
  guards.py            the guard-folding CSE preprocessor and the post-CSE guard check (section 7)
emit/code/amrex/       tree nodes; AMReXSympyVisitor (lowering); AMReXCppVisitor (printing, section 6)
wizards/amrex_wizard.py
```

**Classification is by declaration, never by name.** In version 1, a symbol was a field if its name looked like
`<state><digits>`. That broke MHDuet's names (`gtd_xx`), its read-only fields, and its parameter `trK0`, which
printed as a state load. Version 2 classifies every symbol by its declaration:
- A **field binder**, a required module-level callable `binder(base_name) -> FieldBinder | None`, is consulted by
  every declaration. It sees the declared base name (`gtd`, `Df`, `h`). The returned object maps each component, as
  the module's naming rule names it (`gtd_xy`, `h12`; one name per symmetric pair, the canonical one), to an array
  and a component expression (`simflowny_vars::gtd_xy`, `c_h12`). An equation that assigns a bound field raises. `decl` and the `rhs=` shorthand consult it, so a frontend-agnostic recipe that
  declares its fields with a plain `cc.decl` (MHDuet's `Df`, `vfd`, ...) gets them bound by the subclass, without
  editing the recipe.
- `module.field(name, indices, array=..., component=...)` is an explicit declaration, for recipes written against
  this backend. Declaring the same name both ways is an error.
- `module.output(name, indices, ...)` declares an output: an rhs component in kernel mode, a register output in point
  mode. `decl(name, ..., rhs=R)` stays as a shorthand for a field X whose output is R, written to X's component.
- `add_param` declares parameters. `decl` with no binding declares temporaries.
- Every name has one classification. Declaring a name as two kinds is refused, as is assigning a field or a
  parameter. So is any name the generated code uses itself: the cell indices, the spacing, the array arguments,
  each mode's locals (`rhs_ptr`, `rn`, `tile_ptr`, `tn`, `out`, ...), the access scheme's (`state_ptr`, `sy`, `sz`,
  `sn`, the `in_` component pointers), the coordinate and parameter schemes' arguments. One output is the `rhs=` of
  at most one field. Every declared output must be written by some function.
- An output that the loop writing it (or a later loop) reads again is computed into a local, then stored; EE's
  `EqnList` counts such an lhs as a temporary, so the analysis classifies every lhs by declaration instead.
- The coordinates `x`, `y`, `z` come from the `CoordinateScheme`.
- Anything else in an equation, including EE's `DY`, `DZ`, `DT` and `t` unless the frontend binds them, makes the
  generator raise.

**Shared analysis.** `LoopAnalysis` covers each loop of each function, in launch order. It computes:
- the field reads at each offset;
- the reads of values written by an earlier loop or function;
- the coordinates used;
- the temporaries in baked order, with recycling;
- the tile temporaries.

It enforces:
- no read of a value this loop writes;
- no read of an output that no earlier loop writes;
- stencil reads of fields only;
- |offset| <= `max_stencil_offset` (a constructor argument; 3 for both codes);
- no global temporaries;
- tile temporaries never treated as outputs.

## 5. Extension points

Each code's subclass must supply these; the base class has no defaults for them. A forgotten coordinate scheme would
silently move MHDuet's coordinates, which changes its results.

1. **Arrays and fields** (R2, R3). `ArrayParam(name, cpp_type, access)` declares an array argument with its own
   `AccessScheme`.
   - `StridedPointer` (GRTeclyn): per-component `const Real*` and int strides, `in_X[a + b*sy + c*sz]`.
   - `Array4Call` (MHDuet): `arr(i+a, j+b, k+c, comp)`.

   Each field names its array and its component expression (`c_h12`, `simflowny_vars::gtd_xy`). Component naming
   for tensors is a rule in `naming.py`, selectable per module: GRChombo-style digits, or a callable (MHDuet).
2. **`CoordinateScheme`** (R5): one C++ expression per axis in `i`, `j`, `k` and the spacing.
   - `CellCentered`: `(i + 0.5) * DX`.
   - `NodeCentered`: `prob_lo[0] + i * dx[0]`.
   - Only the coordinates actually used are emitted.
3. **`ParamScheme`** (R6). It also owns the spacing, so dx has a single owner, used by both the stencils and the
   coordinates.
   - `MemberParams` (GRTeclyn): class members assigned in the constructor after a C++ prelude; spacing as members
     `DX`, `DXI`, `DYI`, `DZI`.
   - `StructParams(arg, cpp_type, member=lambda p: p)`: parameters print as `parameters.<name>`, with the name
     unchanged by default; spacing from host arguments `dx[0]`, `dx[1]`, `dx[2]`, inverted once into locals.
4. **Emission mode** (R1, R4, R7, R8, R14). The mode fixes the output style. Only these combinations are supported;
   any other combination raises.

   | mode | outputs | splits | access | params |
   |---|---|---|---|---|
   | `KernelLaunchMode` | stores to an rhs array; overwrites (`X' = X + ...`) reload X before any store | hard splits, scratch MultiFab on `The_Async_Arena()` | any | `MemberParams` |
   | `PointFunctionMode(outputs='refs'\|'struct')` | `Real&` out-parameters, or a returned POD struct; each output assigned exactly once (checked) | refused (a later version adds sequential inline functions) | any | `StructParams` |

   Point mode details:
   - **Fixed signature order**, the same for every function of a module: `i, j, k`; the module's arrays, in
     declaration order; the coordinate and spacing arguments (`prob_lo`, `dx`); the parameter struct; then the
     outputs this function assigns, in output-declaration order. The order is documented in the header comment, and
     a test pins it. Arguments a function does not use are marked `[[maybe_unused]]`.
   - **Each DSL function is one self-contained inline function.** A function may read only fields. A value computed
     by another function reaches it only as a field the host passes in.
   - **MHDuet's `ko` function** outputs the KO increments, each a declared output (`ko_<field>`). The recipe today
     adds KO into the RHS (`ko=True`), so a separate KO function is new recipe code, for MHDuet's agent and its
     reviewers.
   - **No joining of functions.** The version-1 `fuse_functions` option is dropped: under `promote_none`, shared
     temporaries are duplicated as Locals with the same CSE names, so joining the bodies would redeclare them, and
     sharing would need Global temporaries, which both modes refuse. A recipe that wants shared CSE between rhs and
     KO declares one function with both outputs.
   - **Name safety:** point-mode locals and temporaries get a prefix (`ee_`, configurable), because the host
     translation unit may have `using namespace` directives and macros.
5. **Printing hooks** (section 6). `AMReXVisitor.float_literal` and `AMReXVisitor.pow` print literals and powers.
   `AMReXModule.sympy_visitor` returns the visitor that lowers sympy to the code tree, so a subclass may override how
   constants, functions, `Abs`, `Max`/`Min`, conditionals and booleans lower. `AMReXModule.default_cse_optimizations`
   gives the CSE list a bake uses when its options name none; the guard fold is always inserted first. (Version 3.2
   named eight visitor methods; the implementation groups the six lowering ones in the sympy visitor.)
6. **Naming and banner.** Class or namespace, function and header names. The banner is fixed text that the recipe
   supplies: recipe path, arguments and purpose. It never contains the EE commit or a date, so the header is
   deterministic and its md5 is stable across EE commits that do not change the output. The EE commit, Python and
   sympy versions go in a separate `PROVENANCE` file, which the host's apply script writes, as GRTeclyn's does now.

## 6. Printing (both codes)

- **Floats.**
  - A sympy `Float` with at least 53 bits of precision prints as `repr` of its correctly rounded double, the
    shortest round-trip decimal: `1e-06` prints exactly.
  - A `Float` with less than 53 bits makes the generator raise. Its value is a low-precision approximation, and
    printing it would bake that error in silently. A recipe must use full-precision Floats or Rationals.
- **Rationals.** `p/q` prints as `(p.0 / q.0)` when p and q are both exactly representable (|p|, |q| <= 2^53), which
  is one rounding at compile time. Otherwise it prints the correctly rounded double of p/q (`fractions`), which is
  also one rounding.
- **Constants.** π and E print as their correctly rounded doubles (`3.141592653589793`, `2.718281828459045`).
  Version 1's claim that π was 7 ulp low applies only to EE's generic printers; this path was already exact, which
  reviewer A verified. Any other constant makes the generator raise.
- **Powers** (pinned by a printing test):

  | exponent | prints |
  |---|---|
  | 0 | `1.0` |
  | n = 1 to 4 | `(x * ... * x)`, n factors |
  | -n, n = 1 to 4 | `(1.0 / (x * ... * x))` (GRTeclyn's header has six such divisions) |
  | 1/2, -1/2 | `sqrt(x)`, `(1.0 / sqrt(x))` |
  | any other (integer beyond 4, other rationals, symbols) | `pow(x, p)`, p the exact double or the symbol: `pow(chi_max, -1.5)`, `pow(lapse, lapse_power)` |

  This is today's `foreign/grteclyn` rule with the integer range limited to 4 (no GRTeclyn or MHDuet expression has
  a larger integer power), so GRTeclyn's header is unchanged. Version 1's cap of 8 and 3/2 rule are dropped.
- **`Abs`** prints `amrex::Math::abs(x)`. **`Max`/`Min`** (reachable only from sympy expressions that bypass EE's
  index visitor, which rejects them in `add_eqn` today) print `amrex::max(a, b)`/`amrex::min(a, b)`, with every
  operand printed as a `Real` (`amrex::Real(1)`, not `1`). `sympy.Max` and `Min` are not `sympy.Function`s, so the
  base visitor's function-rewrite table cannot reach them; the AMReX sympy visitor handles them itself before
  delegating to the base visitor. A printing test pins this.
- **Booleans.** Relations print as C++ comparisons; `And`, `Or` and `Not` print `&&`, `||`, `!`.
- **Standard functions.** `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh` and `erf` print
  unqualified. Anything else raises.
- **Piecewise** prints as a C++ conditional expression. Only the selected operand is evaluated; section 7 covers what
  CSE can hoist out of it.
- **Includes.** Kernel mode: exactly today's list, `AMReX_Array4.H`, `AMReX_MultiFab.H`, `AMReX_REAL.H`,
  `<cstddef>`, `<type_traits>` (`AMReX_MultiFab.H` brings in `amrex::Math::abs`; adding a line would change
  GRTeclyn's md5). Point mode: `AMReX_Array4.H`, `AMReX_REAL.H`, `AMReX_GpuQualifiers.H`, `AMReX_Array.H`
  (`GpuArray`), `AMReX_Algorithm.H` (`amrex::max`), `AMReX_Math.H` (`amrex::Math::abs`), `<cmath>`.
- **Contraction.** The printer never relies on any particular evaluation order beyond C++'s. Results depend on FMA
  contraction at roundoff level only, which nvcc does by default (`--fmad=true`) with or without fast math.
  Validation runs with the production flags and, as an envelope, without contraction.

## 7. Guards (R10)

MHDuet's recipe writes KO as `Piecewise((f, 0 < f), (0, True)) * S`. A lazy ternary around the factor alone still
computes S and multiplies it by 0, which is 0 × Inf when S is not finite: the version-1 claim was false for this
recipe. Version 2 handles guards in two steps, both inside the backend.

1. **Fold before CSE.** `AMReXModule` adds a CSE preprocessor to its default bake options. It uses the existing
   `cse_optimizations` hook, so there is no intermediate-layer change. The preprocessor rewrites every product
   `Piecewise((a, c), (0, True)) * S` into `Piecewise((a*S, c), (0, True))`, and likewise with the zero branch first.
   - For finite S the value is unchanged.
   - Where the guard is false, the result is 0 instead of `0*S`. That is the intended guard semantics, and what
     MHDuet's `if (factor > 0)` does.
   - The fold needs no recipe change, so MHDuet's KO form stays as it is. Its separate `ko` function, though, is new
     recipe code (section 5.4).
2. **Check after CSE.** The correctness property is: no product of a zero-branch Piecewise with a non-constant factor
   survives into the emitted code. With it, a false guard yields exactly 0 whatever any other temporary holds.
   - The generator follows temporaries through each equation and raises on any surviving product of a Piecewise that
     has a zero branch in any position (two or more branches) with a non-constant factor F, whether the Piecewise is a direct factor or reaches the product through a
     temporary (a recipe's `g = Piecewise(...)`, then `g*S`). This is an unconditional hard error in both modes, so a
     bypassed fold cannot pass silently. A constant multiple is allowed and tested: sympy's CSE preprocessing turns a
     folded guard into `3*(2*a - P)` and then `-3*x0`, which is still exactly 0 when the guard is false. The check
     follows a guard through products, constant multiples and temporaries holding such products, not through sums:
     `DXI*(A + P)`, which the default `BasicCollectAfter` preset can produce, is safe and is tested.
   - A guard held in a temporary (`g = Piecewise(...)`, then `g*S`) is not folded and therefore raises. A recipe
     writes the guarded product directly, as MHDuet's does. Guards with three or more pieces are checked but not
     folded, so they raise too.
   - What CSE may still do is compute some arithmetic of a guarded branch unconditionally. sympy's CSE places all
     temporaries at top level, so the shared stencil loads and, for example, the center weight `-20*f0` of MHDuet's
     KO are computed whether or not the guard holds. That costs work but cannot change the result: those temporaries
     may hold Inf, but a false guard selects 0 without reading them. The generator reports, as a warning and never an error, the
     arithmetic temporaries used only (transitively) inside guarded branches, so the cost is visible. Loads are never
     reported. Raising here would fire on nearly all of MHDuet's 22 guards (the reviewers' emitted code shares KO
     stencil sums with the second-derivative stencils), and the cost is zero in production, where the factors are
     positive.
   - Skipping that work too would need an intermediate-layer option that keeps CSE out of guarded branches: an
     EE-core change and a `[DECISION]` for Max, only if MHDuet's timings call for it.

   R10 accordingly means: a false guard contributes exactly +0.0, and no `0 * S` reaches an output. Where MHDuet skips
   the statement, a -0.0 stays -0.0; here it becomes +0.0, which compares equal and does not affect validation. The skipping of the
   branch's own work is not guaranteed.

3. **The fold cannot be bypassed.**
   - `AMReXModule.bake` inserts the fold first into whatever `cse_optimizations` list arrives, including one from a
     recipe or tuner (`bake` replaces the default list). It goes first because `cse_basic`'s `factor_terms` would
     otherwise split `Mul(Piecewise, S)`, folding with different roundoff.
   - The fold is a custom rewrite of `Mul` nodes. It is not sympy's `piecewise_fold`, which also folds the enclosing
     sum and duplicates the whole right-hand side.
   - With `do_cse=False`, it applies the fold directly to every equation.
   - The fold rewrites only the matched products. Every other node is returned unchanged (the identical object), so
     a header without guards is byte-identical.
   - It runs to a fixpoint, so a product with several guard factors folds completely.
   - Tests cover both bypass paths (a recipe-supplied list, `do_cse=False`), MHDuet's exact KO form, several guard
     factors, and the post-CSE check firing on an unfolded product.

Version 1's statement-level code sinking (its section 9(a)) is dropped. It was unsound with recycled temporaries,
which reassign shared C++ variables, and with C++ block scope.

## 8. Behavior both codes rely on

- **Kernel mode:** functions launch in creation order, and loops within a function in order. A loop sees every rhs
  component and tile temporary that earlier loops wrote, and reloads happen before any store.
- **Point mode:** functions are independent; outputs are assigned exactly once.
- **Determinism:** the same recipe, EE commit and Python and sympy versions give a byte-identical header, for any
  `PYTHONHASHSEED` (tested).
- **Stencil width** is the frontend's `derivative_stencil_width`, unchanged. Reach is checked against
  `max_stencil_offset`.
- **Errors, never silent fallbacks:** unbound symbols, unsupported functions or operators, low-precision Floats,
  unsupported scheme combinations, a hard split in point mode, an unassigned or doubly assigned point output, and a
  zero-branch Piecewise used as a product factor all raise.

## 9. Tests

Plain scripts in `unit_tests/`. The first group runs without AMReX:
1. **Analysis:** one device function per loop; launch order; tile store and reload; reloads before stores;
   classification by declaration (a parameter named like a field, read-only fields, `gtd_xx`-style names); every
   refusal; the reach check.
2. **Each strategy and mode**, on toy modules, against expected fragments plus one golden header per mode, so any
   change in output shows as a diff.
3. **Printing:** π, E, `1e-6`, a low-precision Float (refused), large Rationals, the exponent table (-4, -2, -1, 2, 3, 4, 1/2,
   -1/2, -3/2), `Abs`, `Max` with an integer operand, And/Or/Not, Piecewise.
4. **Guards:** the folding rewrite on MHDuet's exact KO form, several guard factors, both bypass paths; the post-CSE
   error on an unfolded product (direct and through a temporary); the hoisting warning; a NaN in the KO stencil with a
   false guard gives 0; a golden
   output.
5. **Determinism:** the toy and the GRTeclyn header are byte-identical across `PYTHONHASHSEED` 0, 1 and 7 (about
   5 minutes, so it runs on mel5bash or under the local-heavy lock).

Required gates, run on mel5bash or in a qbd GPU job and recorded with each EE milestone:

6. **Compile:** both golden headers and GRTeclyn's CCZ4 header compile against real AMReX 26.10 (GRTeclyn's) and
   25.11 (MHDuet's), with `g++ -fsyntax-only` (CPU AMReX) and `nvcc -x cu -fsyntax-only` (CUDA 12.2 and GCC 11, sm_80,
   in a qbd job). Found by the gate: `StridedPointer` reads `Array4::stride`, which 25.11 does not
   have (25.11 has `jstride`, `kstride`, `nstride`; 26.10 has `stride`), so kernel-launch mode with `StridedPointer`
   needs a release that has it, such as 26.10.
   GRTeclyn uses 26.10. Against 25.11 the gate builds the point toy only (`make POINT_ONLY=TRUE`), which is MHDuet's
   mode with `Array4Call`.
7. **Differential:** a toy point function and a toy kernel class, compiled with g++ against CPU AMReX, run on random
   data. The data has anisotropic dx, an `Array4` with nonzero `begin`, and distinct enums per array. The results are
   compared with sympy's 50-digit evaluation of the baked equations, to 1e-13 relative.
8. **Regression:** GRTeclyn's headers through the subclass (section 10).

## 10. Migration

1. **New baseline first.** Merge master (`8e45a5b7`) into `foreign/grteclyn` with the current backend unchanged.
   - Regenerate the CCZ4 header and the six exploration variants (`bench/explore`, v0-v5).
   - Record their md5s, with the variant definitions in words (the tuning parameters as text, not lambda reprs), in a
     tracked file, `grteclyn/bench/explore/baseline.md`, as the new baseline.
   - If master's CSE changes alter the header, it is revalidated at level 1 before it replaces `27b43325`.
   - Then cherry-pick the `naming.py` move, so the AMReX frontend no longer imports from the GRChombo backend.
2. **`foreign/amrex`:** base classes, both modes, all strategies, the guard handling and the tests. The kernel-mode
   code is `foreign/grteclyn`'s current backend, moved and parameterized.
3. **`foreign/grteclyn` merges `foreign/amrex`.** `GRTeclynKernelClass(AMReXModule)` installs `StridedPointer`,
   `CellCentered`, `MemberParams` and `KernelLaunchMode`. The new-baseline md5s of the CCZ4 header and of v0-v5 must
   reproduce byte for byte. Since the banner is the recipe's and has no commit, this is achievable. Any difference
   blocks the merge until it is explained.
4. **`foreign/mhduet` merges `foreign/amrex`** and adds `MHDuetPointFunction(AMReXModule)`: `Array4Call` arrays
   (`un`, primitives), node coordinates, `StructParams`, `PointFunctionMode(outputs='refs')`, and its field,
   parameter and output declarations.
5. **Master:** `foreign/amrex` is the merge candidate, at Max's decision.

## 11. Review and approval

Twin harsh reviewers with a full design and code-quality review (CLAUDE.md rule 5). After both sign off, the design
goes to Max as a `[DECISION]`, and implementation starts only after his approval.

## 12. Version 1 review findings and their resolution

| finding | reviewer | resolution |
|---|---|---|
| R10 not met for MHDuet's `Piecewise(f) * S` | A B1, B 2 | section 7: fold before CSE (existing CSE-optimization hook), strict post-CSE check, NaN test |
| section 9(a) code sinking unsound (recycling, scope) | A B2, B 3 | dropped (section 7) |
| banner with `git describe` vs the md5 gate; `describe` not deterministic | A B3, B 5 | the commit goes in `PROVENANCE` only; the banner is the recipe's (section 5.6) |
| migration baseline lacks master's 6 commits; `subst_tensor_grchombo` import | A B4 | new baseline after merging master; all of v0-v5 must reproduce; `naming.py` (section 10) |
| field classification by name: `gtd_xx`, read-only fields, `trK0` printed as a load | A B5, B 1 | classification by declaration; `field()`/`output()`; unbound symbols raise (section 4) |
| point-mode contract (cross-function reads, outputs assigned once, argument order) | A B5, B (non-blocking) | section 5.4: functions independent; exactly-once check; fixed, pinned order |
| `fuse_functions` redeclares, shares nothing | A, B 4 | dropped; one function with both outputs instead (section 5.4) |
| too many axes; output scheme fixed by mode; dx owned twice; AccessScheme per array | A | access per `ArrayParam`; outputs fixed by mode; spacing owned by `ParamScheme`; supported-combination table |
| defaults silently select GRTeclyn's schemes | B | no defaults for the required strategies (section 5) |
| printing: low-precision Floats, big Rationals, E, Max/Min, int operands, And/Or/Not, includes, power rules | A, B | section 6 |
| contraction claim overstated | B | section 6 (contraction) |
| `DY`/`DZ`/`DT`/`t` unhandled | A | unbound symbols raise (section 4) |
| `StructParams` example added `p_` | A | name unchanged by default (section 5.3) |
| `using namespace` and macros in MHDuet's translation unit | A | prefixed locals in point mode (section 5.4) |
| maximum stencil offset | A, B | `max_stencil_offset` check (section 4) |
| compile test optional; differential test needs nonzero `begin`, anisotropic dx, distinct enums; determinism | A, B | required compile gate with g++ and nvcc; differential test as specified; `PYTHONHASHSEED` test (section 9) |

Version 2 findings (round 2) and their resolution:

| finding | reviewer | resolution |
|---|---|---|
| the fold can be bypassed (`bake` replaces the CSE list; `do_cse=False`); the post-CSE check only looked inside branches | A R2-B1, B R2-B2 | the fold is always inserted first, or applied directly without CSE; an unconditional error on any zero-branch Piecewise used as a product factor, direct or through temporaries; bypass tests (sections 7.2, 7.3) |
| the strict hoisting check fires on MHDuet's own KO and misses the real hazard | A R2-B2, B R2-B1 | no longer an error: a warning about arithmetic only; R10 restated (section 7.2) |
| the power table sends negative integers to `pow`, breaking the md5 gate | A R2-B3, B R2-B3 | a full table, with -1 to -4 as `1.0/(product)`, pinned by a test (section 6) |
| fields declared with a plain `cc.decl` in frontend-agnostic recipes | A, B | a required module-level field binder (section 4) |
| which outputs a function's signature lists; a fixed signature per module | A | section 5.4 |
| a fold through `piecewise_fold` duplicates the RHS | A | a custom `Mul` rewrite (section 7.3) |
| baseline variants and md5s untracked | A | `grteclyn/bench/explore/baseline.md` (section 10) |
| `MemberParams` in point mode; Max/Min reachability; `AMReX_Math.H`; AMReX 25.11; the cost of the PYTHONHASHSEED test; the KO function is recipe code | A, B | sections 5.4, 6, 9 and 7 |
