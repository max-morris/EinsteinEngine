# A shared AMReX backend: design

Status: design for review (2026-10-20). Branch `foreign/amrex`, from master `8e45a5b7`. Nothing in this document is
implemented on this branch yet. Max's decision (2026-10-20): "I want a shared, subclassable/extensible amrex
backend, as anything else requires a lot of repeated work."

## 1. Purpose

EE generates kernels for two AMReX codes, GRTeclyn and MHDuet, and both need `AMREX_GPU_DEVICE` code on
`amrex::Array4`. They differ in nearly every interface detail: how a kernel is invoked, where fields live, how
coordinates and parameters are spelled, and where results go. This backend implements once what they share: the
DSL-to-C++ lowering, equation classification, temporaries, stencil reads, splits, and correct, deterministic
printing. It gives each code a small subclass for the rest.

**Non-goals.**
- No change to EE's DSL or intermediate layer. Every requirement below is met at the frontend/generator/emit level.
  Section 9 lists the one place where that may not suffice.
- No timing instrumentation. Timers belong to the host code's patch (GRTeclyn's `FKSTimers.hpp`), not to generated
  code.
- No MPI, ghost filling or boundary conditions. The host code owns those.

## 2. What exists

- **EE master (`8e45a5b7`):** no AMReX code. The backends are CarpetX (`cpp_carpetx_*`), vanilla F90 and Cactus. The
  generic C++ printers print floats with 15 significant digits, and π comes out 7 ulp low. MHDuet's separate master
  PR (`ee-floatfix`, Max's decision) fixes that; this backend prints its own literals in any case (section 6).
- **`foreign/grteclyn` (`0bd5e1cc`):** a working, validated, twin-reviewed AMReX backend, specific to GRTeclyn's
  needs:
  - `frontend/dsl/amrex/amrex_frontend.py` (`AMReXKernelClass`), `generators/amrex_generator.py`,
    `emit/code/amrex/` (tree, sympy visitor, C++ visitor) and `wizards/amrex_wizard.py`;
  - `unit_tests/test_amrex_backend.py`.

  It emits a class with one `AMREX_GPU_DEVICE` function per loop and a host `compute()` that launches one
  `ParallelFor` per loop. It supports hard splits, with tile temporaries in a stream-ordered scratch MultiFab. Its
  generated CCZ4 header (md5 `27b43325`) passes GRTeclyn's level-1 validation.
- **`foreign/grchombo`:** a CPU SIMD backend for Chombo's `Cell`/`BoxPointers`. It is not AMReX and is out of scope
  here, apart from two lessons: map `Abs` explicitly (GRChombo's `simd` `abs` bug), and print literals with `repr`.
- **`foreign/mhduet`:** a recipe only. Its backend waits for this design.

The shared backend starts from the `foreign/grteclyn` code. That code becomes the base classes plus the
kernel-launch mode, and the point-function mode is added.

## 3. Requirements

| # | requirement | GRTeclyn | MHDuet |
|---|---|---|---|
| R1 | invocation | a header-only class; `compute(MultiFab& rhs, const MultiFab& state)` launches one `ParallelFor` per loop, in order; the host synchronizes | a header-only `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE` function per DSL function, called inside a host-written fused `ParallelFor`; no launch, scratch, shared memory or atomics |
| R2 | field storage | state: one `Array4<Real const>`, component `c_<name>`; rhs: one `Array4<Real>`, same enum | state: `Array4` parameter `un` (differs per RK stage), components `simflowny_vars::<name>`; primitives: a second `Array4`, another enum |
| R3 | stencil reads | |offset| <= 3; access through per-component pointers with int strides, as GRTeclyn's derivative classes do (register-friendly) | `arr(i+a, j+b, k+c, comp)` with |a|, |b|, |c| <= 3 |
| R4 | outputs | stores to the rhs array; a later loop may read an rhs component written earlier and update it (overwrite) | the RHS values in registers, as `Real&` out-parameters or a returned struct with named members, in a fixed documented order; no memory stores |
| R5 | coordinates | cell centers `(i + 1/2) dx` | nodes `prob_lo[d] + i*dx[d]` (`GpuArray<Real,3>`) |
| R6 | parameters | class members filled in the constructor from GRTeclyn's own parameter structs | members of a struct passed by const reference (`parameters.p_kappa_z1`), not scalar arguments |
| R7 | functions per recipe | three DSL functions (GRTeclyn's three kernels), in order | two (`rhs` without KO, `ko`), each inline; shared CSE only if inlined together |
| R8 | splits | hard splits into several kernels, with tile temporaries in a scratch MultiFab (tuning) | later: optional splitting into several sequential inline functions |
| R9 | printing | round-trip floats, `Abs` mapping, integer powers expanded | the same, plus π exact (`M_PI` or 17 digits), Rationals as exact divisions, `pow` with rational exponents documented |
| R10 | conditionals | ternary for `std::max`-like floors | real branches or ternaries, never `factor * 0` (skip work, no 0 × Inf) |
| R11 | precision and flags | double; production uses `--use_fast_math` (FMA only, for double) | double; no fast math; must build with nvcc 12.2/gcc 11 and with g++ alone |
| R12 | provenance | banner with recipe, EE commit; deterministic output, md5 frozen | the same |
| R13 | stencil width | settable (4th order now, 6th later) | settable; MHDuet asserts 5 |
| R14 | CPU use | the class compiles in AMReX CPU builds (level 1 runs on CPU too) | `HOST_DEVICE` so a CPU harness can call the function |

## 4. Architecture

The backend keeps EE's four-layer pattern. Each layer has one base class, and the code-specific parts are
*strategies* that a subclass supplies. A code needs only a small subclass of the frontend, which installs its
strategies, and seldom a generator or visitor subclass.

```
frontend/dsl/amrex/            AMReXModule (was AMReXKernelClass), AMReXFunction
  layout.py                    GridLayout, ArrayParam, FieldBinding
  schemes.py                   CoordinateScheme, ParamScheme, OutputScheme
generators/amrex/              AMReXGenerator: per-loop analysis (shared) + assembly by EmissionMode
  analysis.py                  LoopAnalysis: classified reads/writes, temporaries, tiles, rhs reloads, coordinates
  modes.py                     EmissionMode, KernelLaunchMode, PointFunctionMode
emit/code/amrex/               tree nodes; AMReXSympyVisitor (lowering); AMReXCppVisitor (printing)
wizards/amrex_wizard.py        AMReXWizard: writes the header and its provenance banner
```

**Shared analysis** (from `foreign/grteclyn`'s generator, unchanged in substance). For each loop of each function,
in launch order, it computes:
- the grid reads at each stencil offset;
- the reads of values written by an earlier loop or function (rhs reloads in kernel mode, in-register values in
  point mode);
- the coordinates used;
- the temporaries in baked order (with recycling), and the tile temporaries written and read across loops.

It also checks every invariant the reviews asked for: no read of a value this loop writes; no read of an rhs
component no earlier loop writes; no stencil read of a non-field; no unknown symbol; no global temporaries; a tile
temporary that its writer does not read goes to the scratch, not to the outputs.

## 5. Extension points

All of these are constructor arguments of `AMReXModule` with defaults, so a subclass can either pass them or
override the corresponding method.

1. **`GridLayout`: which array holds a field, and how it is indexed** (R2, R3).
   - `ArrayParam(name, cpp_type, read_only)` declares a function parameter (`state`, `rhs`, `un`, `prim`), and
     `FieldBinding(array, component_expr)` maps a DSL grid variable component (`h12`, `gtd_xy`) to it.
   - The default binding is GRTeclyn's: `state` with component `c_<name>`, and the rhs of state `X` in `rhs` with
     component `c_<X>`. MHDuet supplies a callable, `lambda comp: FieldBinding('un', f'simflowny_vars::{comp}')`,
     plus the primitives array.
   - Naming of tensor components stays a frontend `substitution_rule`, as now (`subst_tensor_grchombo` gives `h11`;
     MHDuet's gives `gtd_xx`).
2. **`AccessScheme`: how a stencil read prints** (R3).
   - `StridedPointer` (GRTeclyn; today's code): one `const Real*` per component, int `sy`/`sz` strides, loads
     `in_X[a + b*sy + c*sz]`.
   - `Array4Call` (MHDuet): `arr(i+a, j+b, k+c, comp)`.
   - The scheme also prints the per-function preamble it needs (pointers, strides).
3. **`CoordinateScheme`** (R5): an expression template per axis. `CellCentered()` gives `(i + 0.5) * dx`, with dx a
   member. `NodeCentered(prob_lo='prob_lo', dx='dx')` gives `prob_lo[0] + i * dx[0]`. A subclass can supply any
   C++ expression in `i`, `j`, `k`. The DSL symbols `x`, `y`, `z` map to it, and only used coordinates are emitted.
4. **`ParamScheme`** (R6).
   - `MemberParams` (GRTeclyn; today's code): class members assigned in the constructor body after a C++ prelude.
   - `StructParams(arg='parameters', type='const Simflowny_par&', member=lambda p: f'p_{p}')`: one argument, and each
     parameter prints as `parameters.p_kappa_z1`.
   - `dx` is part of this scheme: `DXI = 1/dx` members (GRTeclyn), or `1.0 / dx[0]` hoisted once into a local
     (MHDuet).
5. **`OutputScheme`** (R4).
   - `StoreToArray` (GRTeclyn): writes `rhs(i,j,k,c)` through the cell pointer, and resolves overwrites `X'`
     (read X at the top of the function, store once).
   - `OutParams(order=[...])`: `Real&` parameters in a fixed, documented order with fixed names, assigned once.
   - `ReturnStruct(name, members=[...])`: a POD struct returned by value.
   - Point mode supports overwrites by keeping the overwritten value in a local.
6. **`EmissionMode`: the overall shape** (R1, R7, R8, R14).
   - `KernelLaunchMode` (GRTeclyn; today's code):
     - a class, one `AMREX_GPU_DEVICE AMREX_FORCE_INLINE` member function per loop, and `compute()` launching one
       `ParallelFor` per loop over the valid cells, in creation order;
     - hard splits allowed, with tile temporaries in a scratch MultiFab on `The_Async_Arena()`;
     - the host synchronizes;
     - `static_assert(std::is_trivially_copyable_v<...>)`.
   - `PointFunctionMode` (MHDuet):
     - free functions in a namespace, `AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE`, one per DSL function, with the
       array, index, coordinate, parameter and output arguments that the schemes declare;
     - no memory but the input arrays;
     - a hard split is refused for now (R8 "later" becomes sequential inline functions passing temporaries as
       arguments);
     - two DSL functions (`rhs`, `ko`) are two inline functions.
     - Option `fuse_functions=True` emits them as one function with both outputs, so global CSE can share work
       (R7: "shared CSE only if both are inlined together"). Without it, EE's global CSE must not share temporaries
       across them, which the default `promote_none` already guarantees: shared temporaries become Local and are
       duplicated.
7. **Printing hooks**, in the C++ visitor (R9, R10, R11; section 6): `float_literal`, `pi_literal`,
   `function_name(StandardizedFunctionCallType)`, `abs_call`, `pow`, `conditional`. These are subclassable methods,
   with the defaults below.
8. **Naming and provenance** (R12): class or namespace, function names, header name, banner text. The wizard writes
   the banner from the recipe path, its arguments and the EE commit (`git describe --always --dirty`), and nothing
   time-dependent, so output stays deterministic.

## 6. Printing (defaults, for both codes)

- **Floats:** `repr(float(x))`, the shortest decimal that round-trips to the same double. A sympy `Float` with more
  than 53 bits is rounded to the nearest double once, then printed, so 1e-6 prints as `1e-06`, exactly.
- **π:** printed as the nearest double, `3.141592653589793`, the exact `M_PI`. Any other irrational constant must be
  representable exactly or generation fails; `E` prints as `2.718281828459045` by the same rule.
- **Rationals:** `(p.0 / q.0)`, an exact division in double, evaluated at compile time. Sympy has already
  distributed them over sums, a roundoff-level reordering that MHDuet's justification documents.
- **Integer powers** up to an exponent of 8 are expanded into products: `x*x*x`; a negative exponent gives
  `1.0/(x*x*x)`.
- **Half-integer powers:** `sqrt(x)`, `1.0/sqrt(x)`, and `x*sqrt(x)` for 3/2. Other rational exponents print
  `pow(x, p)` with p the exact double of the rational: `pow(chi_max, -1.5)` for -3/2. The design documents this, and
  a test pins it.
- **`Abs`:** `amrex::Math::abs(x)`. `Max`/`Min` print `amrex::max`/`amrex::min` (both are device-safe).
- **Standard functions:** unqualified `sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `erf`, which
  resolve to the device overloads under nvcc and to `<cmath>` under g++. Any other function makes the generator
  raise.
- **Piecewise** becomes a C++ conditional expression, `(c ? a : b)`. It evaluates only the selected operand, so
  `factor * 0` never appears: a guarded term costs nothing, and cannot turn 0 × Inf into NaN, provided its
  subexpressions were not hoisted out of the branch (section 9).
- **No fast-math assumptions.** The emitted code is correct under IEEE double with or without contraction. Validation
  covers both flag sets.

## 7. Behavior both codes rely on

- **Launch order and visibility (kernel mode).** Functions launch in creation order, and loops in order. A loop sees
  every rhs component and tile temporary that earlier loops wrote. Reloads happen before any store.
- **Determinism.** The same recipe, EE commit and Python and sympy versions give a byte-identical header.
  `foreign/grteclyn`'s apply script records the md5.
- **Stencil width** is a frontend constructor argument (`derivative_stencil_width`), passed through unchanged. Any
  offset is allowed; the reads print whatever offsets the stencils produce.
- **Errors, never silent fallbacks:** unsupported operators or functions, a non-field under a stencil, a hard split
  in point mode, and unbound variables.

## 8. Tests

Plain scripts in `unit_tests/`, like the existing ones, runnable without AMReX:
1. **Analysis** (from `test_amrex_backend.py`): one device function per loop, launch order, tile store and reload,
   rhs reload before store, coordinates, the refusals.
2. **Each scheme**, on a toy module: `StridedPointer` against `Array4Call` reads; `CellCentered` against
   `NodeCentered`; `MemberParams` against `StructParams`; `StoreToArray` against `OutParams` against
   `ReturnStruct`; both emission modes. Each is checked against expected text fragments, plus one small golden header
   per mode, so any change in output is a visible diff.
3. **Printing:** π, `1e-6`, 2/3 and -3/2 powers, `Abs`, `Max`, Piecewise (a lazy ternary), integer powers, and a
   refused unknown function.
4. **Compile test** (run when `AMREX_HOME` is set, otherwise skipped): `g++ -fsyntax-only` of both golden headers
   against AMReX, which catches C++ errors without a GPU.
5. **Differential test** (run when a compiler is present): compile the toy point function with g++, evaluate it on
   random data, and compare with sympy's evaluation of the baked equations at 50 digits; agreement to 1e-13 relative.
   This is a small, generic version of the exact checks the reviews ran.
6. **Regression:** GRTeclyn's CCZ4 recipe must still generate md5 `27b43325` through the subclass (section 10), or
   every byte of difference must be explained and the header revalidated.

## 9. Open issue (may need Max): conditionals and hoisting

A C++ ternary is lazy, but EE's global CSE runs before emission and may hoist subexpressions of a Piecewise branch
(for example a KO stencil sum) into temporaries that are computed unconditionally. Then the work is no longer
skipped, and an Inf in the unselected branch's stencil is still computed (though not used: the selected value
stays finite, so no 0 × Inf appears in the result).

Two ways forward:
- **(a) Backend only (proposed):** at emission, sink each temporary whose every use lies inside one branch of one
  Piecewise into that branch, emitting the equation as `T x; if (c) { ...; x = a; } else { x = b; }`. This is code
  motion within the backend, valid because temporaries have no side effects. It restores both skipping and isolation
  for the common case (a guarded term built only from its own temporaries).
- **(b) Intermediate layer:** tell CSE not to extract across a Piecewise boundary. That is an EE-core change and a
  `[DECISION]`.

Proposal: implement (a), and fall back to (b) only if MHDuet's validation shows a guard whose work cannot be sunk.

## 10. Migration

1. **`foreign/amrex`:** the base classes, both modes, all schemes and the tests. The kernel-mode code is
   `foreign/grteclyn`'s current backend, moved and parameterized, with no behavior change.
2. **`foreign/grteclyn`:**
   - merges `foreign/amrex`;
   - its frontend becomes `GRTeclynKernelClass(AMReXModule)`, installing `StridedPointer`, `CellCentered`,
     `MemberParams`, `StoreToArray` and `KernelLaunchMode`;
   - the CCZ4 header must stay md5 `27b43325`;
   - the GRTeclyn-specific files disappear.
3. **`foreign/mhduet`:** merges `foreign/amrex` and adds `MHDuetPointFunction(AMReXModule)`, installing `Array4Call`,
   `NodeCentered`, `StructParams`, `OutParams` and `PointFunctionMode`, plus its field and parameter maps.
4. **Master:** `foreign/amrex` is the merge candidate, at Max's decision.

## 11. Review and approval

This design gets twin harsh reviewers with a full design and code-quality review (CLAUDE.md rule 5). After both sign
off, it goes to Max as a `[DECISION]`. Implementation starts only after his approval.
