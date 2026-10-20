# AthenaK backend

Generates a C++ header with one host function that runs Kokkos loops through AthenaK's `par_for` (athena.hpp) over
the (m, k, j, i) cells of a MeshBlockPack. It is meant to replace a loop of AthenaK, such as the Z4c right-hand side
`Z4c::CalcRHS<NGHOST>`, with no edits to the generated code.

Modules: `frontend/dsl/athenak/athenak_frontend.py` (`AthenaKKernel`), `generators/athenak_generator.py`,
`emit/code/athenak/` (code tree, sympy lowering, C++ printer), `wizards/athenak_wizard.py`. Example recipe:
`recipes/foreign/athenak/z4c_athenak.py`. Tests: `unit_tests/test_athenak_backend.py`.

## Writing a recipe

```python
kk = AthenaKKernel("MyKernel", namespace="z4c::ee",
                   args=[("MeshBlockPack *", "pmy_pack")],                 # the C++ function's parameters
                   preamble=["auto &u0 = pmy_pack->pz4c->u0;", ...],       # host code: views, bounds, nmb, opt
                   loop_bounds="0, nmb - 1, ks, ke, js, je, is, ie",
                   inv_spacing=("1/size.d_view(m).dx1", ...),              # device expressions -> DXI, DYI, DZI
                   checks=[AthenaKCheck("opt.x != 0", "x is not implemented")],
                   derivative_stencil_width=7)
kappa = kk.add_param("kappa", init="opt.damp_kappa1")                      # const Real kappa = ...; captured
g = kk.decl("g", [li, lj], symmetries=[(li, lj)], view="u0", enum="z4c::Z4c::I_Z4C_G")  # gxy -> I_Z4C_GXY
fn = kk.create_function("rhs", label="EE rhs")                             # one par_for, Kokkos label
fn.add_eqn(...)
kk.bake()
AthenaKWizard(kk, out_dir="...").generate_header(banner="// ...")
```

- A grid variable needs `view` (a DvceArray5D indexed (m, n, k, j, i)) and `enum` (its index n, or for a tensor the
  prefix of its components' indices); its components are named by `subst_tensor_xyz`. Anything else is a
  temporary. Reads become `view(m, enum, k + z, j + y, i + x)` at their stencil offsets.
- Each function is one `par_for`, run in creation order, so a later function may read an earlier one's output at
  any offset. A function may also update its own outputs at its own cell (`kk.overwrite`), as `rhs += ...` does.
- Rejected: a stencil read of a variable written in the same function (a race between threads), hard splits
  (`split_loop()` inside a function; create another function instead), and global temporaries.
- Math is printed with `Kokkos::` functions; Piecewise becomes `?:` with `<`, `>`, `<=`, `>=` conditions.
