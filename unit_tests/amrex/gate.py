#  Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
#
#  This file is part of the Einstein Engine (EinsteinEngine).
#
#  EinsteinEngine is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  EinsteinEngine is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
The AMReX backend's gates (docs/AMREX-BACKEND-DESIGN.md, section 9, items 6 and 7).
  python gate.py generate GEN_DIR    writes KernelToy.hpp and PointToy.H (toys.py)
  python gate.py compare RUN_DIR     compares the driver's outputs (main.cpp) with a 50-digit evaluation of the baked
                                     equations on the driver's own inputs; passes if every output's
                                     max|C++ - reference| / max|reference| <= 1e-13
run_gate.sh does both around an AMReX build of main.cpp.
"""

import sys
from collections import defaultdict
from pathlib import Path

import mpmath

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toys  # noqa: E402

from EinsteinEngine import AMReXWizard  # noqa: E402

TOLERANCE = 1e-13


def read(path: Path) -> dict[tuple[int, int, int, int], mpmath.mpf]:
    out = {}
    for line in path.read_text().splitlines():
        i, j, k, c, v = line.split()
        out[(int(i), int(j), int(k), int(c))] = mpmath.mpf(v)
    return out


def compare(name: str, outputs: dict[tuple[int, int, int, int], mpmath.mpf], reference: dict) -> bool:
    ok = True
    by_output: dict[str, list[tuple[mpmath.mpf, mpmath.mpf]]] = defaultdict(list)
    for (i, j, k), refs in reference.items():
        for comp, (index, ref) in refs.items():
            by_output[comp].append((outputs[(i, j, k, index)], ref))
    for comp, pairs in sorted(by_output.items()):
        scale = max(abs(r) for _, r in pairs)
        err = max(abs(o - r) for o, r in pairs) / scale
        status = 'ok' if err <= TOLERANCE else 'FAIL'
        ok &= err <= TOLERANCE
        print(f'{name} {comp}: {len(pairs)} cells, max|C++ - ref| / max|ref| = {float(err):.2e} ({status})')
    return ok


def main() -> int:
    if sys.argv[1] == 'generate':
        out = sys.argv[2]
        AMReXWizard(toys.kernel_toy(), out_dir=out).generate_header("// AMReX backend gate: kernel toy (toys.py)")
        AMReXWizard(toys.point_toy(), out_dir=out, extension='.H').generate_header(
            "// AMReX backend gate: point toy (toys.py)")
        return 0
    run = Path(sys.argv[2])
    print(f'tolerance: every output passes if max|C++ - ref| / max|ref| <= {TOLERANCE:g}')
    ok = True
    # Kernel toy: c_u = 0, c_w = 1; outputs u_rhs -> c_u, w_rhs -> c_w; dx 0.1, kappa 1.3. Absent in a point-only
    # build (main.cpp, GATE_POINT_ONLY).
    if not (run / 'kernel_outputs.txt').exists():
        print('kernel: not built (point-only gate)')
    else:
        cc = toys.kernel_toy()
        state = read(run / 'kernel_inputs.txt')
        outs = read(run / 'kernel_outputs.txt')
        dx = mpmath.mpf(0.1)
        reference = {}
        for (i, j, k, c) in outs:
            if c != 0:
                continue
            comps = toys.STATE_COMPONENTS
            vals = toys.values(cc, lambda v, o: state[(i + o[0], j + o[1], k + o[2], comps[v])],
                               {'kappa': mpmath.mpf(1.3)}, (dx, dx, dx),
                               tuple(mpmath.mpf(float((n + 0.5) * 0.1)) for n in (i, j, k)))
            reference[(i, j, k)] = {'u_rhs': (0, vals['u_rhs']), 'w_rhs': (1, vals['w_rhs'])}
        ok &= compare('kernel', outs, reference)
    # Point toy: un (u = 0, trK = 1), prim (Df = 0); outputs v_rhs -> 0, ko_u -> 1.
    cc = toys.point_toy()
    un, prim = read(run / 'point_un.txt'), read(run / 'point_prim.txt')
    outs = read(run / 'point_outputs.txt')
    dxs = tuple(mpmath.mpf(d) for d in (0.1, 0.2, 0.3))
    lo = (-1.5, 0.25, 2.0)
    reference = {}
    for (i, j, k, c) in outs:
        if c != 0:
            continue

        def field(v: str, o: tuple[int, int, int]) -> mpmath.mpf:
            key = (i + o[0], j + o[1], k + o[2])
            return prim[(*key, 0)] if v == 'Df' else un[(*key, toys.POINT_COMPONENTS[v])]
        coords = tuple(mpmath.mpf(float(lo[a] + n * (0.1, 0.2, 0.3)[a])) for a, n in enumerate((i, j, k)))
        vals = toys.values(cc, field, {'trK0': mpmath.mpf(0.7), 'eps': mpmath.mpf(0.2)}, dxs, coords)
        reference[(i, j, k)] = {'v_rhs': (0, vals['v_rhs']), 'ko_u': (1, vals['ko_u'])}
    ok &= compare('point', outs, reference)
    print('gate:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
