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
The shared per-loop analysis of the AMReX backend (docs, section 4): classification by declaration, the checks, and
the statement list of one loop of one function, independent of the emission mode.
"""

from dataclasses import dataclass, field
from typing import Optional

import sympy as sy

from EinsteinEngine.common.sympywrap import free_symbols
from EinsteinEngine.emit.code.common.code_tree import Expr
from EinsteinEngine.frontend.dsl.amrex.amrex_frontend import AMReXFunction, AMReXModule, OutputInfo
from EinsteinEngine.frontend.dsl.amrex.layout import FieldBinding
from EinsteinEngine.frontend.dsl.amrex.schemes import KernelLaunchMode
from EinsteinEngine.generators.amrex.guards import check_guards
from EinsteinEngine.generators.generator_exception import GeneratorException
from EinsteinEngine.generators.substitute_recycled_temporaries import substitute_recycled_temporaries
from EinsteinEngine.intermediate.eqnlist import EqnList

STENCIL_FN = 'stencil'
COORDS = ('x', 'y', 'z')
SPACING = ('DXI', 'DYI', 'DZI')


def unprimed(s: sy.Basic | str) -> str:
    return str(s).replace("'", "")


@dataclass
class Statement:
    """One equation of a loop, in emission order: `lhs` (as emitted, after recycling), its original name, `rhs`."""
    lhs: str
    original: str
    rhs: sy.Expr
    output: Optional[OutputInfo]  # set if it writes an output component
    local: bool  # an output that this loop or a later one also reads: computed into a local first, then written
    reassign: bool
    mutable: bool
    comment: Optional[str]
    tile_index: Optional[int]  # set if it computes a tile temporary later loops read


@dataclass
class LoopAnalysis:
    function: str
    loop_idx: int
    name: str  # the device/inline function's name
    comment: str
    field_reads: dict[str, FieldBinding] = field(default_factory=dict)  # every field component read
    output_reads: dict[str, OutputInfo] = field(default_factory=dict)  # rhs components an earlier loop wrote
    tile_loads: dict[str, int] = field(default_factory=dict)
    coords: list[str] = field(default_factory=list)
    spacing: list[str] = field(default_factory=list)
    params: list[str] = field(default_factory=list)
    statements: list[Statement] = field(default_factory=list)
    writes: dict[str, OutputInfo] = field(default_factory=dict)  # output component -> info
    guards: set[str] = field(default_factory=set)  # values this loop sets to a guard (later loops' check)
    not_guards: set[str] = field(default_factory=set)  # values this loop sets to anything else


def analyze_function(module: AMReXModule, fn: AMReXFunction, prior_writes: set[str],
                     guards: Optional[set[str]] = None) -> tuple[list[LoopAnalysis], list[str], set[str]]:
    """The loops of one function, its tile temporaries (scratch-component order) and the output components it writes.
    `guards`: the names (outputs, tile temporaries) that earlier loops set to a guard; updated with this function's."""
    guards = set() if guards is None else guards
    complex_ = fn.eqn_complex
    complex_._calc_tile_temps()
    complex_._calc_vars()
    if len(module.global_temporaries) > 0:
        raise GeneratorException("Global temporaries are not supported by the AMReX backend.")
    tiles = sorted(str(t) for t in complex_.tile_temporaries)
    tile_index = {t: i for i, t in enumerate(tiles)}
    nloops = len(complex_.eqn_lists)
    loops: list[LoopAnalysis] = list()
    writes: set[str] = set()
    prior = set(prior_writes)
    for loop_idx, eqn_list in enumerate(complex_.eqn_lists):
        loops.append(_analyze_loop(module, fn, loop_idx, nloops, eqn_list, prior, tile_index, guards))
        # Only outputs and tile temporaries carry a value to later loops; a name's latest assignment decides.
        carried = set(module.output_bases_components()) | set(tiles)
        guards -= loops[-1].not_guards
        guards |= loops[-1].guards & carried
        prior |= set(loops[-1].writes)
        writes |= set(loops[-1].writes)
    return loops, tiles, writes


def _analyze_loop(module: AMReXModule, fn: AMReXFunction, loop_idx: int, nloops: int, eqn_list: EqnList,
                  prior: set[str], tile_index: dict[str, int], guards: set[str]) -> LoopAnalysis:
    where = fn.name if nloops == 1 else f"{fn.name} loop {loop_idx}"
    name = fn.name if nloops == 1 else f"{fn.name}_{loop_idx}"
    loop = LoopAnalysis(fn.name, loop_idx, name, fn.source_annotations.loops.get(loop_idx) or name)
    subst = substitute_recycled_temporaries(eqn_list)
    original_lhs = [str(lhs) for lhs, _ in eqn_list.sorted_eqns]
    stored_tiles = {str(t) for t in eqn_list.uninitialized_tile_temporaries}
    loaded_tiles = {str(t) for t in eqn_list.preinitialized_tile_temporaries}
    if bad := sorted({str(x.new) for x in subst.substitutions} & loaded_tiles):
        raise GeneratorException(f"{where}: recycling reuses the loaded tile temporaries {bad}.")
    loop.guards, loop.not_guards = check_guards(dict(eqn_list.sorted_eqns), set(eqn_list.temporaries), where, guards)

    # Every left-hand side is classified by declaration: assigning a field or a parameter is refused; an lhs declared
    # as an output is a write of that output, whether or not EE's EqnList counts it as one (it counts an lhs that a
    # later equation of the loop reads as a temporary, and a tile temporary that its writer loop does not read itself
    # among the loop's outputs).
    for raw_lhs in original_lhs:
        w = unprimed(raw_lhs)
        if module.field_binding(w) is not None:
            raise GeneratorException(f"{where}: assigns {w}, which is a field (an input); fields are read-only.")
        if w in module.params:
            raise GeneratorException(f"{where}: assigns {w}, which is a parameter.")
        if w in COORDS or w in SPACING:
            raise GeneratorException(f"{where}: assigns {w}, which the generated code defines.")
    for lhs in eqn_list.outputs:
        if module.output_info(unprimed(lhs)) is None and str(lhs) not in stored_tiles:
            raise GeneratorException(f"{where}: {unprimed(lhs)} is written but is not a declared output.")
    true_outputs = {str(lhs) for lhs in eqn_list.outputs if str(lhs) not in stored_tiles}
    written_raw: set[str] = set()
    for raw_lhs in original_lhs:
        w = unprimed(raw_lhs)
        if (info := module.output_info(w)) is None:
            continue
        if "'" in raw_lhs and not isinstance(module.mode, KernelLaunchMode):
            raise GeneratorException(f"{where}: {raw_lhs} overwrites an output; point mode assigns each output once.")
        if w in loop.writes:
            raise GeneratorException(f"{where}: {w} is written twice in one loop.")
        loop.writes[w] = info
        if raw_lhs in true_outputs:
            written_raw.add(raw_lhs)

    lhs_raw = {str(lhs) for lhs, _ in subst.eqns}
    params = {str(p) for p in eqn_list.params}
    coords: set[str] = set()
    spacing: set[str] = set()
    used_params: set[str] = set()
    for _, rhs in subst.eqns:
        for call in rhs.atoms(sy.Function):
            if getattr(call.func, 'name', None) == STENCIL_FN:
                var = unprimed(call.args[0])
                if (binding := module.field_binding(var)) is None:
                    raise GeneratorException(f"{where}: stencil read of {var}, which is not a field.")
                loop.field_reads[var] = binding
        for s in free_symbols(rhs):
            raw, name_ = str(s), unprimed(s)
            if raw in written_raw:
                raise GeneratorException(f"{where}: reads {s}, which this loop writes as an output.")
            if name_ in SPACING:
                spacing.add(name_)
                continue
            if raw in params:
                used_params.add(raw)
                continue
            if raw in lhs_raw or raw in loaded_tiles:
                continue
            if "'" in raw:
                raise GeneratorException(f"{where}: reads the overwrite {s} before it is written.")
            if name_ in COORDS:
                coords.add(name_)
            elif (binding := module.field_binding(name_)) is not None:
                loop.field_reads[name_] = binding
            elif (info := module.output_info(name_)) is not None:
                if not isinstance(module.mode, KernelLaunchMode):
                    raise GeneratorException(f"{where}: reads the output {name_}; in point mode a function reads only "
                                             f"fields (pass another function's result in as a field).")
                if name_ not in prior:
                    raise GeneratorException(
                        f"{where}: reads {name_}, which no earlier loop writes; the functions are launched in creation "
                        f"order {module.function_order}, each loop in order.")
                loop.output_reads[name_] = info
            else:
                raise GeneratorException(f"{where}: {name_} is neither a field, an output, a parameter, a coordinate, "
                                         f"the grid spacing nor a temporary (unbound symbols are refused).")
    if overlap := sorted(set(loop.output_reads) & lhs_raw):
        raise GeneratorException(f"{where}: {overlap} are both output reads and temporaries.")

    reassigned = {x.eqn_idx for x in subst.substitutions}
    recycled_targets = {unprimed(x.new) for x in subst.substitutions}
    annotations = {str(lhs): ann for lhs, ann in fn.source_annotations.eqns[loop_idx].items()}
    for eqn_idx, (lhs, rhs) in enumerate(subst.eqns):
        lhs_name = unprimed(lhs)
        original = original_lhs[eqn_idx]
        info = module.output_info(unprimed(original))
        local = info is not None and original not in true_outputs
        tile = tile_index[original] if original in stored_tiles else None
        loop.statements.append(Statement(lhs_name, original, rhs, info, local, eqn_idx in reassigned,
                                         lhs_name in recycled_targets, annotations.get(original), tile))
    loop.tile_loads = {t: tile_index[t] for t in sorted(loaded_tiles)}
    loop.coords = sorted(coords)
    loop.spacing = [d for d in SPACING if d in spacing]
    loop.params = sorted(used_params)
    return loop

