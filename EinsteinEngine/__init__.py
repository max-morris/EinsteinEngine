#!/usr/bin/env python3

#  Copyright (C) 2024-2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
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

# Export dynamic derivative functions
from EinsteinEngine.frontend.dsl.cactus.carpetx import ExplicitSyncBatch, NewRadXBoundaryBatch
from EinsteinEngine.intermediate.eqn_ordering import EqnOrderingFn, maximize_symbol_reuse, prioritize_rare_symbols, bayesian_optimization, lexicographical_order, insertion_order
from EinsteinEngine.frontend.definitions import *
from EinsteinEngine.common.intent_override import IntentOverride
from EinsteinEngine.intermediate.soft_split_retainment_predicate import *
from EinsteinEngine.common.sympywrap import cbrt, sqrt, mk_matrix, log, cos, sin, tan, cot, sec, csc, cosh, sinh, tanh, sech, csch, coth, \
    erf, pi, atan
from EinsteinEngine.intermediate.temp_kind import TempKind
from EinsteinEngine.intermediate.temporary_promotion_predicate import *
from EinsteinEngine.frontend.dsl.use_indices import D, div, idx_to_int, IndexedSubstFnType, MkSubstType, subst_tensor_xyz, \
    noop, stencil, DD, DDI
from EinsteinEngine.common.cse_optimization_level import CseOptimizationLevel
from EinsteinEngine.emit.ccl.schedule.schedule_tree import GroupOrFunction, ScheduleBlock, AtOrIn, IntentRegion
from EinsteinEngine.emit.tree import Identifier, String, Centering
from EinsteinEngine.common.sympywrap import Applier, sqrt, cbrt, log, exp, Pow, PowType, UFunc, diff, \
    inv, det, sympify, simplify, cse, mk_idx, mk_symbol, \
    mk_matrix, do_subs, mk_function, mk_eq, do_replace, mk_indexed_base, mk_zeros, \
    free_indexed, mk_indexed, mk_wild, mk_idxes, free_symbols, h_step, mk_piecewise
from EinsteinEngine.frontend.dsl.cactus.cactus_param import CactusParam
from EinsteinEngine.frontend.dsl.cactus.cactus_frontend import ScheduleBin, ThornFunctionBakeOptions, ThornFunction, ThornDef, \
    parities
from EinsteinEngine.generators.cactus_generator import SyncMode
from EinsteinEngine.generators.cpp_carpetx_generator import CppCarpetXGenerator
from EinsteinEngine.wizards.thorn_wizards import CppCarpetXWizard
from sympy import Idx
from sympy.core.relational import Relational

ui: Idx
li: Idx
uj: Idx
lj: Idx
uk: Idx
lk: Idx
ua: Idx
la: Idx
ub: Idx
lb: Idx
uc: Idx
lc: Idx
ud: Idx
ld: Idx
u0: Idx
l0: Idx
u1: Idx
l1: Idx
u2: Idx
l2: Idx
u3: Idx
l3: Idx
u4: Idx
l4: Idx
u5: Idx
l5: Idx

def _populate_index_globals() -> None:
    global ui, li, uj, lj, uk, lk, ua, la, ub, lb, uc, lc, ud, ld
    global u0, l0, u1, l1, u2, l2, u3, l3, u4, l4, u5, l5
    ui, li = mk_idxes("ui li")
    uj, lj = mk_idxes("uj lj")
    uk, lk = mk_idxes("uk lk")
    ua, la = mk_idxes("ua la")
    ub, lb = mk_idxes("ub lb")
    uc, lc = mk_idxes("uc lc")
    ud, ld = mk_idxes("ud ld")
    u0, l0 = mk_idxes("u0 l0")
    u1, l1 = mk_idxes("u1 l1")
    u2, l2 = mk_idxes("u2 l2")
    u3, l3 = mk_idxes("u3 l3")
    u4, l4 = mk_idxes("u4 l4")
    u5, l5 = mk_idxes("u5 l5")


_populate_index_globals()


__all__ = [
    "Identifier", "String", "Centering",
    "CppCarpetXGenerator", "SyncMode",
    "cbrt", "sqrt", "mk_matrix", "mk_piecewise", "log", "Relational",
    "GroupOrFunction", "ScheduleBlock", "AtOrIn",
    "CppCarpetXWizard", "ExplicitSyncBatch",
    "parities",
    "ScheduleBin", "sympify",
    "sin", "cos", "tan", "cot", "sec", "csc",
    "sinh", "cosh", "tanh", "sech", "csch", "coth",
    "erf", "atan", "pi",
    "D", "div", "idx_to_int", "IndexedSubstFnType", "MkSubstType", "CactusParam", "ThornFunction", "ScheduleBin", "ThornDef",
    "subst_tensor_xyz",
    "noop", "stencil", "DD", "DDI",
    "ui", "uj", "uk", "ua", "ub", "uc", "ud", "u0", "u1", "u2", "u3", "u4", "u5",
    "li", "lj", "lk", "la", "lb", "lc", "ld", "l0", "l1", "l2", "l3", "l4", "l5",
    "Applier", "sqrt", "cbrt", "log", "exp", "Pow", "PowType", "UFunc", "diff",
    "inv", "det", "sympify", "simplify", "cse", "mk_idx", "mk_symbol",
    "mk_matrix", "do_subs", "mk_function", "mk_eq", "do_replace", "mk_indexed_base", "mk_zeros",
    "free_indexed", "mk_indexed", "mk_wild", "mk_idxes", "free_symbols", "h_step", "ThornFunctionBakeOptions",
    "promote_all", "promote_none", "promote_rank", "promote_percentile", "promote_threshold", "CseOptimizationLevel",
    "retain_percentile", "retain_rank", "retain_threshold", "retain_all", "retain_none",
    "NewRadXBoundaryBatch", "TempKind",
    "EqnOrderingFn", "maximize_symbol_reuse", "prioritize_rare_symbols", "lexicographical_order", "insertion_order", "promote_all", "promote_none", "promote_rank",
    "promote_percentile", "promote_threshold", "CseOptimizationLevel", "NewRadXBoundaryBatch", "TempKind",
    "IntentRegion", "IntentOverride", "bayesian_optimization", "kreiss_oliger_stencil"]
