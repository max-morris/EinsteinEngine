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
from EinsteinEngine.frontend.dsl.indices import *
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
from EinsteinEngine.wizards.vanilla_f90_wizard import VanillaF90Wizard
from EinsteinEngine.frontend.dsl.dsl_function_frontend import DslFunctionFrontend
from EinsteinEngine.frontend.dsl.f90.vanilla_f90_frontend import VanillaF90Module
from EinsteinEngine.generators.vanilla_f90_generator import VanillaF90Generator
from EinsteinEngine.frontend.dsl.finite_difference import DX, DY, DZ, DT

from sympy.core.relational import Relational


__all__ = [
    "Identifier", "String", "Centering",
    "CppCarpetXGenerator", "SyncMode",
    "cbrt", "sqrt", "mk_matrix", "mk_piecewise", "log", "Relational",
    "GroupOrFunction", "ScheduleBlock", "AtOrIn",
    "CppCarpetXWizard", "ExplicitSyncBatch", "VanillaF90Wizard", "VanillaF90Generator", "VanillaF90Module", "DslFunctionFrontend",
    "parities",
    "ScheduleBin", "sympify",
    "sin", "cos", "tan", "cot", "sec", "csc",
    "sinh", "cosh", "tanh", "sech", "csch", "coth",
    "erf", "atan", "pi",
    "D", "div", "idx_to_int", "IndexedSubstFnType", "MkSubstType", "CactusParam", "ThornFunction", "ScheduleBin", "ThornDef",
    "subst_tensor_xyz",
    "noop", "stencil", "DD", "DDI", "DX", "DY", "DZ", "DT",
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
    "IntentRegion", "IntentOverride", "bayesian_optimization", "kreiss_oliger_stencil", "finite_difference_stencil", "pull_out"]
