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

from .emit.tree import Identifier, String, Centering
from .generators.cpp_carpetx_generator import CppCarpetXGenerator
from .dsl.carpetx import ExplicitSyncBatch, NewRadXBoundaryBatch
from .generators.cactus_generator import SyncMode
from .dsl.sympywrap import cbrt, sqrt, mkMatrix, log, cos, sin, tan, cot, sec, csc, cosh, sinh, tanh, sech, csch, coth, erf, pi, atan
from .emit.ccl.schedule.schedule_tree import GroupOrFunction, ScheduleBlock, AtOrIn, IntentRegion
from .generators.wizards import CppCarpetXWizard
from .dsl.use_indices import parities, ThornFunctionBakeOptions, CseOptimizationLevel
from .dsl.temporary_promotion_predicate import *
from .dsl.temp_kind import TempKind
from .dsl.intent_override import IntentOverride

from .dsl.use_indices import D, div, to_num, IndexedSubstFnType, MkSubstType, Param, ThornFunction, ScheduleBin, ThornDef, \
       set_dimension, get_dimension, lookup_pair, subst_tensor_xyz, mk_pair, \
       noop,stencil,DD,DDI, \
       ui, uj, uk, ua, ub, uc, ud, u0, u1, u2, u3, u4, u5, \
       li, lj, lk, la, lb, lc, ld, l0, l1, l2, l3, l4, l5
from .dsl.functions import *

# Export dynamic derivative functions
import EinsteinEngine.dsl.functions as functions
_div_names = ["divx", "divy", "divz", "divxx", "divxy", "divxz", "divyy", "divyz", "divzz"]
for _name in _div_names:
    if hasattr(functions, _name):
        globals()[_name] = getattr(functions, _name)

from .dsl.sympywrap import Applier,sqrt,cbrt,log,exp,Pow,PowType,UFunc,diff,\
    inv,det,sympify,simplify,cse,mkIdx,mkSymbol,\
    mkMatrix,do_subs,mkFunction,mkEq,do_replace,mkIndexedBase,mkZeros,\
    free_indexed,mkIndexed,mkWild,mkIdxs,free_symbols,h_step,mkPiecewise
from sympy import Expr, Idx, Matrix, Indexed, Symbol
from sympy.core.relational import Relational

__all__ = [
    "Identifier", "String", "Centering",
    "CppCarpetXGenerator", "SyncMode",
    "cbrt", "sqrt", "mkMatrix", "mkPiecewise", "log", "Relational",
    "GroupOrFunction", "ScheduleBlock", "AtOrIn",
    "CppCarpetXWizard", "ExplicitSyncBatch",
    "parities",
    "ScheduleBin", "sympify",
    "sin", "cos", "tan", "cot", "sec", "csc",
    "sinh", "cosh", "tanh", "sech", "csch", "coth",
    "erf", "atan", "pi",
    "D", "div", "to_num", "IndexedSubstFnType", "MkSubstType", "Param", "ThornFunction", "ScheduleBin", "ThornDef",
    "set_dimension", "get_dimension", "lookup_pair", "subst_tensor_xyz", "mk_pair",
    "noop","stencil","DD","DDI",
    "ui", "uj", "uk", "ua", "ub", "uc", "ud", "u0", "u1", "u2", "u3", "u4", "u5",
    "li", "lj", "lk", "la", "lb", "lc", "ld", "l0", "l1", "l2", "l3", "l4", "l5",
    "divx", "divy", "divz", "divxx", "divxy", "divxz", "divyy", "divyz", "divzz",
    "Applier","sqrt","cbrt","log","exp","Pow","PowType","UFunc","diff",
    "inv","det","sympify","simplify","cse","mkIdx","mkSymbol",
    "mkMatrix","do_subs","mkFunction","mkEq","do_replace","mkIndexedBase","mkZeros",
    "free_indexed","mkIndexed","mkWild","mkIdxs","free_symbols", "h_step", "ThornFunctionBakeOptions",
    "promote_all", "promote_none", "promote_rank", "promote_percentile", "promote_threshold", "CseOptimizationLevel", "NewRadXBoundaryBatch", "TempKind", "IntentRegion", "IntentOverride"]
