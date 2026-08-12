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

import gc

from EinsteinEngine.intermediate import eqn_ordering
from sympy.core.cache import clear_cache as clear_sympy_cache

def clear_caches() -> None:
    """Clear unbounded symbolic caches."""
    for fn_name in (
        "_dummy_stencil_symbol",
        "_expr_with_stencil_dummies",
        "_symbol_frequency",
        "_free_symbols",
        "_free_symbols_with_dummies",
    ):
        fn = getattr(eqn_ordering, fn_name, None)
        if fn is not None and hasattr(fn, "cache_clear"):
            fn.cache_clear()
    clear_sympy_cache()  # type: ignore[no-untyped-call]
    gc.collect()
