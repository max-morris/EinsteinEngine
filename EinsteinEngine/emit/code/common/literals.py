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

"""Floating-point literals that read back as exactly the intended double."""

import math


def real_literal(fl: float, exponent_letter: str = "e") -> str:
    """The shortest decimal that reads back as exactly the double `fl` (Python's round-trip repr, up to 17 significant
    digits), e.g. 0.1, 2.0, 1e-10, 3.141592653589793. `exponent_letter` replaces repr's "e" (Fortran double precision
    uses "D"); with a letter other than "e", a value without an exponent gets one of 0 (2.0D0), so that the literal has
    the requested kind. Raises ValueError for inf and nan, which have no literal."""
    fl = float(fl)
    if not math.isfinite(fl):
        raise ValueError(f"no floating-point literal for the non-finite value {fl}")
    text = repr(fl)
    if exponent_letter == "e":
        return text
    mantissa, _, exponent = text.partition("e")
    return f"{mantissa}{exponent_letter}{int(exponent) if exponent else 0}"
