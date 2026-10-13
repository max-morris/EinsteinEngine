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

from typing import Optional

# The fractional powers the CarpetX backend emits as a named call instead of pow(), keyed
# by the exponent's numerator and denominator in lowest terms. The emitter and the CarpetX
# cost model both read this table, so they cannot disagree about which powers are cheap.
_NAMED_ROOTS: dict[tuple[float, float], str] = {
    (1, 2): 'sqrt',
    (1, 3): 'cbrt',
}


def named_root(numerator: float, denominator: float) -> Optional[str]:
    """The function CarpetX emits for x**(numerator/denominator), or None if it emits pow()."""
    return _NAMED_ROOTS.get((numerator, denominator))
