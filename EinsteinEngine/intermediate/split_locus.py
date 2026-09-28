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

from enum import Enum


class SplitLocus(Enum):
    """
    The point in the bake at which a function's auto split predicates are evaluated.

    Each locus has a matching ordering function, and the predicates are called with 0-based positions
    in the order that exists at that point; position ``i`` means "split after the i-th element".
    "Population" means global CSE populating the equation lists with temporaries (not the paper's populating of the
    abstract program).

    Early:          Elements are author-level ``add_eqn`` calls (all components of a tensor equation stay together),
                    ordered by the ``early_ordering_fn`` bake option, or in pre-population order (which is still the
                    recipe order) when it is None. This is the historical behavior.
    PrePopulation:  Elements are scalar equations before CSE temporaries are populated, in the order of the pre-CSE
                    bake: by the ``pre_population_ordering_fn`` bake option, or by ``ordering_fn`` when it is None.
    PostPopulation: Elements are scalar equations after global CSE, ordered by the ``ordering_fn`` bake option.
    """
    Early = 0
    PrePopulation = 1
    PostPopulation = 2

    def __repr__(self) -> str:
        return self.name
