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
    The point in the bake at which a function's auto split predicates are evaluated. Position i (0-based) means "split
    after the i-th element" of the order at that point. Population is global CSE adding its temporaries to the lists.

    Early:          Elements are add_eqn calls, ordered by `early_ordering_fn`, else in pre-population order.
    PrePopulation:  Elements are scalar equations in the pre-CSE baked order (by `pre_population_ordering_fn`, else
                    by `ordering_fn`).
    PostPopulation: Elements are scalar equations in the post-CSE baked order (by `ordering_fn`, or without CSE, the
                    pre-CSE one). With manual soft splits that rebake is a fast one, so a Bayesian `ordering_fn` is
                    replaced by its `pre_cse_stand_in`.
    """
    Early = 0
    PrePopulation = 1
    PostPopulation = 2

    def __repr__(self) -> str:
        return self.name
