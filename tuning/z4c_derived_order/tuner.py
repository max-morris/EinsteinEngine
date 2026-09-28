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

"""Z4c RHS splitting at the early locus, with the add_eqn groups in a fixed derived order.

The order lists the 0-based indices of the Z4c RHS add_eqn calls, ranked by their positions in the post-CSE order (as
derived on the feature/eqn-order-instrumentation branch). With that branch's split checkpoint
(tuning/z4c_splitting_rostam/split_tuning_fixed_order.jsonl), its keys shifted to 0-based by
scripts/shift_checkpoint_indices.py, generate_best is meant to generate the same code as that branch's tuned build
(see generate-best.sh). The number of split positions is found by probing the recipe, which must read
auto_split_locus and early_ordering_fn with get_optional_tuning_param.
"""

from EinsteinEngine.intermediate.eqn_ordering import add_eqn_order
from EinsteinEngine.tuning.tune_splitting import CombinatorialSplitTuner
from EinsteinEngine.tuning.tuning import Tuner

Z4C_DERIVED_ORDER = [8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11]


def get_tuner() -> Tuner:
    return CombinatorialSplitTuner(early_ordering_fn=add_eqn_order(Z4C_DERIVED_ORDER))
