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

"""Z4c RHS splitting at the early locus, with the add_eqn groups in the order prioritize_rare_symbols gives them.

The early locus cuts between the author-level add_eqn calls of z4c_rhs, in an order of those calls. This tuner has the
engine derive that order with ``rank_by_post_population(prioritize_rare_symbols)``: at the start of every bake, a
trial bake of the whole thorn (unsplit, with prioritize_rare_symbols ordering z4c_rhs) finds where each call's
equations land after CSE, and the calls are ranked by the median of those positions (see
``EinsteinEngine.intermediate.eqn_ordering.rank_by_post_population``). For today's recipe the derived order is
``Z4C_STEVE_ORDER`` below, which is what the feature/eqn-order-instrumentation branch derived offline (its
tuning/z4c_splitting_rostam/ordertest/derive_order.py) and hardcoded. With that branch's split checkpoint
(tuning/z4c_splitting_rostam/split_tuning_fixed_order.jsonl), its keys shifted to 0-based by
scripts/shift_checkpoint_indices.py, generate_best generates the same code as that branch's tuned build (see
generate-best.sh; checked byte for byte).

The derived rule follows the recipe: if z4c_rhs or the rest of the thorn changes, so does the order, and the probe
file next to the checkpoint (``<checkpoint>.probe.json``) records the order each probe derived, so that resuming a
checkpoint recorded under a different order is refused. The price is the trial bake, which adds about 17 s to every
generation of the recipe (every tuning trial, the probe, and generate_best's run).

``get_explicit_order_tuner()`` is the alternative: it pins ``Z4C_STEVE_ORDER`` with ``add_eqn_order``, which skips the
trial bake and does not follow recipe changes. Both generate the same code today, but they describe their
``early_ordering_fn`` differently, so a checkpoint's probe file recorded with one does not match the other; to switch
a checkpoint that already has entries, start a new checkpoint file (or delete the probe file, losing the check).

The number of split positions is found by probing the recipe, which must read auto_split_locus and early_ordering_fn
with get_optional_tuning_param.
"""

from EinsteinEngine.intermediate.eqn_ordering import add_eqn_order, prioritize_rare_symbols, rank_by_post_population
from EinsteinEngine.tuning.tune_splitting import CombinatorialSplitTuner
from EinsteinEngine.tuning.tuning import Tuner

# The z4c_rhs add_eqn calls (0-based) in the order rank_by_post_population(prioritize_rare_symbols) derives for today's
# recipe, as derived offline on the feature/eqn-order-instrumentation branch. Kept for get_explicit_order_tuner, and as
# a record of the order the checkpoint above was tuned under.
Z4C_STEVE_ORDER = [8, 1, 0, 2, 3, 5, 10, 7, 4, 9, 12, 6, 11]


def get_tuner() -> Tuner:
    return CombinatorialSplitTuner(early_ordering_fn=rank_by_post_population(prioritize_rare_symbols))


def get_explicit_order_tuner() -> Tuner:
    """The same tuner with the order pinned to Z4C_STEVE_ORDER (no trial bake). To use it, make get_tuner return it."""
    return CombinatorialSplitTuner(early_ordering_fn=add_eqn_order(Z4C_STEVE_ORDER))
