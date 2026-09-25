#  Copyright (C) 2026 Steven R. Brandt and other Einstein Engine contributors.
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

#  Tuner definition for the Z4c RHS splitting sweep on rostam, loaded by
#  remote_tuner.py. remote_tuner executes this file and uses the Tuner
#  returned by get_tuner().

from EinsteinEngine.tuning.tune_splitting import CombinatorialSplitTuner
from EinsteinEngine.tuning.tuning import Tuner

#  The split predicates are queried once per add_eqn() call on fun_z4c_rhs,
#  with the running count 1..N. recipes/Cottonmouth/Z4c.py makes exactly 13
#  such calls, so 13 is the number of split points that actually exist.
#  (The z4c_splitting_qbd sample says 15, which just wastes four search
#  dimensions on predicate indices that are never queried.)
N_SPLIT_POINTS = 13


def get_tuner() -> Tuner:
    return CombinatorialSplitTuner(n_vars=N_SPLIT_POINTS)
