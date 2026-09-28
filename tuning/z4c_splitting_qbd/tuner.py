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

#  Tuner definition for the Z4c RHS splitting sweep, loaded by remote_tuner.py.
#  remote_tuner executes this file and uses the Tuner returned by get_tuner().

from EinsteinEngine.tuning.tune_splitting import CombinatorialSplitTuner
from EinsteinEngine.tuning.tuning import Tuner


def get_tuner() -> Tuner:
    return CombinatorialSplitTuner(n_vars=15)
