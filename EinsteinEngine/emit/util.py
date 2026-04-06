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

from typing import Union

from EinsteinEngine.dsl.stencil_idx import StencilIdxWithCentering, StencilIdxWithNameAndCentering

StencilIdxWithCenteringLike = Union[StencilIdxWithCentering, StencilIdxWithNameAndCentering]


def encode_stencil_idx(stencil_idx: StencilIdxWithCenteringLike) -> str:
    encoded = 'stencil_idx'

    for idx in stencil_idx.indices:
        if idx >= 0:
            encoded += f'_{idx}'
        else:
            encoded += f'_m{-idx}'

    encoded += f'_{stencil_idx.centering.string_repr}'

    return encoded
