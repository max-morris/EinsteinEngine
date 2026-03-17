from typing import Union

from EmitCactus.dsl.stencil_idx import StencilIdxWithCentering, StencilIdxWithNameAndCentering

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
