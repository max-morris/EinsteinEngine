from typing import NamedTuple

from EmitCactus import Centering

class StencilIdx(NamedTuple):
    x: int
    y: int
    z: int


class StencilIdxWithName(NamedTuple):
    indices: StencilIdx
    var_name: str


class StencilIdxWithCentering(NamedTuple):
    indices: StencilIdx
    centering: Centering


class StencilIdxWithNameAndCentering(NamedTuple):
    indices: StencilIdx
    var_name: str
    centering: Centering

    @staticmethod
    def from_stencil_idx(idx: StencilIdxWithName, centering: Centering) -> 'StencilIdxWithNameAndCentering':
        return StencilIdxWithNameAndCentering(idx.indices, idx.var_name, centering)

