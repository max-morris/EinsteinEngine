from abc import ABC, abstractmethod

from EmitCactus.dsl.use_indices import ThornDef, ScheduleTarget
from EmitCactus.emit.ccl.interface.interface_tree import VariableGroup, Access, DataType, GroupType, InterfaceRoot, \
    TagPropertyNode, RhsTag, CheckpointTag, GroupTags, ParityTag
from EmitCactus.emit.ccl.param.param_tree import ParamRoot
from EmitCactus.emit.ccl.schedule.schedule_tree import ScheduleRoot, ScheduleBlock
from EmitCactus.emit.code.code_tree import CodeRoot
from EmitCactus.emit.tree import Identifier, String, Bool
from EmitCactus.util import get_or_compute, OrderedSet
from typing import Dict, Set, Optional, TypedDict
from typing_extensions import Unpack
from enum import auto, Enum


class InteriorSyncMode(Enum):
    """
    Determines explicit syncing behavior for variables (grid functions) which are written on the interior.
    """

    HandsOff = auto()
    """
    Never emit SYNC statements. Use this if you are relying on `presync-only`.
    """

    IgnoreRhs = auto()
    """
    Emit SYNC statements for all variables written on the interior, except those which are the RHS of a state variable.
    """

    MixedRhs = auto()
    """
    Emit SYNC statements for all variables written on the interior, except those which are the RHS of a state variable.
    When targeting CarpetX, also produce an `ExplicitSyncBatch` containing all the state variables. 
    """

    Always = auto()
    """
    Emit SYNC statements for all variables written on the interior.
    """


class CactusGeneratorOptions(TypedDict, total=False):
    extra_schedule_blocks: list[ScheduleBlock]
    interior_sync_mode: InteriorSyncMode
    interior_sync_schedule_target: ScheduleTarget


class CactusGenerator(ABC):
    thorn_def: ThornDef
    variable_groups: Dict[str, VariableGroup]
    var_names: OrderedSet[str]
    options: CactusGeneratorOptions

    vars_to_ignore: Set[str] = {'t', 'x', 'y', 'z', 'DXI', 'DYI', 'DZI'}

    def __init__(self, thorn_def: ThornDef, options: CactusGeneratorOptions):
        self.thorn_def = thorn_def
        self.variable_groups = dict()
        self.var_names = OrderedSet()
        self.options = options if options is not None else dict()

        if 'interior_sync_mode' not in self.options:
            self.options['interior_sync_mode'] = InteriorSyncMode.Always

        for tf in self.thorn_def.thorn_functions.values():
            for iv in tf.eqn_complex.inputs:
                var_name = str(iv)
                if var_name not in self.vars_to_ignore:
                    self.var_names.add(var_name)
            for ov in tf.eqn_complex.outputs:
                var_name = str(ov)
                if var_name not in self.vars_to_ignore:
                    self.var_names.add(var_name)

        for var_name in [v for v in self.var_names if self._var_is_locally_declared(v)]:
            group_name = self.thorn_def.var2base.get(var_name, var_name)
            tags: list[TagPropertyNode] = list()

            if (var_rhs := self.thorn_def.rhs.get(group_name)) is not None:
                tags.append(RhsTag(String(f"{self.thorn_def.name}::{var_rhs}")))
            else:
                tags.append(CheckpointTag(Bool(False)))  # CarpetX currently requires this

            if (var_parity := self.thorn_def.base2parity.get(group_name)) is not None:
                tags.append(ParityTag(var_parity))

            get_or_compute(self.variable_groups, group_name, lambda k: VariableGroup(
                access=Access.Public,
                group_name=Identifier(k),
                data_type=DataType.Real,
                variable_names=list(),
                group_type=GroupType.GF,
                centering=self.thorn_def.centering.get(group_name, None),  # type: ignore[arg-type]
                tags=GroupTags(tags)
            )).variable_names.append(Identifier(var_name))

    @abstractmethod
    def get_src_file_name(self, which_fn: str) -> str:
        ...

    @abstractmethod
    def generate_makefile(self) -> str:
        ...

    @abstractmethod
    def generate_schedule_ccl(self) -> ScheduleRoot:
        ...

    @abstractmethod
    def generate_interface_ccl(self) -> InterfaceRoot:
        ...

    @abstractmethod
    def generate_param_ccl(self) -> ParamRoot:
        ...

    @abstractmethod
    def generate_function_code(self, which_fn: str) -> CodeRoot:
        ...

    def _var_is_locally_declared(self, var_name: str) -> bool:
        return self.thorn_def.var2base.get(var_name, var_name) not in self.thorn_def.base2thorn

    def _get_qualified_var_name(self, var_name: str) -> str:
        var_base = self.thorn_def.var2base.get(var_name, var_name)
        from_thorn: Optional[str] = self.thorn_def.base2thorn.get(var_base, None)
        return var_name if from_thorn is None else f'{from_thorn}::{var_name}'

    def _get_qualified_group_name_from_var_name(self, var_name: str) -> str:
        var_base = self.thorn_def.var2base.get(var_name, var_name)
        from_thorn: Optional[str] = self.thorn_def.base2thorn.get(var_base, None)
        group_name = self.thorn_def.base2group.get(var_base, var_name)
        return group_name if from_thorn is None else f'{from_thorn}::{group_name}'
