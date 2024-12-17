from abc import ABC, abstractmethod

from EmitCactus.dsl.use_indices import ThornDef
from EmitCactus.emit.ccl.interface.interface_tree import VariableGroup, Access, DataType, GroupType, InterfaceRoot
from EmitCactus.emit.ccl.param.param_tree import ParamRoot
from EmitCactus.emit.ccl.schedule.schedule_tree import ScheduleRoot, ScheduleBlock
from EmitCactus.emit.code.code_tree import CodeRoot
from EmitCactus.emit.tree import Identifier, String
from EmitCactus.util import get_or_compute, OrderedSet
from typing import Dict, Set, Optional, TypedDict
from typing_extensions import Unpack


class CactusGeneratorOptions(TypedDict, total=False):
    extra_schedule_blocks: list[ScheduleBlock]

class CactusGenerator(ABC):
    thorn_def: ThornDef
    variable_groups: Dict[str, VariableGroup]
    var_names: OrderedSet[str]
    options: CactusGeneratorOptions

    vars_to_ignore: Set[str] = {'t', 'x', 'y', 'z', 'DXI', 'DYI', 'DZI'}

    def __init__(self, thorn_def: ThornDef, **options: Unpack[CactusGeneratorOptions]):
        self.thorn_def = thorn_def
        self.variable_groups = dict()
        self.var_names = OrderedSet()
        self.options = options if options is not None else dict()

        for tf in self.thorn_def.thorn_functions.values():
            for iv in tf.eqn_list.inputs:
                var_name = str(iv)
                if var_name not in self.vars_to_ignore:
                    self.var_names.add(var_name)
            for ov in tf.eqn_list.outputs:
                var_name = str(ov)
                if var_name not in self.vars_to_ignore:
                    self.var_names.add(var_name)

        for var_name in [v for v in self.var_names if self._var_is_locally_declared(v)]:
            group_name = self.thorn_def.var2base.get(var_name, var_name)

            tags: Optional[String] = None
            if (var_rhs := self.thorn_def.rhs.get(group_name)) is not None:
                tags = String(f'rhs="{self.thorn_def.name}::{var_rhs}"', single_quotes=True)
            else:
                # TODO: long-term we don't want this, CarpetX currently needs it
                tags = String(f'checkpoint="no"', single_quotes=True)

            get_or_compute(self.variable_groups, group_name, lambda k: VariableGroup(
                access=Access.Public,
                group_name=Identifier(k),
                data_type=DataType.Real,
                variable_names=list(),
                group_type=GroupType.GF,
                centering=self.thorn_def.centering.get(group_name, None),  # type: ignore[arg-type]
                tags=tags
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
