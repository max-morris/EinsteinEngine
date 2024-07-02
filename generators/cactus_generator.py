from abc import ABC, abstractmethod
from collections import OrderedDict

from dsl.use_indices import ThornDef
from emit.ccl.interface.interface_tree import VariableGroup, Access, DataType, GroupType, InterfaceRoot
from emit.ccl.param.param_tree import ParamRoot
from emit.ccl.schedule.schedule_tree import ScheduleRoot
from emit.code.code_tree import CodeRoot
from emit.tree import Identifier, String
from util import get_or_compute, OrderedSet
from typing import Dict, Set, Optional


class CactusGenerator(ABC):
    thorn_def: ThornDef
    variable_groups: Dict[str, VariableGroup]
    var_names: OrderedSet[str]

    vars_to_ignore: Set[str] = {'x', 'y', 'z'}
    vars_predeclared: Set[str] = {'regrid_error'}

    def __init__(self, thorn_def: ThornDef):
        self.thorn_def = thorn_def
        self.variable_groups = dict()
        self.var_names = OrderedSet()

        for tf in self.thorn_def.thorn_functions.values():
            for iv in tf.eqn_list.inputs:
                var_name = str(iv)
                if var_name not in self.vars_to_ignore:
                    self.var_names.add(var_name)
            for ov in tf.eqn_list.outputs:
                var_name = str(ov)
                if var_name not in self.vars_to_ignore:
                    self.var_names.add(var_name)

        for var_name in [v for v in self.var_names if v not in self.vars_predeclared]:
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
