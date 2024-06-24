from abc import ABC, abstractmethod

from dsl.use_indices import ThornDef
from emit.ccl.interface.interface_tree import VariableGroup, Access, DataType, GroupType, InterfaceRoot
from emit.ccl.param.param_tree import ParamRoot
from emit.ccl.schedule.schedule_tree import ScheduleRoot
from emit.code.code_tree import CodeRoot
from emit.tree import Identifier
from util import get_or_compute


class CactusGenerator(ABC):
    thorn_def: ThornDef
    variable_groups: dict[str, VariableGroup]
    var_names: set[str] = set()

    def __init__(self, thorn_def: ThornDef):
        self.thorn_def = thorn_def
        self.variable_groups = dict()
        self.var_names = set()

        for tf in self.thorn_def.thorn_functions.values():
            for iv in tf.eqnlist.inputs:
                self.var_names.add(str(iv))
            for ov in tf.eqnlist.outputs:
                self.var_names.add(str(ov))

        for var_name in self.var_names:
            group_name = self.thorn_def.var2base.get(var_name, 'scalar_gfs')

            get_or_compute(self.variable_groups, group_name, lambda k: VariableGroup(
                access=Access.Public,
                group_name=Identifier(k),
                data_type=DataType.Real,
                variable_names=list(),
                group_type=GroupType.GF
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
