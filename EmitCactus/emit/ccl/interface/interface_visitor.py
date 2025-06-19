from multimethod import multimethod

# noinspection PyUnresolvedReferences
# MyPy needs these
from typing import Any

from EmitCactus.emit.ccl.interface.interface_tree import InterfaceNode, InterfaceRoot, HeaderSection, IncludeSection, \
    UsesInclude, IncludeIn, FunctionSection, FunctionAlias, FunctionAliasArg, FunctionAliasFpArg, RequiresFunction, \
    UsesFunction, ProvidesFunction, VariableSection, VariableGroup, GroupTags, TagPropertyNode, ParityTag, TensorParity, \
    SingleIndexParity
from EmitCactus.emit.tree import Identifier, Integer, Verbatim, String, Bool
from EmitCactus.emit.visitor import Visitor, visit_each


class InterfaceVisitor(Visitor[InterfaceNode]):

    @multimethod
    def visit(self, n: InterfaceNode) -> Any:
        self.not_implemented(n)

    @visit.register
    def _(self, n: Identifier) -> str:
        return n.identifier

    @visit.register
    def _(self, n: Integer) -> str:
        return f'{n.integer}'

    @visit.register
    def _(self, n: Verbatim) -> str:
        return n.text

    @visit.register
    def _(self, n: String) -> str:
        return f'"{n.text}"' if not n.single_quotes else f"'{n.text}'"

    @visit.register
    def _(self, n: Bool) -> str:
        return 'yes' if n.b else 'no'

    @visit.register
    def _(self, n: InterfaceRoot) -> str:
        return f'{self.visit(n.header_section)}\n{self.visit(n.include_section)}\n{self.visit(n.function_section)}\n{self.visit(n.variable_section)}'

    @visit.register
    def _(self, n: HeaderSection) -> str:
        s = f'implements: {self.visit(n.implements)}'

        if len(n.inherits) != 0:
            sorted_inherits = sorted(n.inherits, key=lambda x: repr(x))
            s += f'\ninherits: {",".join(visit_each(self, sorted_inherits))}'

        s += '\nUSES INCLUDE HEADER: timer.hxx'

        if len(n.friends) != 0:
            s += f'\nfriend: {",".join(visit_each(self, n.friends))}'

        return s

    @visit.register
    def _(self, n: IncludeSection) -> str:
        return '\n'.join(visit_each(self, n.directives))

    @visit.register
    def _(self, n: UsesInclude) -> str:
        return f'USES INCLUDE {n.typ.representation}: {self.visit(n.file_name)}'

    @visit.register
    def _(self, n: IncludeIn) -> str:
        return f'INCLUDES {n.typ.representation}: {self.visit(n.file_to_include)} in {self.visit(n.file_name)}'

    @visit.register
    def _(self, n: FunctionSection) -> str:
        return '\n'.join(visit_each(self, n.declarations))

    @visit.register
    def _(self, n: FunctionAlias) -> str:
        return f'{n.return_type.representation} FUNCTION {self.visit(n.alias)}({", ".join(visit_each(self, n.args))})'

    @visit.register
    def _(self, n: FunctionAliasArg) -> str:
        return f'{n.arg_type.representation} {n.arg_intent.representation}{" ARRAY " if n.is_array else " "}{self.visit(n.arg_name)}'

    @visit.register
    def _(self, n: FunctionAliasFpArg) -> str:
        return f'{n.fp_return_type.representation} {n.fp_intent.representation}{" ARRAY " if n.is_array else " "}CCTK_FPOINTER {self.visit(n.fp_name)}({", ".join(visit_each(self, n.fp_args))})'

    @visit.register
    def _(self, n: RequiresFunction) -> str:
        return f'REQUIRES FUNCTION {self.visit(n.alias)}'

    @visit.register
    def _(self, n: UsesFunction) -> str:
        return f'USES FUNCTION {self.visit(n.alias)}'

    @visit.register
    def _(self, n: ProvidesFunction) -> str:
        return f'PROVIDES FUNCTION {self.visit(n.alias)} WITH {self.visit(n.provider)} LANGUAGE {n.language.representation}'

    @visit.register
    def _(self, n: VariableSection) -> str:
        return '\n'.join(visit_each(self, n.variable_groups))

    @visit.register
    def _(self, n: VariableGroup) -> str:
        s = f'{n.access.representation}:\n{n.data_type.representation} {self.visit(n.group_name)}'

        if n.vector_size is not None:
            s += f'[{self.visit(n.vector_size)}]'

        if n.group_type is not None:
            s += f' TYPE={n.group_type.representation}'

        if n.dim is not None:
            s += f' DIM={self.visit(n.dim)}'

        if n.time_levels is not None:
            s += f' TIMELEVELS={self.visit(n.time_levels)}'

        if n.array_size is not None:
            s += f' SIZE={",".join(visit_each(self, n.array_size))}'

        if n.array_distrib is not None:
            s += f' DISTRIB={n.array_distrib.representation}'

        if n.array_ghost_size is not None:
            s += f' GHOSTSIZE={self.visit(n.array_ghost_size)}'

        if n.stagger_spec is not None:
            s += f' STAGGER={self.visit(n.stagger_spec)}'

        if n.tags is not None:
            s += f' TAGS={self.visit(n.tags)}'

        if n.centering is not None:
            s += f' CENTERING={{ {n.centering.string_repr} }}'

        s += '\n{\n    '

        s += ', '.join(visit_each(self, n.variable_names))

        s += '\n}'

        if n.group_description is not None:
            s += f' {self.visit(n.group_description)}'

        return s

    @visit.register
    def _(self, n: GroupTags) -> str:
        tags = ' '.join(visit_each(self, n.tags))
        return f"'{tags}'"

    @visit.register
    def _(self, n: TagPropertyNode) -> str:
        return f'{self.visit(n.get_key())}={self.visit(n.get_value())}'

    @visit.register
    def _(self, n: TensorParity) -> str:
        parities = '  '.join(visit_each(self, n.parities))
        return f'{{{parities}}}'

    @visit.register
    def _(self, n: SingleIndexParity) -> str:
        reps = [p.representation for p in [n.x_parity, n.y_parity, n.z_parity]]
        return ' '.join(reps)
