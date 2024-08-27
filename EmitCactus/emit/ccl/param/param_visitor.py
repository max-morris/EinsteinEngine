from multimethod import multimethod

from typing import Any, cast

from EmitCactus.emit.ccl.param.param_tree import ParamNode, ParamRoot, Param, ParamAccess, IntParamDescWildcard, \
    IntParamDescSingle, IntParamClosedLowerBound, IntParamOpenLowerBound, IntParamClosedUpperBound, \
    IntParamOpenUpperBound, IntParamDescRange, IntParamDescRangeWithStep, IntParamRange, RealParamDescWildcard, \
    RealParamDescSingle, RealParamClosedLowerBound, RealParamOpenLowerBound, RealParamClosedUpperBound, \
    RealParamOpenUpperBound, RealParamDescRange, RealParamDescRangeWithStep, RealParamRange, KeywordParamRange, \
    StringParamRange
from EmitCactus.emit.tree import Identifier, Integer, Verbatim, String, Bool, Float
from EmitCactus.emit.visitor import Visitor, VisitorException, visit_each


class ParamVisitor(Visitor[ParamNode]):

    @multimethod
    def visit(self, n: ParamNode) -> Any:
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
        return "true" if n.b else "false"

    @visit.register
    def _(self, n: Float) -> str:
        return f'{n.fl}'

    @visit.register
    def _(self, n: ParamRoot) -> str:
        return '\n'.join(visit_each(self, n.params))

    @visit.register
    def _(self, n: Param) -> str:
        s: str

        if n.param_access is ParamAccess.Shares:
            if n.shares_with is None:
                raise VisitorException('param_access of ParamAccess.Shares requires shares_with to be set')
            s = f'{n.param_access.representation}:{self.visit(n.shares_with)}\n'
        else:
            s = f'{n.param_access.representation}:\n'

        s += f'{n.extends_uses.representation} ' if n.extends_uses is not None else ''

        s += f'{n.param_type.representation} {self.visit(n.param_name)}'

        if n.arr_len is not None:
            s += f'[{self.visit(n.arr_len)}] '
        else:
            s += ' '

        s += self.visit(n.param_desc) + ' '

        if n.alias_name is not None:
            s += f'AS {self.visit(n.alias_name)} '

        if n.steerability is not None:
            s += f'STEERABLE={n.steerability.representation} '

        if n.accumulator is not None:
            s += f'ACCUMULATOR={self.visit(n.accumulator)} '

        if n.accumulator_base is not None:
            s += f'ACCUMULATOR-BASE={self.visit(n.accumulator_base)} '

        s += '\n{\n'

        s += '\n'.join(visit_each(self, n.range_descriptions))

        s += '\n} ' + self.visit(n.default_value)

        return s

    @visit.register
    def _(self, _: IntParamDescWildcard) -> str:
        return '*'

    @visit.register
    def _(self, n: IntParamDescSingle) -> str:
        return cast(str, self.visit(n.integer))

    @visit.register
    def _(self, n: IntParamClosedLowerBound) -> str:
        return f'[{self.visit(n.integer)}'

    @visit.register
    def _(self, n: IntParamOpenLowerBound) -> str:
        return f'({self.visit(n.integer)}'

    @visit.register
    def _(self, n: IntParamClosedUpperBound) -> str:
        return f'{self.visit(n.integer)}]'

    @visit.register
    def _(self, n: IntParamOpenUpperBound) -> str:
        return f'{self.visit(n.integer)})'

    @visit.register
    def _(self, n: IntParamDescRange) -> str:
        return f'{self.visit(n.lower_bound)}:{self.visit(n.upper_bound)}'

    @visit.register
    def _(self, n: IntParamDescRangeWithStep) -> str:
        return f'{self.visit(n.lower_bound)}:{self.visit(n.upper_bound)}:{self.visit(n.step)}'

    @visit.register
    def _(self, n: IntParamRange) -> str:
        return f'{self.visit(n.range_desc)} :: {self.visit(n.comment)}'

    @visit.register
    def _(self, _: RealParamDescWildcard) -> str:
        return '*'

    @visit.register
    def _(self, n: RealParamDescSingle) -> str:
        return cast(str, self.visit(n.real))

    @visit.register
    def _(self, n: RealParamClosedLowerBound) -> str:
        return f'[{self.visit(n.real)}'

    @visit.register
    def _(self, n: RealParamOpenLowerBound) -> str:
        return f'({self.visit(n.real)}'

    @visit.register
    def _(self, n: RealParamClosedUpperBound) -> str:
        return f'{self.visit(n.real)}]'

    @visit.register
    def _(self, n: RealParamOpenUpperBound) -> str:
        return f'{self.visit(n.real)})'

    @visit.register
    def _(self, n: RealParamDescRange) -> str:
        return f'{self.visit(n.lower_bound)}:{self.visit(n.upper_bound)}'

    @visit.register
    def _(self, n: RealParamDescRangeWithStep) -> str:
        return f'{self.visit(n.lower_bound)}:{self.visit(n.upper_bound)}:{self.visit(n.step)}'

    @visit.register
    def _(self, n: RealParamRange) -> str:
        return f'{self.visit(n.range_desc)} :: {self.visit(n.comment)}'

    @visit.register
    def _(self, n: KeywordParamRange) -> str:
        return f'{", ".join(visit_each(self, n.values))} :: {self.visit(n.comment)}'

    @visit.register
    def _(self, n: StringParamRange) -> str:
        return f'{", ".join(visit_each(self, n.values))} :: {self.visit(n.comment)}'
