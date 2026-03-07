from typing import cast
from collections import OrderedDict

from multimethod import multimethod
from sympy import (Symbol, Expr, Basic, Pow, IndexedBase, Function, Piecewise)
from sympy.core.numbers import Zero, One, NegativeOne, Integer, Float, Pi, Rational
from sympy.core.operations import AssocOp
from sympy.core.relational import Relational

from EmitCactus import mkSymbol
from EmitCactus.dsl.dsl_exception import DslException


class SplitMaxxer:
    def __init__(self, name_base: str):
        self.new_eqns: OrderedDict[Symbol, Expr] = OrderedDict()
        self.new_eqns_inv: OrderedDict[Expr, Symbol] = OrderedDict()
        self._name_base = name_base
        self._name_counter = 0

    def _new_symbol(self) -> Symbol:
        name = f"{self._name_base}_splitmaxxed_{self._name_counter}"
        s = mkSymbol(name)
        self._name_counter += 1
        return s

    def _put_new_eqn(self, new_expr: Expr) -> Symbol:
        if (existing_symbol := self.new_eqns_inv.get(new_expr, None)) is not None:
            return existing_symbol
        else:
            new_symbol = self._new_symbol()
            self.new_eqns[new_symbol] = new_expr
            self.new_eqns_inv[new_expr] = new_symbol
            return new_symbol

    @multimethod
    def visit(self, expr: Basic, top: bool = False) -> Expr:
        raise NotImplementedError(f"visit({expr.func}) not implemented in SplitMaxxer expr={expr}")

    @visit.register
    def _(self, expr: AssocOp, top: bool = False) -> Expr:
        new_args: list[Expr] = []

        assert len(expr.args) >= 2, f"Expected 2 arguments for {expr.func}, got {len(expr.args)}"

        for arg in expr.args[:2]:
            new_args.append(self.visit(arg))

        if len(expr.args) == 2 and top:
            return cast(Expr, expr.func(*new_args))

        new_symbol = self._put_new_eqn(expr.func(*new_args[:2]))
        return cast(Expr, self.visit(expr.func(new_symbol, *expr.args[2:]), top=top))

    @visit.register
    def _(self, expr: Relational | Pow | Function, top: bool = False) -> Expr:  # AssocOp is the supertype of Add and Mul
        new_args: list[Expr] = []

        for arg in expr.args:
            new_args.append(self.visit(arg))

        if top or hasattr(expr, 'name') and expr.name == 'noop':
            return cast(Expr, expr.func(*new_args))

        return self._put_new_eqn(expr.func(*new_args))

    @visit.register
    def _(self, expr: Symbol, top: bool = False) -> Expr:
        return expr

    @visit.register
    def _(self, expr: Zero | One | NegativeOne | Integer | Float | Pi | Rational, top: bool = False) -> Expr:
        return expr

    @visit.register
    def _(self, expr: IndexedBase, top: bool = False) -> Expr:
        raise DslException(f"IndexedBase {expr} not expected in SplitMaxxer")

    @visit.register
    def _(self, expr: Piecewise, top: bool = False) -> Expr:
        return expr  # todo: We really only use Piecewise in very limited and simple cases (e.g., step function), and I don't feel like implementing them right now.