"""
Use the Sympy Indexed type for relativity expressions.
"""
import re
import sys
import typing
from collections import defaultdict
from enum import auto
from itertools import chain
from enum import auto, Enum
from typing import *

import sympy.logic.boolalg
from multimethod import multimethod
from mypy_extensions import VarArg
from nrpy.finite_difference import setup_FD_matrix__return_inverse_lowlevel
from nrpy.helpers.coloring import coloring_is_enabled as colorize
from sympy import Integer, Eq, Symbol, Indexed, IndexedBase, Matrix, Idx, Basic, MatrixBase, exp, \
    ImmutableDenseMatrix, Expr
from sympy.core.relational import Relational

from EmitCactus.dsl.coef import coef
from EmitCactus.dsl.dsl_exception import DslException
from EmitCactus.dsl.eqnlist import EqnList, DXI, DYI, DZI, DX, DY, DZ, EqnComplex
from EmitCactus.dsl.symm import Sym
from EmitCactus.dsl.sympywrap import *
from EmitCactus.dsl.temporary_promotion_predicate import TemporaryPromotionStrategy, promote_all
from EmitCactus.emit.ccl.interface.interface_tree import TensorParity, Parity, SingleIndexParity
from EmitCactus.emit.ccl.schedule.schedule_tree import ScheduleBlock, GroupOrFunction
from EmitCactus.emit.tree import Centering, Identifier
from EmitCactus.util import OrderedSet, ScheduleBinEnum, get_or_compute, ScheduleFrequency

__all__ = ["D", "div", "to_num", "IndexedSubstFnType", "MkSubstType", "Param", "ThornFunction", "ScheduleBin",
           "ThornDef",
           "set_dimension", "get_dimension", "lookup_pair", "subst_tensor", "subst_tensor_xyz", "mk_pair",
           "noop", "stencil", "DD", "DDI",
           "ui", "uj", "uk", "ua", "ub", "uc", "ud", "u0", "u1", "u2", "u3", "u4", "u5",
           "li", "lj", "lk", "la", "lb", "lc", "ld", "l0", "l1", "l2", "l3", "l4", "l5"]

from .temp_kind import TempKind
from ..generators.sympy_complexity import SympyComplexityVisitor

one = sympify(1)
zero = sympify(0)

lookup_pair: Dict[Idx, Idx] = dict()


###
def mk_mk_subst(s: str) -> str:
    next_sub = 'a'
    pos = 0
    new_s = ""
    for g in re.finditer(r'\b([ul])([0-9])\b', s):
        new_s += s[pos:g.start()]
        pos = g.end()
        up_down = g.group(1)
        _index = g.group(2)
        new_s += up_down
        new_s += next_sub
        next_sub = chr(ord(next_sub) + 1)
    new_s += s[pos:]
    return new_s


###
import sympy as sy


class InvalidIndexError(DslException):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class IndexTracker:
    def __init__(self) -> None:
        self.free: OrderedSet[Idx] = OrderedSet()
        self.contracted: OrderedSet[Idx] = OrderedSet()
        self.used: OrderedSet[Idx] = OrderedSet()

    def all(self) -> OrderedSet[Idx]:
        """
        The set of all contracted and free.
        """
        ret: OrderedSet[Idx] = OrderedSet()
        for a in self.free:
            ret.add(a)
        for a in self.contracted:
            ret.add(a)
            ret.add(lookup_pair[a])
        return ret

    def used_overlap(self, used: OrderedSet[Idx]) -> bool:
        for u in self.used:
            if u in used:
                return True
        for u in used:
            if u in self.used:
                return True
        return False

    def add(self, idx: Idx) -> bool:
        """
        We keep single indices. So if we get ua and la,
        only la goes in contracted. If we get ua and lc,
        both indices go in free. Used should not be added
        here.
        """
        global lookup_pair
        if (idx in self.free) or (idx in self.contracted):
            return False
        # TODO: Factor this logic out elsewhere
        letter_or_num = ord(str(idx)[1])
        if ord('0') <= letter_or_num <= ord('9'):
            return True
        pdx = lookup_pair.get(idx, None)
        assert pdx is not None, f"{idx} not in {lookup_pair}"
        if pdx in self.free:
            self.free.remove(pdx)
            if str(idx)[0] == 'u':
                assert pdx not in self.contracted
                self.contracted.add(pdx)
            else:
                assert idx not in self.contracted
                self.contracted.add(idx)
        else:
            self.free.add(idx)
        return True

    def __repr__(self) -> str:
        return "(free:" + repr(self.free) + ", contracted:" + repr(self.contracted) + ", used:" + repr(self.used) + ")"


class IndexContractionVisitor:
    def __init__(self, defn: Dict[str, Tuple[str, List[Idx]]]) -> None:
        self.defn = defn

    @multimethod
    def visit(self, expr: sy.Basic) -> Tuple[Expr, IndexTracker]:
        raise Exception(str(expr) + " " + str(type(expr)))

    @visit.register
    def _(self, expr: sy.Add) -> Tuple[Expr, IndexTracker]:
        it: Optional[IndexTracker] = None
        last_arg = None
        new_expr = zero
        for a in expr.args:
            a_expr, a_it = self.visit(a)
            a_expr, a_it = self.contract(a_expr, a_it)
            new_expr += a_expr
            if it is None:
                it = a_it
            # TODO: check for used/free mismatch
            if it.free != a_it.free:
                raise InvalidIndexError(f"Invalid indices in add '{a}:{it.free}' != '{last_arg}:{a_it.free}':")
            last_arg = a
        if it is None:
            return new_expr, IndexTracker()
        else:
            return new_expr, it

    @staticmethod
    def contract(expr: Expr, it: IndexTracker) -> Tuple[Expr, IndexTracker]:
        for lo_idx in it.contracted:
            new_expr: Expr = zero
            up_idx = lookup_pair[lo_idx]
            for i in range(get_dimension()):
                lo_idx_val = [l0, l1, l2, l3, l4, l5][i]
                up_idx_val = [u0, u1, u2, u3, u4, u5][i]
                new_expr += do_isub(expr, dict(), {lo_idx: lo_idx_val, up_idx: up_idx_val})
            expr = new_expr
        it.used = it.contracted
        it.contracted = OrderedSet()
        return expr, it

    @visit.register
    def _(self, expr: sy.Mul) -> Tuple[Expr, IndexTracker]:
        it = IndexTracker()
        new_expr = one
        for a in expr.args:
            a_expr, a_it = self.visit(a)
            if a_it.used_overlap(it.used):
                raise InvalidIndexError(repr(expr))
            new_expr *= a_expr
            for idx in a_it.used:
                it.used.add(idx)
            for idx in a_it.all():
                if not it.add(idx):
                    raise InvalidIndexError(repr(expr))
        return self.contract(new_expr, it)

    @visit.register
    def _(self, expr: sy.Piecewise) -> Tuple[Expr, IndexTracker]:
        it = IndexTracker()
        expr_args = typing.cast(Iterable[Tuple[Expr, Expr]], expr.args)
        new_args: List[Tuple[Expr, Expr]] = list()

        for (e, c) in expr_args:
            if isinstance(e, Idx):
                if not it.add(e):
                    raise InvalidIndexError(repr(expr))
                new_args.append((e, c))
            else:
                e_expr, e_it = self.visit(e)
                c_expr, c_it = self.visit(c)
                new_args.append((e_expr, c_expr))

                for idx in e_it.all():
                    it.add(idx)

                for idx in c_it.all():
                    it.add(idx)

        ret = self.contract(expr.func(*new_args), it)

        return ret

    @visit.register
    def _relational(self, expr: Relational) -> Tuple[Expr, IndexTracker]:
        it = IndexTracker()
        new_args: List[Expr] = list()
        for a in expr.args:
            if isinstance(a, Idx):
                if not it.add(a):
                    raise InvalidIndexError(repr(expr))
                new_args.append(a)
            else:
                a_expr, a_it = self.visit(a)
                new_args.append(a_expr)
                for idx in a_it.all():
                    it.add(idx)
        ret = self.contract(expr.func(*new_args), it)
        return ret

    @visit.register
    def _(self, expr: sy.core.numbers.Pi) -> Tuple[Expr, IndexTracker]:
        return expr, IndexTracker()

    @visit.register
    def _(self, expr: sy.Symbol) -> Tuple[Expr, IndexTracker]:
        return expr, IndexTracker()

    @visit.register
    def _(self, expr: sy.Integer) -> Tuple[Expr, IndexTracker]:
        return expr, IndexTracker()

    @visit.register
    def _(self, expr: sy.Rational) -> Tuple[Expr, IndexTracker]:
        return expr, IndexTracker()

    @visit.register
    def _(self, expr: sy.Float) -> Tuple[Expr, IndexTracker]:
        return expr, IndexTracker()

    @visit.register
    def _(self, expr: sy.Idx) -> Tuple[Expr, IndexTracker]:
        return expr, IndexTracker()

    @visit.register
    def _(self, _expr: sympy.logic.boolalg.BooleanTrue) -> Tuple[Expr, IndexTracker]:
        return sympify(True), IndexTracker()

    @visit.register
    def _(self, _expr: sympy.logic.boolalg.BooleanFalse) -> Tuple[Expr, IndexTracker]:
        return sympify(False), IndexTracker()

    @visit.register
    def _(self, expr: sy.Indexed) -> Tuple[Expr, IndexTracker]:
        basename = str(expr.args[0])
        if basename in self.defn:
            bn, indices = self.defn[basename]
            if len(indices) + 1 != len(expr.args):
                raise InvalidIndexError(f"indices used on a non-indexed quantity '{expr}' in:")
        else:
            assert len(self.defn) == 0
        it = IndexTracker()
        for a in expr.args[1:]:
            _a_it = self.visit(a)
            assert isinstance(a, Idx)
            if not it.add(a):
                raise InvalidIndexError(str(expr))
        return self.contract(expr, it)

    @visit.register
    def _(self, expr: sy.Function) -> Tuple[Expr, IndexTracker]:
        it = IndexTracker()
        new_args: List[Expr] = list()
        for a in expr.args:
            if isinstance(a, Idx):
                if not it.add(a):
                    raise InvalidIndexError(repr(expr))
                new_args.append(a)
            else:
                a_expr, a_it = self.visit(a)
                new_args.append(a_expr)
                for idx in a_it.all():
                    it.add(idx)
        ret = self.contract(expr.func(*new_args), it)
        return ret

    @visit.register
    def _(self, expr: sy.Pow) -> Tuple[Expr, IndexTracker]:
        new_args: List[Expr] = list()
        for a in expr.args:
            new_arg, it = self.visit(a)
            new_args += [new_arg]
            if len(it.free) != 0 or len(it.contracted) != 0:
                raise InvalidIndexError(repr(expr))
        return sy.Pow(*new_args), IndexTracker()

    @visit.register
    def _(self, expr: sy.IndexedBase) -> Tuple[Expr, IndexTracker]:
        basename = str(expr)
        if basename not in self.defn:
            if len(self.defn) == 0:
                n = 0
            else:
                raise InvalidIndexError(f"Undefined symbol in '{self.defn}':")
        else:
            bn, indices = self.defn[basename]
            n = len(indices)
        if n != 0:
            if n == 1:
                msg = "1 index"
            else:
                msg = f"{n} indices"
            raise InvalidIndexError(
                f"Expression '{expr}' was declared with {msg}, but was used in this expression without indices: ")
        return expr, IndexTracker()


### ind subs
class IndexSubsVisitor:
    def __init__(self, defn: Dict[Indexed|IndexedBase, Expr]) -> None:
        self.defn = defn
        self.idx_subs: Dict[Idx, Idx] = dict()

    @multimethod
    def visit(self, expr: sy.Add) -> Expr:
        r = sympify(0)
        for a in expr.args:
            r += self.visit(a)
        return r

    @visit.register
    def _(self, expr: sy.Mul) -> Expr:
        r = sympify(1)
        for a in expr.args:
            r *= self.visit(a)
        return r

    @visit.register
    def _(self, expr: sy.Symbol) -> Expr:
        return expr

    @visit.register
    def _(self, expr: sy.Integer) -> Expr:
        return expr

    @visit.register
    def _(self, expr: sy.Rational) -> Expr:
        return expr

    @visit.register
    def _(self, expr: sy.Float) -> Expr:
        return expr

    @visit.register
    def _(self, expr: sy.core.numbers.Pi) -> Expr:
        return expr

    @visit.register
    def _(self, expr: sy.Idx) -> Expr:
        res = self.idx_subs.get(expr, None)
        if res is None:
            return expr
        else:
            return res

    @visit.register
    def _(self, expr: sy.IndexedBase) -> Expr:
        return self.defn.get(expr, expr)

    @visit.register
    def _(self, expr: sy.Indexed) -> Expr:
        r: Indexed = expr
        if len(self.idx_subs) > 0:
            indexes: List[Idx] = list()
            for a in expr.args[1:]:
                assert isinstance(a, Idx)
                indexes.append(self.idx_subs.get(a, a))
            r = mkIndexed(expr.base, *indexes)
        res = self.defn.get(r, None)
        if res is None:
            return r
        else:
            return res

    @visit.register
    def _(self, expr: sy.Function) -> Expr:
        f = expr.func
        args = tuple([self.visit(a) for a in expr.args])
        r = f(*args)
        assert isinstance(r, Expr)
        return r

    @visit.register
    def _(self, expr: sy.Piecewise) -> Expr:
        expr_args = typing.cast(Iterable[Tuple[Expr, Expr]], expr.args)
        args = tuple([(self.visit(e), self.visit(c)) for (e, c) in expr_args])
        return mkPiecewise(*args)

    @visit.register
    def _(self, expr: Relational) -> Expr:
        expr_args = tuple([self.visit(a) for a in expr.args])
        return typing.cast(Expr, expr.func(*expr_args))

    @visit.register
    def _(self, _expr: sympy.logic.boolalg.BooleanTrue) -> Expr:
        return sympify(True)

    @visit.register
    def _(self, _expr: sympy.logic.boolalg.BooleanFalse) -> Expr:
        return sympify(False)

    @visit.register
    def _(self, expr: sy.Pow) -> Expr:
        return cast(Expr, sy.Pow(self.visit(expr.args[0]), self.visit(expr.args[1])))


def do_isub(expr: Expr, subs: Optional[Dict[Indexed|IndexedBase, Expr]] = None, idx_subs: Optional[Dict[Idx, Idx]] = None) -> Expr:
    if subs is None:
        subs = dict()
    if idx_subs is None:
        idx_subs = dict()
    isub = IndexSubsVisitor(subs)
    isub.idx_subs = idx_subs
    # FIXME Why is this cast needed?
    return cast(Expr, isub.visit(expr))


def check_indices(rhs: Expr, defn: Optional[Dict[str, Tuple[str, List[Idx]]]] = None) -> IndexTracker:
    """
    This function not only checks the validity of indexed expressions, it returns
    all free and contracted indices.
    """

    if defn is None:
        defn = dict()

    err = IndexContractionVisitor(defn)
    ret: IndexTracker
    _, ret = err.visit(rhs)
    return ret


###
# Need Expand Visitor
###

####
# Generic derivatives
div = mkFunction("div")
D = mkFunction("D")
"""
Symbolic derivative function.
"""

# This is required due to a bug in pdoc.
if div.__module__ is None:
    div.__module__ = "use_indices"

pair_tmp_name = "A"


def mk_pair(s: Optional[str] = None) -> Tuple[Idx, Idx]:
    """
    Returns a tuple containing an upper/lower index pair.
    """
    global pair_tmp_name
    if s is None:
        s = pair_tmp_name
        tmp_num = ord(pair_tmp_name[-1])
        if tmp_num == ord("Z"):
            pair_tmp_name += "A"
        else:
            pair_tmp_name = pair_tmp_name[0:-1] + chr(tmp_num + 1)
        tmp_num += 1
    u, l = mkIdxs(f"u{s} l{s}")
    lookup_pair[l] = u
    lookup_pair[u] = l
    return u, l


def is_down(ind: Idx) -> bool:
    s = str(ind)
    assert s[0] in ["u", "l"], f"ind={ind}"
    return s[0] == "l"


def to_num(ind: Idx) -> int:
    s = str(ind)
    assert s[0] in ["u", "l"], f"ind={ind}"
    return int(s[1])


# Some basic indices to use
ui, li = mk_pair('i')
uj, lj = mk_pair('j')
uk, lk = mk_pair('k')
ua, la = mk_pair('a')
ub, lb = mk_pair('b')
uc, lc = mk_pair('c')
ud, ld = mk_pair('d')
u0, l0 = mk_pair('0')
u1, l1 = mk_pair('1')
u2, l2 = mk_pair('2')
u3, l3 = mk_pair('3')
u4, l4 = mk_pair('4')
u5, l5 = mk_pair('5')
up_indices = u0, u1, u2, u3, u4, u5
down_indices = l0, l1, l2, l3, l4, l5

### dmv
from .sympywrap import *

x = mkSymbol("x")
y = mkSymbol("y")
z = mkSymbol("z")
t = mkSymbol("t")
no_idx = mkIdx("no_idx")
dummy = mkSymbol("_dummy_")


def _mk_div(div_fun: UFunc, expr: Expr, *args: Idx) -> Expr:
    r = div_fun(expr, *args)
    assert isinstance(r, Expr)
    return r


class DivMakerVisitor:
    def __init__(self, div_fun: UFunc, coords: Optional[List[Symbol]] = None) -> None:
        self.div_func = div_fun
        self.div_name = str(div_fun)
        self.params: Set[Symbol] = set()
        if coords is None:
            coords = [x, y, z]
        self.coords = coords
        self.idx_map = dict()
        for i in range(len(coords)):
            self.idx_map[coords[i]] = down_indices[i]

    @multimethod
    def visit(self, expr: sy.Basic, idx: sy.Idx) -> Expr:
        raise Exception(str(expr) + " " + str(type(expr)))

    standard_fn_divs: dict[sy.Function, Callable[['DivMakerVisitor', sy.Expr, sy.Idx], sy.Expr]] = {
        sy.sin: lambda self, r, idx: cos(r) * self.visit(r, idx),
        sy.cos: lambda self, r, idx: -sin(r) * self.visit(r, idx),
        sy.tan: lambda self, r, idx: sec(r) ** 2 * self.visit(r, idx),
        sy.cot: lambda self, r, idx: -csc(r) ** 2 * self.visit(r, idx),
        sy.sec: lambda self, r, idx: sec(r) * tan(r) * self.visit(r, idx),
        sy.csc: lambda self, r, idx: -csc(r) * cot(r) * self.visit(r, idx),
        sy.exp: lambda self, r, idx: exp(r) * self.visit(r, idx),
        sy.log: lambda self, r, idx: (1 / r) * self.visit(r, idx),
        sy.cosh: lambda self, r, idx: sinh(r) * self.visit(r, idx),
        sy.sinh: lambda self, r, idx: cosh(r) * self.visit(r, idx),
        sy.tanh: lambda self, r, idx: sech(r) ** 2 * self.visit(r, idx),
        sy.coth: lambda self, r, idx: -csch(r) ** 2 * self.visit(r, idx),
        sy.sech: lambda self, r, idx: -sech(r) * tanh(x) * self.visit(r, idx),
        sy.csch: lambda self, r, idx: -csch(r) * coth(x) * self.visit(r, idx),
        sy.erf: lambda self, r, idx: 2 * exp(-r ** 2) / sqrt(pi) * self.visit(r, idx)
    }

    @visit.register
    def _(self, expr: sy.Add, idx: sy.Idx) -> Expr:
        r = zero
        for a in expr.args:
            r += self.visit(a, idx)
        return r

    @visit.register
    def _(self, expr: sy.Mul, idx: sy.Idx) -> Expr:
        if idx is not no_idx:
            s = zero
            for i in range(len(expr.args)):
                term = one
                for j in range(len(expr.args)):
                    a = expr.args[j]
                    if i == j:
                        term *= self.visit(a, idx)
                    else:
                        term *= self.visit(a, no_idx)
                s += term
            return s
        else:
            s = one
            for a in expr.args:
                s *= self.visit(a, no_idx)
            return s

    @visit.register
    def _(self, expr: sy.Symbol, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        ####
        # TODO: generalize for other dimensions than 3
        # assert get_dimension()==3
        if idx == l0:
            if expr == x:
                return one
            elif expr in [y, z]:
                return zero
            elif expr in self.params:
                return zero

        elif idx == l1:
            if expr == y:
                return one
            elif expr in [x, z]:
                return zero
            elif expr in self.params:
                return zero

        elif idx == l2:
            if expr == z:
                return one
            elif expr in [x, y]:
                return zero
            elif expr in self.params:
                return zero

        else:
            raise Exception(f"Bad index passed to derivative: {expr}: idx={idx}")

        return _mk_div(self.div_func, expr, idx)

    @visit.register
    def _(self, expr: sy.Integer, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        return zero

    @visit.register
    def _(self, expr: sy.core.numbers.Pi, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        return zero

    @visit.register
    def _(self, expr: sy.Rational, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        return zero

    @visit.register
    def _(self, expr: sy.Float, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        return zero

    @visit.register
    def _(self, expr: sy.Idx, idx: sy.Idx) -> Expr:
        raise DslException("Derivative of Index")

    @visit.register
    def _(self, expr: sy.Piecewise, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        return zero

    @visit.register
    def _(self, expr: sy.Indexed, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        return _mk_div(self.div_func, expr, idx)

    @visit.register
    def _(self, expr: sy.IndexedBase, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        return _mk_div(self.div_func, expr, idx)

    @visit.register
    def _(self, expr: sy.Function, idx: sy.Idx) -> Expr:
        r = expr.args[0]

        if not isinstance(r, Expr):
            raise DslException("Expected the first argument/term of " + str(expr) + " to be an expression")

        name = expr.func.__name__
        if name == self.div_name:
            # Handle div of div
            sub: Expr = self.visit(r, no_idx)
            if len(expr.args) > 2:
                for idx1 in expr.args[1:]:
                    sub = self.visit(sub, idx1)
                return sub
            if isinstance(sub, sy.Function) and sub.func.__name__ == self.div_name:
                args = sorted(sub.args[1:] + expr.args[1:], key=lambda x: str(x))
                return _mk_div(self.div_func, cast(Expr, sub.args[0]), *args)

            for idx1 in expr.args[1:]:
                sub = self.visit(sub, idx1)

            if idx is not no_idx:
                sub = self.visit(self.div_func(sub, idx), no_idx)

            return sub
        elif idx is no_idx:
            return expr
        else:
            if expr.func in self.standard_fn_divs:
                f = self.standard_fn_divs[expr.func](self, r, idx)
            elif len(expr.args) == 1:
                fd = mkFunction(name+"'")
                f = fd(r) * self.visit(r, idx)
            else:
                raise DslException(f"Derivative of {expr} is not handled by EmitCactus")
            assert isinstance(f, Expr)
            return f

    @visit.register
    def _(self, expr: sy.Pow, idx: sy.Idx) -> Expr:
        if idx is no_idx:
            return expr
        else:
            r = expr.args[0]
            n = expr.args[1]
            ret = n * r ** (n - 1) * self.visit(r, idx)
            assert isinstance(ret, Expr)
            return ret


dmv = DivMakerVisitor(div)
dmv2 = DivMakerVisitor(D)


def do_div(expr: Basic) -> Expr:
    r = dmv.visit(expr, no_idx)
    r = dmv2.visit(r, no_idx)
    assert isinstance(r, Expr)
    return r


### dmv

TA = TypeVar("TA")


def checked_cast(obj: Any, typ: Type[TA]) -> TA:
    """
    Checked cast
    """
    assert isinstance(obj, typ), f"expected type {typ} found type {type(obj)}"
    return obj


def sub_idxs(idx: Idx, values: Dict[Idx, Idx]) -> Idx:
    return checked_cast(do_subs(idx, values), Idx)


def to_num_tup_2(li: List[Idx], values: Dict[Idx, Idx]) -> Tuple[int, ...]:
    return tuple([to_num(sub_idxs(x, values)) for x in li])


def to_num_tup(li: Tuple[Basic, ...], values: Dict[Idx, Idx]) -> Tuple[int, ...]:
    return to_num_tup_2([checked_cast(x, Idx) for x in li], values)


stencil = mkFunction("stencil")
DD = mkFunction("DD")
DDI = mkFunction("DDI")
noop = mkFunction("noop")

dimension: int = 3


def set_dimension(d: int) -> None:
    global dimension
    dimension = d


def get_dimension() -> int:
    return dimension


ord0 = ord('0')
ord9 = ord('9')


def is_letter_index(sym: Basic) -> bool:
    if type(sym) != Idx:
        return False
    s = str(sym)
    if sym not in lookup_pair:
        return False
    if s[0] not in ["u", "l"]:
        return False
    n = ord(s[1])
    return n < ord0 or n > ord9


def get_indices(xpr: Expr) -> OrderedSet[Idx]:
    """ Return all indices of IndexedBase objects in xpr. """
    ret: OrderedSet[Idx] = OrderedSet()
    for symbol in free_indexed(xpr):
        if is_letter_index(symbol):
            ret.add(symbol)
    return ret


def by_name(x: Idx) -> str:
    """ Return a string suitable for sorting a list of upper/lower indices. """
    s = str(x)
    assert x in lookup_pair
    return s[1:] + s[0]


num0 = ord('0')
num9 = ord('9')


def is_numeric_index(x: Idx) -> bool:
    s = str(x)
    assert x in lookup_pair
    n = ord(s[1])
    return num0 <= n <= num9


def is_lower(x: Idx) -> bool:
    s = str(x)
    return s[0] == 'l'


def is_upper(x: Idx) -> bool:
    s = str(x)
    return s[0] == 'u'


def get_pair(x: Idx) -> Tuple[Idx, Idx]:
    if is_lower(x):
        return lookup_pair[x], x
    else:
        return x, lookup_pair[x]


def is_pair(a: Idx, b: Idx) -> bool:
    sa = str(a)
    sb = str(b)
    assert a in lookup_pair
    assert b in lookup_pair
    if sa[1:] == sb[1:] and ((sa[0] == 'u' and sb[0] == 'l') or (sa[0] == 'l' and sb[0] == 'u')):
        return True
    else:
        return False


# Check that this works
assert is_pair(ui, li)
assert is_pair(li, ui)
assert not is_pair(ui, lj)
assert not is_pair(li, uj)


def get_free_indices(xpr: Expr) -> OrderedSet[Idx]:
    """ Return all uncontracted indices in xpr. """
    indices = list(get_indices(xpr))
    indices = sorted(indices, key=by_name)
    ret: OrderedSet[Idx] = OrderedSet()
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and is_pair(indices[i], indices[i + 1]):
            i += 2
        else:
            ret.add(indices[i])
            i += 1
    return ret


def get_contracted_indices(xpr: Expr) -> OrderedSet[Idx]:
    """ Return all contracted indices in xpr. """
    indices = list(get_indices(xpr))
    indices = sorted(indices, key=by_name)
    ret: OrderedSet[Idx] = OrderedSet()
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and is_pair(indices[i], indices[i + 1]):
            ret.add(indices[i])
            i += 2
        else:
            i += 1
    return ret


def incr(index_list: List[Idx], index_values: Dict[Idx, Idx]) -> bool:
    """ Increment the indices in index_list, creating an index_values table with all possible permutations. """
    if len(index_list) == 0:
        return False
    ix = 0
    if len(index_values) == 0:
        for ind_ in index_list:
            u_ind, ind = get_pair(ind_)
            index_values[ind] = l0
            index_values[u_ind] = u0
        return True
    while True:
        if ix >= len(index_list):
            return False
        u_ind, ind = get_pair(index_list[ix])
        index_value = to_num(index_values[ind])
        if index_value == dimension - 1:
            index_values[ind] = l0
            index_values[u_ind] = u0
            ix += 1
        else:
            index_values[ind] = down_indices[index_value + 1]
            index_values[u_ind] = up_indices[index_value + 1]
            break
    return True


def expand_contracted_indices(in_expr: Expr, sym: Sym) -> Expr:
    viz = IndexContractionVisitor(dict())
    expr, it = viz.visit(in_expr)
    expr = sym.apply(expr)
    assert isinstance(expr, Expr)
    return expr


def expand_free_indices(xpr: Expr, sym: Sym) -> List[Tuple[Expr, Dict[Idx, Idx], List[Idx]]]:
    index_list: List[Idx] = sorted(list(get_free_indices(xpr)), key=str)
    output: List[Tuple[Expr, Dict[Idx, Idx], List[Idx]]] = list()
    xpr = expand_contracted_indices(xpr, sym)
    index_values: Dict[Idx, Idx] = dict()
    while incr(index_list, index_values):
        assert len(index_values) != 0, "Something very bad happened"
        if type(xpr) == Indexed:
            result = do_subs(xpr, index_values)
            sym_result = sym.apply(result)
            if result != sym_result:
                continue
        out_xpr = do_subs(xpr, index_values, sym)
        output += [(out_xpr, index_values.copy(), index_list)]
    return output


def _mk_name_for_tensor(sym: Indexed) -> str:
    base_name = str(sym.base)

    for ind in sym.args[1:]:
        assert isinstance(ind, Idx)
        if is_lower(ind):
            base_name += "D"
        elif is_upper(ind):
            base_name += "U"
        else:
            raise DslException(f"Index {ind} in {sym} does not follow the correct naming convention."
                               f"Lower indices must be prefixed with l, and upper indices with u.")
    for ind in sym.args[1:]:
        assert isinstance(ind, Idx)
        base_name += str(to_num(ind))

    return base_name


def subst_tensor(sym: Indexed, *_idxs: int) -> Expr:
    """
    Defines a symbol for a tensor using standard NRPy+ rules.
    For an upper index put a U, for a lower index put a D.
    Follow the string of U's and D's with the integer value
    of the up/down index.

    :param sym: The tensor expression with integer indices.

    :return: A new SymPy symbol.
    """

    return mkSymbol(_mk_name_for_tensor(sym))


def _mk_name_for_tensor_xyz(sym: Indexed, *_args: Idx) -> str:
    base_name = str(sym.base)
    for ind in sym.args[1:]:
        assert isinstance(ind, Idx)
        base_name += ["x", "y", "z"][to_num(ind)]
    return base_name


def subst_tensor_xyz(sym: Indexed, *_idxs: int) -> Symbol:
    """
    Defines a symbol for a tensor using standard Cactus rules.
    Don't distinguish up/down indices. Use suffixes based on
    x, y, and z at the end.

    :param sym: The tensor expression with integer indices.

    :return: A new sympy symbol
    """
    return mkSymbol(_mk_name_for_tensor_xyz(sym))


BaseIndexedSubstFnType = Callable[[Indexed, VarArg(int)], Expr]
IndexedSubstFnType = (
        Callable[[Indexed, int], Expr] |
        Callable[[Indexed, int, int], Expr] |
        Callable[[Indexed, int, int, int], Expr] |
        BaseIndexedSubstFnType
)
MkSubstType = IndexedSubstFnType | Expr | ImmutableDenseMatrix | MatrixBase

ParamDefaultType = Union[float, int, str, bool]
ParamValuesType = Optional[Union[Tuple[float, float], Tuple[int, int], Tuple[bool, bool], str, Set[str]]]
MinMaxType = Union[Tuple[float, float], Tuple[int, int]]


class Param:
    def __init__(self, name: str, default: ParamDefaultType, desc: str, values: ParamValuesType) -> None:
        self.name = name
        self.values = values
        self.desc = desc
        self.default = default

    def get_min_max(self) -> MinMaxType:
        ty = self.get_type()
        if ty == int:
            if self.values is not None:
                return cast(MinMaxType, self.values)
            return -2 ** 31, 2 ** 31 - 1
        elif ty == float:
            if self.values is not None:
                return cast(MinMaxType, self.values)
            return sys.float_info.min, sys.float_info.max
        else:
            assert False

    def get_values(self) -> ParamValuesType:
        if self.values is not None:
            return self.values
        ty = self.get_type()
        if ty == bool:
            return False, True
        elif ty == str:
            return ".*"
        else:
            return self.get_min_max()

    def get_type(self) -> Type[Any]:
        if self.values is None:
            return type(self.default)
        elif isinstance(self.values, set):
            assert isinstance(self.default, str)
            return set  # keywords
        elif isinstance(self.values, str):
            # values is a regex
            assert isinstance(self.default, str)
            return str
        elif isinstance(self.values, tuple) and len(self.values) == 2:
            assert type(self.default) in [int, float]
            assert type(self.values[0]) in [int, float]
            assert type(self.values[1]) in [int, float]
            if isinstance(self.default, float) or isinstance(self.values[0], float) or isinstance(self.values[1],
                                                                                                  float):
                return float
            else:
                return int
        else:
            assert False

    def __repr__(self) -> str:
        return f"Param({self.name})"


# First derivatives
for i in range(dimension):
    div_nm = "div" + "xyz"[i]
    globals()[div_nm] = mkFunction(div_nm)

# Second derivatives
for i in range(dimension):
    for j in range(i, dimension):
        div_nm = "div" + "xyz"[i] + "xyz"[j]
        globals()[div_nm] = mkFunction(div_nm)


def to_div(out: Expr) -> Expr:
    nm = "div"
    for k in out.args[1:]:
        assert isinstance(k, Idx)
        nm += "xyz"[to_num(k)]
    div_nn = mkFunction(nm)
    arg = out.args[0]  # div(v, i, j) -> v
    return cast(Expr, div_nn(arg))


class ApplyDiv(Applier):
    def __init__(self) -> None:
        self.val: Optional[Expr] = None

    def m(self, expr: Expr) -> bool:
        # noinspection PyUnresolvedReferences
        if expr.is_Function and hasattr(expr, "name") and expr.name == "div":
            for arg in expr.args[1:]:
                assert isinstance(arg, Idx)
                if not is_numeric_index(arg):
                    self.val = None
                    return False
            self.val = to_div(expr)
            return True
        else:
            self.val = None
            return False

    def r(self, _expr: Basic) -> Optional[Expr]:
        return self.val

    def apply(self, arg: Basic) -> Basic:
        return cast(Basic, arg.replace(self.m, self.r))  # type: ignore[no-untyped-call]


def mkterm(v: Basic, i: int, j: int, k: int) -> Any:
    """
    Create a stencil term for output. Note that
    the 0,0,0 element is special.
    """
    if i == 0 and j == 0 and k == 0:
        return v
    else:
        return stencil(v, i, j, k)


def sort_exprs(expr: Tuple[Any, Any]) -> float:
    sort_key: float = 2 * expr[0].p / expr[0].q
    if sort_key < 0:
        sort_key = -sort_key + 1
    return sort_key


class ApplyDivN(Applier):
    """
    Use NRPy to calculate the stencil coefficients.
    """

    def __init__(self, n: int, funs1: Dict[Tuple[UFunc, Idx], Expr], funs2: Dict[Tuple[UFunc, Idx, Idx], Expr],
                 fun_args: Dict[str, int]) -> None:
        self.val: Optional[Expr] = None
        self.n = n
        self.fd_matrix = setup_FD_matrix__return_inverse_lowlevel(n, 0)
        self.funs1 = funs1
        self.funs2 = funs2
        self.fun_args = fun_args

    def is_user_func(self, f: Expr) -> Optional[Expr]:
        f_func = f.func
        if not f.is_Function or not isinstance(f_func, UFunc):
            return None
        # noinspection PyUnresolvedReferences
        if hasattr(f, "name") and f.name in self.fun_args:
            nargs = self.fun_args[f.name]
            if len(f.args) != nargs:
                raise DslException(
                    f"function {f.name} called with wrong number of args. Expected {nargs}, got {len(f.args)}. Expr: {f}")
            return None
        elif len(f.args) == 2:
            _, arg1 = f.args
            if not isinstance(arg1, Idx):
                raise DslException(f"Expected an index argument but found {type(arg1)} in call {f}")
            return self.funs1.get((f_func, arg1), None)
        elif len(f.args) == 3:
            _, arg1, arg2 = f.args

            if not isinstance(arg1, Idx):
                raise DslException(f"Expected an index argument but found {type(arg1)} in first argument in call {f}")
            if not isinstance(arg2, Idx):
                raise DslException(f"Expected an index argument but found {type(arg2)} in second argument in call {f}")

            if arg1 == arg2:
                return self.funs1.get((f_func, arg1), None)
            else:
                # noinspection PyTypeChecker
                return self.funs2.get((f_func, arg1, arg2), None)
        return None

    def m(self, expr: Expr) -> bool:
        # noinspection PyUnresolvedReferences
        if (fun_def := self.is_user_func(expr)) is not None:
            arg0 = expr.args[0]
            if not isinstance(arg0, Expr):
                raise DslException("Expected the first argument/term of " + str(expr) + " to be an expression")
            self.val = do_subs(fun_def, {dummy: arg0})
            return True
        elif expr.is_Function and hasattr(expr, "name") and expr.name == "stencil":
            new_expr1: List[int | sy.Integer] = list()
            for arg in expr.args[1:]:
                if isinstance(arg, Idx):
                    new_expr1.append(to_num(arg))
                elif isinstance(arg, sy.Integer):
                    new_expr1.append(arg)
                else:
                    assert False, f"arg={arg}, type={type(arg)}"
            self.val = expr.func(expr.args[0], *new_expr1)
            return True

        elif expr.is_Function and hasattr(expr, "name") and expr.name in ["div", "D"]:
            new_expr = list()
            dxt = sympify(1)
            if len(expr.args) == 2:
                coefs = self.fd_matrix.col(1)
                if expr.args[1] == l0:
                    for i in range(len(coefs)):
                        term = coefs[i]
                        new_expr += [(term, mkterm(expr.args[0], i - len(coefs) // 2, 0, 0))]
                    dxt = DXI
                elif expr.args[1] == l1:
                    for i in range(len(coefs)):
                        term = coefs[i]
                        new_expr += [(term, mkterm(expr.args[0], 0, i - len(coefs) // 2, 0))]
                    dxt = DYI
                elif expr.args[1] == l2:
                    for i in range(len(coefs)):
                        term = coefs[i]
                        new_expr += [(term, mkterm(expr.args[0], 0, 0, i - len(coefs) // 2))]
                    dxt = DZI
            elif len(expr.args) == dimension:
                if expr.args[1:] == (l0, l0):
                    coefs = 2 * self.fd_matrix.col(2)
                    for i in range(len(coefs)):
                        term = coefs[i]
                        new_expr += [(term, mkterm(expr.args[0], i - len(coefs) // 2, 0, 0))]
                    dxt = DXI ** 2
                elif expr.args[1:] == (l1, l1):
                    coefs = 2 * self.fd_matrix.col(2)
                    for i in range(len(coefs)):
                        term = coefs[i]
                        new_expr += [(term, mkterm(expr.args[0], 0, i - len(coefs) // 2, 0))]
                    dxt = DYI ** 2
                elif expr.args[1:] == (l2, l2):
                    coefs = 2 * self.fd_matrix.col(2)
                    for i in range(len(coefs)):
                        term = coefs[i]
                        new_expr += [(term, mkterm(expr.args[0], 0, 0, i - len(coefs) // 2))]
                    dxt = DZI ** 2
                elif expr.args[1:] in ((l0, l1), (l1, l0)):
                    coefs = self.fd_matrix.col(1)
                    for i in range(len(coefs)):
                        term_i = coefs[i]
                        for j in range(len(coefs)):
                            term = coefs[j] * term_i
                            new_expr += [(term, mkterm(expr.args[0], i - len(coefs) // 2, j - len(coefs) // 2, 0))]
                    dxt = DXI * DYI
                elif expr.args[1:] in ((l0, l2), (l2, l0)):
                    coefs = self.fd_matrix.col(1)
                    for i in range(len(coefs)):
                        term_i = coefs[i]
                        for j in range(len(coefs)):
                            term = coefs[j] * term_i
                            new_expr += [(term, mkterm(expr.args[0], i - len(coefs) // 2, 0, j - len(coefs) // 2))]
                    dxt = DXI * DZI
                elif expr.args[1:] in ((l1, l2), (l2, l1)):
                    coefs = self.fd_matrix.col(1)
                    for i in range(len(coefs)):
                        term_i = coefs[i]
                        for j in range(len(coefs)):
                            term = coefs[j] * term_i
                            new_expr += [(term, mkterm(expr.args[0], 0, i - len(coefs) // 2, j - len(coefs) // 2))]
                    dxt = DYI * DZI
                else:
                    raise Exception()

            if len(new_expr) > 0:
                new_expr = sorted(new_expr, key=sort_exprs)
                self.val = sympify(0)
                i = 0
                while i < len(new_expr):
                    if i + 1 < len(new_expr) and abs(new_expr[i][0]) == abs(new_expr[i + 1][0]):
                        # We use noop for grouping because otherwise, Sympy will change things
                        if new_expr[i][0] != new_expr[i + 1][0]:
                            self.val += new_expr[i][0] * noop(new_expr[i][1] - new_expr[i + 1][1])
                        else:
                            self.val += new_expr[i][0] * noop(new_expr[i][1] + new_expr[i + 1][1])
                        i += 2
                    else:
                        self.val += new_expr[i][0] * new_expr[i][1]
                        i += 1
                self.val = self.val * dxt
            else:
                print("args:", expr.args)
            if self.val is None:
                raise Exception(str(expr))
            return True
        else:
            self.val = None
            return False

    def r(self, _expr: Expr) -> Optional[Expr]:
        return self.val

    def apply(self, arg: Basic) -> Basic:
        return cast(Basic, arg.replace(self.m, self.r))  # type: ignore[no-untyped-call]


val = mkSymbol("val")
x = mkSymbol("x")
y = mkSymbol("y")
z = mkSymbol("z")


class ScheduleBin(ScheduleBinEnum):
    Init = auto(), 'Init', True,  ScheduleFrequency.Once, 0
    DriverInit = auto(), 'ODESolvers_Initial', False, ScheduleFrequency.Once, 1
    PostInit = auto(), 'PostInit', True,  ScheduleFrequency.Once, 2
    Evolve = auto(), 'Evolve', False, ScheduleFrequency.EachStep, 3
    PostStep = auto(), 'ODESolvers_PostStep', False, ScheduleFrequency.EachStep, 4
    Analysis = auto(), 'Analysis', True, ScheduleFrequency.EachStep, 5
    EstimateError = auto(), 'EstimateError', False, ScheduleFrequency.Inconsistent, 6

    @staticmethod
    def _schedule_synthetic_fns(bins: Collection['ScheduleBin']) -> Collection['ScheduleBin']:
        ret: list['ScheduleBin'] = list()
        freqs: set[ScheduleFrequency] = set()
        bins = sorted(bins, key=lambda b: b.relative_order)

        for bin in bins:
            if bin.schedule_frequency == ScheduleFrequency.Inconsistent:
                freqs.add(bin.schedule_frequency)
                ret.append(bin)
                print(f'Warning: A global temp is accessed by a thorn function in schedule bin {bin}, which has an inconsistent schedule frequency. The temporary will be recomputed, perhaps redundantly.')
            elif bin is ScheduleBin.PostInit:  # Never elide PostInit targets. Needed for the timestep 0 PostInit hack.
                freqs.add(bin.schedule_frequency)
                ret.append(bin)
            elif len(freqs) > 0 and bin.schedule_frequency not in freqs:
                freqs.add(bin.schedule_frequency)
                ret.append(bin)
                print(f'Warning: A global temp is accessed by thorn functions in schedule bins {freqs} with disparate schedule frequencies. The temporary will be recomputed, perhaps redundantly.')
            elif len(freqs) == 0:
                freqs.add(bin.schedule_frequency)
                ret.append(bin)
            else:
                assert bin.schedule_frequency in freqs

        return ret


ScheduleTarget = ScheduleBin | ScheduleBlock
def safe_name(schedule_target: ScheduleTarget) -> str:
    if isinstance(schedule_target, ScheduleBlock):
        return schedule_target.name.identifier
    else:
        return schedule_target.generic_name


class ThornFunctionBakeOptions(TypedDict, total=False):
    do_cse: bool
    do_madd: bool
    do_recycle_temporaries: bool
    do_split_output_eqns: bool

class ThornFunction:
    """
    Represents a function within a Cactus thorn. Important member functions include `add_eqn` for specifying
    the computations this function will perform, and `bake` for finalizing the function.
    """

    def __init__(self,
                 name: str,
                 schedule_target: ScheduleTarget,
                 thorn_def: "ThornDef",
                 schedule_before: Optional[Collection[str]],
                 schedule_after: Optional[Collection[str]]) -> None:
        self.schedule_target = schedule_target
        self.name = name
        self.thorn_def = thorn_def
        self.eqn_complex: EqnComplex = EqnComplex(thorn_def.is_stencil)
        self.been_baked: bool = False
        self.schedule_before: Collection[str] = schedule_before or list()
        self.schedule_after: Collection[str] = schedule_after or list()

        if isinstance(schedule_target, ScheduleBlock) and schedule_target.group_or_function is GroupOrFunction.Function:
            raise DslException("Cannot schedule into this schedule block because it is not a schedule group.")

    @property
    def _eqn_list(self) -> EqnList:
        return self.eqn_complex.get_active_eqn_list()

    def _add_eqn2(self, lhs2: Symbol, rhs2: Expr) -> None:
        rhs2 = self.thorn_def.do_subs(expand_contracted_indices(rhs2, self.thorn_def.symmetries))
        for item in free_symbols(rhs2):
            if str(item) in self.thorn_def.params:
                assert item.is_Symbol
                self._eqn_list.add_param(item)
        divs = self.thorn_def.apply_div

        class FindBad:
            def __init__(self, outer: ThornFunction) -> None:
                self.outer = outer.thorn_def
                self.msg: Optional[str] = None

            def m(self, expr: Expr) -> bool:
                if isinstance(expr, Idx):
                    self.msg = f"Index passed to add_eqn: '{expr}'"
                elif type(expr) == Indexed:
                    if len(expr.args) != 1:
                        mms = mk_mk_subst(str(expr))
                        self.msg = f"'{expr}' does not evaluate a Symbol. Did you forget to call mk_subst({mms},...)?"
                return False

            def r(self, expr: Expr) -> Expr:
                return expr

        rhs2_: Basic = do_isub(rhs2)
        assert isinstance(rhs2_, Expr)
        rhs2_ = divs.apply(rhs2_)
        assert isinstance(rhs2_, Expr)
        rhs2 = rhs2_
        fb = FindBad(self)
        do_replace(rhs2, fb.m, fb.r)
        if fb.msg is not None:
            print(self.thorn_def.subs)
            raise Exception(fb.msg)
        assert not lhs2.is_Number, f"The left hand side of an equation can't be a number: '{lhs2}'"
        self._eqn_list.add_eqn(lhs2, rhs2)
        print(colorize("Add eqn:", "green"), lhs2, colorize("->", "cyan"), rhs2)

    def get_free_indices(self, expr: Expr) -> OrderedSet[Idx]:
        it = check_indices(expr, self.thorn_def.defn)
        return it.free

    def split_loop(self) -> None:
        if self.been_baked:
            raise DslException("Cannot split loop because the EqnComplex has already been baked.")
        self.eqn_complex.new_eqn_list()

    @multimethod
    def add_eqn(self, lhs: Indexed, rhs: Expr) -> None:
        check_indices(rhs, self.thorn_def.defn)

        if self.been_baked:
            raise DslException("add_eqn should not be called on a baked ThornFunction")

        lhs2: Symbol
        if self.get_free_indices(lhs) != self.get_free_indices(rhs):
            raise DslException(f"Free indices of '{lhs}' and '{rhs}' do not match.")
        count = 0
        for tup in expand_free_indices(lhs, self.thorn_def.symmetries):
            count += 1
            lhs_x, idxs, _ = tup
            lhs2_: Basic = do_isub(lhs_x, self.thorn_def.subs)
            if not isinstance(lhs2_, Symbol):
                mms = mk_mk_subst(repr(lhs2_))
                raise Exception(f"'{lhs2_}' does not evaluate a Symbol. Did you forget to call mk_subst({mms},...)?")
            lhs2 = lhs2_
            rhs0 = rhs
            rhs2 = self.thorn_def.do_subs(rhs0, idxs)
            self._add_eqn2(lhs2, rhs2)
        if count == 0:
            # TODO: Understand what's going on with arg 0
            for ind in lhs.args[1:]:
                assert isinstance(ind, Idx)
                assert is_numeric_index(ind)
            lhs2 = cast(Symbol, self.thorn_def.do_subs(lhs))
            rhs2 = self.thorn_def.do_subs(rhs)
            self._add_eqn2(lhs2, rhs2)

    @add_eqn.register
    def _(self, lhs: IndexedBase, rhs: Expr) -> None:

        if self.been_baked:
            raise Exception("add_eqn should not be called on a baked ThornFunction")

        lhs2 = cast(Symbol, self.thorn_def.do_subs(lhs))
        eci = expand_contracted_indices(rhs, self.thorn_def.symmetries)
        rhs2 = do_isub(eci)
        self._add_eqn2(lhs2, rhs2)

    @add_eqn.register
    def _(self, lhs: Indexed, rhs: Matrix) -> None:

        if self.been_baked:
            raise DslException("add_eqn should not be called on a baked ThornFunction")

        count = 0
        for tup in expand_free_indices(lhs, self.thorn_def.symmetries):
            count += 1
            lhs_x, idxs, _ = tup
            lhs2_ = do_isub(lhs_x, self.thorn_def.subs)
            lhs2 = lhs2_
            arr_idxs = to_num_tup(lhs.args[1:], idxs)
            rhs0 = rhs[arr_idxs]
            rhs2 = self.thorn_def.do_subs(rhs0, idxs)
            assert isinstance(lhs2, Symbol)
            self._add_eqn2(lhs2, rhs2)
        assert count > 0

    @add_eqn.register
    def _(self, lhs: IndexedBase, rhs: Expr) -> None:
        var = lhs.args[0]
        assert isinstance(var, Symbol)
        self._add_eqn2(var, rhs)

    @add_eqn.register
    def _(self, lhs: Indexed, rhs: List[Expr]) -> None:

        if self.been_baked:
            raise Exception("add_eqn should not be called on a baked ThornFunction")

        count = 0
        for tup in expand_free_indices(lhs, self.thorn_def.symmetries):
            count += 1
            lhs_x, idxs, idx = tup
            num_idx = to_num(idxs[idx[0]])
            lhs2_ = do_isub(lhs_x, self.thorn_def.subs)
            lhs2 = lhs2_
            rhs2 = rhs[num_idx]
            assert isinstance(lhs2, Symbol)
            self._add_eqn2(lhs2, rhs2)
        assert count > 0

    def madd(self) -> None:
        self.eqn_complex.do_madd()

    def cse(self) -> None:
        self.eqn_complex.do_cse()

    def dump(self) -> None:
        self.eqn_complex.dump()

    def eqn_bake(self) -> None:
        self.eqn_complex.bake()

    def recycle_temporaries(self) -> None:
        self.eqn_complex.recycle_temporaries()

    def split_output_eqns(self) -> None:
        self.eqn_complex.split_output_eqns()

    @staticmethod
    def _mk_default_thorn_function_bake_options() -> ThornFunctionBakeOptions:
        return {
            'do_cse': False,
            'do_madd': False,
            'do_recycle_temporaries': True,
            'do_split_output_eqns': True
        }

    def bake(self, **kwargs: Unpack[ThornFunctionBakeOptions]) -> None:
        """
        Finalize this function in preparation to be passed to a generator.
        :param do_cse: If true, perform SymPy's common subexpression elimination.
        :param do_madd: If true, attempt to generate fused multiply-add function calls where appropriate.
        :param do_recycle_temporaries: If true, attempt to conserve register use by recycling temporary variables.
        :param do_split_output_eqns: If true, split apart equations whose LHSes are output variables.
        :return:
        """
        if self.been_baked:
            raise DslException("bake should not be called more than once")
        print(f"*** {self.name} ***")

        options = self._mk_default_thorn_function_bake_options()
        options.update(kwargs)

        # Doing a first pass of complexity analysis for CSE
        for eqn_list in self.eqn_complex.eqn_lists:
            eqn_list._run_preliminary_complexity_analysis()

        if options['do_madd']:
            self.madd()
        #if do_cse:
        #    self.cse()

        self.eqn_bake()

        if options['do_split_output_eqns']:
            self.split_output_eqns()

        if options['do_recycle_temporaries']:
            self.recycle_temporaries()

        self.been_baked = True

    def show_tensortypes(self) -> None:
        keys: Set[str] = OrderedSet()
        for k1 in self.eqn_complex.inputs:
            keys.add(str(k1))
        for k2 in self.eqn_complex.outputs:
            keys.add(str(k2))
        for k in keys:
            group, indices, members = self.get_tensortype(k)
            print(colorize(k, "green"), "is a member of", colorize(group, "green"), "with indices",
                  colorize(indices, "cyan"), "and members", colorize(members, "magenta"))

    def get_tensortype(self, item: Union[str, Symbol]) -> Tuple[str, List[Idx], List[str]]:
        return self.thorn_def.get_tensortype(item)


class ThornDef:
    """
    Represents a Cactus thorn. A ThornDef object contains everything EmitCactus knows about a thorn over the course
    of evaluating a recipe. It is also an important interface for declaring variables, adding new thorn functions,
    and more.
    """

    # noinspection SpellCheckingInspection
    _xyz_subst_thorns: list[str] = ["ADMBaseX", "TmunuBaseX", "HydroBaseX"]

    def __init__(self, arr: str, name: str, *, run_simplify: bool = True) -> None:
        self.fun_args: Dict[str, int] = dict()
        self.run_simplify = run_simplify
        self.coords: List[Symbol] = list()
        self.apply_div: Applier = ApplyDiv()
        self.arrangement = arr
        self.name = name
        self.symmetries = Sym()
        self.base2group: Dict[str, str] = dict()
        self.gfs: Dict[str, IndexedBase] = dict()
        self.subs: Dict[Indexed|IndexedBase, Expr] = dict()
        self.params: Dict[str, Param] = dict()
        self.var2base: Dict[str, str] = dict()
        self.groups: Dict[str, List[str]] = dict()
        self.props: Dict[str, List[Integer]] = dict()
        self.defn: Dict[str, Tuple[str, List[Idx]]] = dict()
        self.centering: Dict[str, Optional[Centering]] = dict()
        self.thorn_functions: Dict[str, ThornFunction] = dict()
        self.rhs: Dict[str, Symbol] = dict()
        self.temp: OrderedSet[str] = OrderedSet()
        self.base2thorn: Dict[str, str] = dict()
        self.base2parity: Dict[str, TensorParity] = dict()
        self.is_stencil: Dict[UFunc, bool] = {
            mkFunction("muladd"): False,
            mkFunction("stencil"): True
        }
        self.funs1: Dict[Tuple[UFunc, Idx], Expr] = dict()
        self.funs2: Dict[Tuple[UFunc, Idx, Idx], Expr] = dict()
        self.div_makers: Dict[str, DivMakerVisitor] = dict()
        self.set_derivative_stencil(5)
        self.div_makers["div"] = DivMakerVisitor(div)
        self.div_makers["D"] = DivMakerVisitor(D)
        self.global_temporaries: OrderedSet[Symbol] = OrderedSet()
        for dmv in self.div_makers.values():
            dmv.params = self.mk_param_set()

    def _grid_variables(self) -> set[Symbol]:
        gv: set[Symbol] = set()
        for tf in self.thorn_functions.values():
            gv |= tf.eqn_complex._grid_variables()
        return gv

    def do_global_cse(self, promotion_strategy: TemporaryPromotionStrategy = promote_all()) -> None:
        TfName = NewType("TfName", str)
        LocalElIdx = NewType("LocalElIdx", int)

        tf_names: list[TfName] = sorted([TfName(name) for name in self.thorn_functions.keys()])
        old_tf_shapes: OrderedDict[TfName, list[int]] = OrderedDict()
        old_tf_lhses: OrderedDict[TfName, list[Symbol]] = OrderedDict()
        old_tf_rhses: OrderedDict[TfName, list[Expr]] = OrderedDict()

        for tf_name in tf_names:
            old_tf_shapes[tf_name] = list()
            old_tf_lhses[tf_name] = list()
            old_tf_rhses[tf_name] = list()

        for tf_name, tf in sorted([(TfName(name), tf) for name, tf in self.thorn_functions.items()], key=lambda kv: tf_names.index(kv[0])):
            for eqn_list in tf.eqn_complex.eqn_lists:
                old_tf_shapes[tf_name].append(len(eqn_list.eqns))
                for lhs, rhs in sorted(eqn_list.eqns.items(), key=lambda kv: eqn_list.order.index(kv[0])):
                    old_tf_lhses[tf_name].append(lhs)
                    old_tf_rhses[tf_name].append(rhs)

        substitutions_list: list[tuple[Symbol, Expr]]
        new_rhses: list[Expr]
        substitutions_list, new_rhses = cse(list(chain(*old_tf_rhses.values())))

        substitutions = {lhs: rhs for lhs, rhs in substitutions_list}
        substitutions_order = {lhs: idx for idx, (lhs, _) in enumerate(substitutions_list)}

        new_temp_direct_reads: dict[Symbol, dict[TfName, set[LocalElIdx]]] = {sym: dict() for sym in substitutions.keys()}
        new_temp_dependencies: dict[Symbol, set[Symbol]] = {sym: set() for sym in substitutions.keys()}
        new_temp_dependents: dict[Symbol, set[Symbol]] = {sym: set() for sym in substitutions.keys()}

        temp_rhs_occurrences: dict[Symbol, int] = defaultdict(int)
        for rhs in new_rhses:
            for temp in free_symbols(rhs).intersection(substitutions.keys()):
                temp_rhs_occurrences[temp] += 1

        for rhs in substitutions.values():
            for temp in free_symbols(rhs).intersection(substitutions.keys()):
                temp_rhs_occurrences[temp] += 1

        global_eqn_idx = 0
        for tf_index, tf in enumerate(sorted(self.thorn_functions.values(), key=lambda tf: tf_names.index(TfName(tf.name)))):
            tf_name = TfName(tf.name)

            for el_idx, el_shape in enumerate(old_tf_shapes[tf_name]):
                eqn_list = tf.eqn_complex.eqn_lists[el_idx]
                el_new_free_symbols: set[Symbol] = set(chain(*[free_symbols(rhs) for rhs in new_rhses[global_eqn_idx:global_eqn_idx + el_shape]]))
                new_temps = el_new_free_symbols.intersection(substitutions.keys())

                for new_temp, temp_rhs in [(new_temp, substitutions[new_temp]) for new_temp in new_temps]:
                    assert new_temp not in eqn_list.inputs
                    assert new_temp not in eqn_list.params
                    assert new_temp not in eqn_list.outputs
                    assert new_temp not in eqn_list.eqns

                    get_or_compute(new_temp_direct_reads[new_temp], tf_name, lambda _: set()).add(LocalElIdx(el_idx))

                    # Temps might be substituted for expressions which contain other temps.
                    # We need to recursively check the RHSes to ensure we compute the dependencies in the appropriate loops.
                    def drill(lhs: Symbol, rhs: Expr) -> None:
                        temp_dependencies = free_symbols(rhs).intersection(substitutions.keys())
                        assert lhs not in temp_dependencies
                        for td in temp_dependencies:
                            new_temp_dependencies[lhs].add(td)
                            new_temp_dependents[td].add(lhs)
                            drill(td, substitutions[td])

                    drill(new_temp, temp_rhs)

                for lhs in old_tf_lhses[tf_name]:
                    assert lhs in eqn_list.eqns
                    eqn_list.eqns[lhs] = new_rhses[global_eqn_idx]
                    global_eqn_idx += 1

                #global_eqn_idx += el_shape

                eqn_list.uncse()  # todo: Kept this around from the previous implementation, but do we need it anymore?

        temp_kinds: dict[Symbol, TempKind] = dict()
        tfs_reading_direct: dict[Symbol, dict[ThornFunction, set[LocalElIdx]]] = defaultdict(lambda: dict())
        tfs_active_reads: dict[Symbol, dict[ThornFunction, set[LocalElIdx]]] = defaultdict(lambda: dict())
        synthetic_global_dependents: dict[Symbol, set[Symbol]] = defaultdict(set)

        for new_temp, new_rhs in substitutions.items():
            tf_names_reading_direct = set(new_temp_direct_reads[new_temp].keys())

            for tf_name in tf_names_reading_direct:
                els_reading_direct = new_temp_direct_reads[new_temp].get(tf_name, set())
                tfs_reading_direct[new_temp][self.thorn_functions[tf_name]] = els_reading_direct

        complexities: dict[Symbol, int] = dict()
        for tf in self.thorn_functions.values():
            for eqn_list in tf.eqn_complex.eqn_lists:
                complexities.update(eqn_list.complexity)

        complexity_visitor = SympyComplexityVisitor(
            lambda s: s in self._grid_variables() #or temp_kinds.get(s, None) == TempKind.Global
        )
        for new_temp, new_rhs in substitutions.items():
            complexities[new_temp] = complexity_visitor.complexity(new_rhs)

        promotion_predicate = promotion_strategy(complexities)

        for new_temp, new_rhs in sorted(substitutions.items(),
                                        key=lambda kv: substitutions_order[kv[0]],
                                        reverse=True):
            tfs_active_reads[new_temp].update(tfs_reading_direct[new_temp])

            for td in new_temp_dependents[new_temp]:
                if temp_kinds.get(td, None) == TempKind.Global:
                    synthetic_global_dependents[new_temp].add(td)
                else:
                    if len(synthetic_global_dependents[td]) > 0:
                        synthetic_global_dependents[new_temp].update(synthetic_global_dependents[td])
                    for transitive_read_tf, transitive_read_els in tfs_active_reads.get(td, dict()).items():
                        get_or_compute(tfs_active_reads[new_temp], transitive_read_tf, lambda _: set()).update(transitive_read_els)

            assert len(synthetic_global_dependents[new_temp]) + len(tfs_active_reads[new_temp]) > 0, f"Temporary {new_temp} has 0 active reads"

            if len(synthetic_global_dependents[new_temp]) + len(tfs_active_reads[new_temp]) > 1:
                temp_kinds[new_temp] = TempKind.Global
            elif len(synthetic_global_dependents[new_temp]) == 1:
                temp_kinds[new_temp] = TempKind.Local
            else:
                assert len(tfs_active_reads[new_temp]) == 1
                if len(list(tfs_active_reads[new_temp].values())[0]) > 1:
                    temp_kinds[new_temp] = TempKind.Tile
                else:
                    temp_kinds[new_temp] = TempKind.Local

            assert new_temp in temp_kinds
            temp_kinds[new_temp] = temp_kinds[new_temp].clamp(promotion_predicate(new_temp))

        checked_deps: set[Symbol] = set()
        def compute_centerings(temp: Symbol) -> None:
            if temp in checked_deps:
                return

            checked_deps.add(temp)

            for td in new_temp_dependencies[temp]:
                compute_centerings(td)

            centerings = {
                c for c in {
                    self.centering.get(self.var2base.get(str(sym)) or str(sym)) for sym in free_symbols(substitutions[temp])
                } if c is not None
            }

            if len(centerings) == 0:
                #raise DslException(f"Could not infer a centering for temp {temp} -> {substitutions[temp]}; none of its dependencies have centerings")
                #todo: Cases where a temp has no grid functions in its RHS might require us to check its dependents
                print(f"Warning: Could not infer a centering for temp {temp} -> {substitutions[temp]}; none of its dependencies have centerings. Defaulting to VVV.")
                centerings = {Centering.VVV}
            elif len(centerings) > 1:
                raise DslException(f"Could not infer a centering for temp {temp} -> {substitutions[temp]}; its dependencies have conflicting centerings {centerings}")

            assert len(centerings) == 1
            self.centering[str(temp)] = centerings.pop()

        for new_temp in substitutions.keys():
            compute_centerings(new_temp)

        schedule_blocks: dict[Identifier, ScheduleBlock] = dict()
        schedule_bin_targets: dict[Symbol, dict[ScheduleBin, set[ThornFunction]]] = defaultdict(lambda: defaultdict(set))
        schedule_block_targets: dict[Symbol, dict[Identifier, set[ThornFunction]]] = defaultdict(lambda: defaultdict(set))

        for new_temp in substitutions.keys():
            print(colorize("Temporary:", "cyan"), new_temp, colorize(f"[kind = {temp_kinds.get(new_temp, TempKind.Inline)}]", "magenta"))

        for new_temp, new_rhs in substitutions.items():
            if new_temp not in temp_kinds:
                continue
            elif temp_kinds[new_temp] == TempKind.Local:
                for tf, els_reading in tfs_active_reads[new_temp].items():
                    ec = tf.eqn_complex
                    primary_el = ec.eqn_lists[primary_idx := min(els_reading)]

                    primary_el.add_eqn(new_temp, substitutions[new_temp])
                    primary_el.temporaries.add(new_temp)
            elif temp_kinds[new_temp] == TempKind.Tile:
                for tf, els_reading in tfs_active_reads[new_temp].items():
                    ec = tf.eqn_complex
                    primary_el = ec.eqn_lists[primary_idx := min(els_reading)]

                    primary_el.add_eqn(new_temp, substitutions[new_temp])
                    ec._tile_temporaries.add(new_temp)
                    primary_el.uninitialized_tile_temporaries.add(new_temp)
                    for eqn_list in [ec.eqn_lists[el_idx] for el_idx in els_reading if el_idx != primary_idx]:
                        eqn_list.uninitialized_tile_temporaries.add(new_temp)
            else:  # TempKind.Global
                self._add_symbol(new_temp, centering=self.centering[str(new_temp)])
                self.global_temporaries.add(new_temp)

                for tf in tfs_active_reads[new_temp]:
                    if isinstance(tf.schedule_target, ScheduleBlock):
                        name = tf.schedule_target.name
                        if name in schedule_blocks:
                            assert schedule_blocks[name] == tf.schedule_target
                        schedule_blocks[name] = tf.schedule_target
                        schedule_block_targets[new_temp][name].add(tf)
                    else:
                        schedule_bin_targets[new_temp][tf.schedule_target].add(tf)

        for new_temp in substitutions.keys():
            if temp_kinds.get(new_temp, None) != TempKind.Global:
                continue


            def mk_synthetic_fn(schedule_target: ScheduleTarget,
                                schedule_before: Collection[str],
                                schedule_after: Collection[str]) -> ThornFunction:
                synthetic_fn = self.create_function(
                    f'synthetic_compute_{new_temp}_{safe_name(schedule_target)}',
                    schedule_target,
                    schedule_before=schedule_before,
                    schedule_after=schedule_after
                )
                synthetic_fn._add_eqn2(new_temp, substitutions[new_temp])

                def add_deps(temp: Symbol) -> None:
                    for td in new_temp_dependencies[temp]:
                        if temp_kinds.get(td, None) != TempKind.Global:
                            if td not in synthetic_fn._eqn_list.eqns:
                                synthetic_fn._add_eqn2(td, substitutions[td])
                            add_deps(td)

                add_deps(new_temp)

                synthetic_fn.bake(do_cse=False, do_madd=False, do_recycle_temporaries=False, do_split_output_eqns=False)
                return synthetic_fn

            # Rancid hack: In CarpetX, Evolve DOES NOT run on step 0, while Analysis DOES. This breaks global temps
            #  if they happen to be initialized in Evolve then read in Analysis. To get around this, we will use
            #  PostInit to initialize any synthetic temps that are read in Analysis.
            for dic in [dic for dic in schedule_bin_targets.values() if ScheduleBin.Analysis in dic]:
                dic[ScheduleBin.PostInit].update(set())  # Just touch the set so defaultdict initializes it

            for bin in ScheduleBin._schedule_synthetic_fns(schedule_bin_targets[new_temp].keys()):
                tfs = schedule_bin_targets[new_temp][bin]
                schedule_after = sorted(list(chain(*[[f'synthetic_compute_{td}_{safe_name(bin)}' for dep_bin in schedule_bin_targets[td].keys() if bin == dep_bin] for td in new_temp_dependencies[new_temp] if temp_kinds.get(td, None) == TempKind.Global])))
                if bin is ScheduleBin.PostInit:
                    schedule_after.append('ODESolvers_PostStep')  # Hack to ensure AMR and synchronization happen first
                mk_synthetic_fn(bin, sorted([tf.name for tf in tfs]), schedule_after)

            if len(schedule_block_targets) > 0:
                print(f'Warning: Global temporary {new_temp} is accessed in at least one custom schedule block,'
                      f' on which EmitCactus cannot perform schedule analysis. The temporary will be recomputed for each'
                      f' custom block, perhaps redundantly.')

            for block, tfs in [(schedule_blocks[id], tfs) for id, tfs in schedule_block_targets[new_temp].items()]:
                schedule_after = sorted(list(chain(*[[f'synthetic_compute_{td}_{safe_name(block)}' for dep_block_name in schedule_block_targets[new_temp].keys() if block.name == dep_block_name] for td in new_temp_dependencies[new_temp] if temp_kinds.get(td, None) == TempKind.Global])))
                mk_synthetic_fn(block, sorted([tf.name for tf in tfs]), schedule_after)

        for tf in self.thorn_functions.values():
            for idx, eqn_list in enumerate(tf.eqn_complex.eqn_lists):
                print(f'*** Rebaking {tf.name} loop {idx} after do_global_cse ***')
                eqn_list.bake(force_rebake=True)
                eqn_list.dump()


    def get_free_indices(self, expr: Expr) -> OrderedSet[Idx]:
        it = check_indices(expr, self.defn)
        return it.free

    def set_derivative_stencil(self, n: int) -> None:
        assert n % 2 == 1, "n must be odd"
        assert n > 1, "n must be > 1"
        self.apply_div = ApplyDivN(n, self.funs1, self.funs2, self.fun_args)

    def get_tensortype(self, item: Union[str, Symbol]) -> Tuple[str, List[Idx], List[str]]:
        k = str(item)
        assert k in self.gfs.keys(), f"Not a defined symbol {item}"
        v = self.var2base.get(k, None)
        if v is None:
            return "none", list(), list()  # scalar
        return v, self.defn[v][1], self.groups[v]

    def create_function(self,
                        name: str,
                        schedule_target: ScheduleTarget,
                        *,
                        schedule_before: Optional[Collection[str]] = None,
                        schedule_after: Optional[Collection[str]] = None) -> ThornFunction:
        tf = ThornFunction(name, schedule_target, self, schedule_before, schedule_after)
        self.thorn_functions[name] = tf
        return tf

    def add_param(self, name: str, default: ParamDefaultType, desc: str, values: ParamValuesType = None) -> Symbol:
        self.params[name] = Param(name, default, desc, values)
        return mkSymbol(name)

    def _add_sym(self, tens: Indexed, ix1: Idx, ix2: Idx, sgn: int = 1) -> None:
        i1 = -1
        i2 = -1
        for i in range(1, len(tens.args)):
            if tens.args[i] == ix1:
                i1 = i - 1
            if tens.args[i] == ix2:
                i2 = i - 1
        assert i1 != -1, f"Index {ix1} not in {tens}"
        assert i2 != -2, f"Index {ix2} not in {tens}"
        assert i1 != i2, f"Index {ix1} cannot be symmetric with itself in {tens}"
        if i1 > i2:
            i1, i2 = i2, i1
        self.symmetries.add(tens.base, i1, i2, sgn)

    def decl_fun(self, fn_name: str, args: int = 1, is_stencil: bool = False) -> UFunc:
        fun = mkFunction(fn_name)
        self.fun_args[fn_name] = args
        self.is_stencil[fun] = is_stencil

        return fun

    def _decl_scalar(self, basename: str) -> Symbol:
        ret = mkIndexedBase(basename, tuple())
        self.gfs[basename] = ret
        self.defn[basename] = (basename, list())

        base = ret.args[0]
        assert isinstance(base, Symbol)
        return base

    def mk_coords(self, with_time: bool = False) -> List[Symbol]:
        # Note that x, y, and z are special symbols
        if dimension == 3:
            if with_time:
                self.coords = [self._decl_scalar("t"), self._decl_scalar("x"), self._decl_scalar("y"),
                               self._decl_scalar("z")]
            else:
                self.coords = [self._decl_scalar("x"), self._decl_scalar("y"), self._decl_scalar("z")]
        elif dimension == 4:
            # TODO: No idea whether this works
            self.coords = [self._decl_scalar("t"), self._decl_scalar("x"), self._decl_scalar("y"),
                           self._decl_scalar("z")]
        else:
            assert False
        return self.coords

    def mk_param_set(self) -> Set[Symbol]:
        ret: Set[Symbol] = set()
        for k in self.params:
            ret.add(mkSymbol(k))
        return ret

    def do_div(self, expr: Expr) -> Expr:
        params = self.mk_param_set()
        r = expr
        for k, v in self.div_makers.items():
            v.params = params
            r = v.visit(r, no_idx)
        return r

    @multimethod
    def mk_stencil(self, func_name: str, idx: Idx, expr: Expr) -> UFunc:
        result = self.mk_stencil(func_name, expr, [idx])
        assert isinstance(result, UFunc)
        self.div_makers[func_name] = DivMakerVisitor(result)
        return result

    @mk_stencil.register
    def _(self, func_name: str, idx_a1: Idx, idx_a2: Idx, expr_a: Expr,
          idx_b1: Idx, idx_b2: Idx, expr_b: Expr) -> UFunc:
        self.mk_stencil(func_name, idx_a1, idx_a2, expr_a)
        result = self.mk_stencil(func_name, idx_b1, idx_b2, expr_b)
        assert isinstance(result, UFunc)
        return result

    @mk_stencil.register
    def _(self, func_name: str, idx1: Idx, idx2: Idx, expr: Expr) -> UFunc:
        result = self.mk_stencil(func_name, expr, [idx1, idx2])
        assert isinstance(result, UFunc)
        self.div_makers[func_name] = DivMakerVisitor(result)
        return result

    @mk_stencil.register
    def _(self, func_name: str, expr: Expr, idx_list: List[Idx]) -> UFunc:

        @multimethod
        def mk_sten(idx_map: Dict[Idx, Idx], expr: sy.Function) -> Expr:
            # TODO: Rewrite so it does not require 3 dimensions
            assert get_dimension() == 3
            if expr.func == stencil:
                if len(expr.args) != 1:
                    raise DslException(expr)
                arg = mk_sten(idx_map, expr.args[0])
                c0 = coef(l0, arg)
                c1 = coef(l1, arg)
                c2 = coef(l2, arg)
                ret = stencil(dummy, c0, c1, c2)
                assert isinstance(ret, Expr)
                return ret
            elif expr.func == DD:
                if len(expr.args) != 1:
                    raise DslException(expr)
                arg = mk_sten(idx_map, expr.args[0])
                if arg == l0:
                    return DX
                elif arg == l1:
                    return DY
                elif arg == l2:
                    return DZ
                assert False
            elif expr.func == DDI:
                if len(expr.args) != 1:
                    raise DslException(expr)
                arg = mk_sten(idx_map, expr.args[0])
                if arg == l0:
                    return DXI
                elif arg == l1:
                    return DYI
                elif arg == l2:
                    return DZI
                assert False
            elif expr.func == noop:
                arg = mk_sten(idx_map, expr.args[0])
                retv : Expr = noop(arg)
                return retv
            else:
                raise DslException("Bad Func")

        @mk_sten.register
        def _(_idx_map: Dict[Idx, Idx], expr: sy.Float) -> Expr:
            return expr

        @mk_sten.register
        def _(_idx_map: Dict[Idx, Idx], expr: sy.Integer) -> Expr:
            return expr

        @mk_sten.register
        def _(_idx_map: Dict[Idx, Idx], expr: sy.Rational) -> Expr:
            return expr

        @mk_sten.register
        def _(idx_map: Dict[Idx, Idx], expr: sy.Pow) -> Expr:
            result: Expr = Pow(mk_sten(idx_map, expr.args[0]), expr.args[1])
            return result

        @mk_sten.register
        def _(idx_map: Dict[Idx, Idx], expr: Idx) -> Expr:
            retval = idx_map.get(expr, expr)
            return retval

        @mk_sten.register
        def _(idx_map: Dict[Idx, Idx], expr: sy.Add) -> Expr:
            ret = zero
            for a in expr.args:
                term = mk_sten(idx_map, a)
                ret += term
            return ret

        @mk_sten.register
        def _(idx_map: Dict[Idx, Idx], expr: sy.Mul) -> Expr:
            ret = one
            for a in expr.args:
                term = mk_sten(idx_map, a)
                ret *= term
            return ret

        func = mkFunction(func_name)

        if len(idx_list) == 1 or (len(idx_list) == 2 and idx_list[0] == idx_list[1]):
            idx = idx_list[0]
            is_down_idx = is_down(idx)
            for i in range(get_dimension()):
                if is_down_idx:
                    idx0 = down_indices[i]
                else:
                    idx0 = up_indices[i]
                result = mk_sten({idx: idx0}, expr)
                self.funs1[(func, idx0)] = result
        elif len(idx_list) == 2:
            idx1 = idx_list[0]
            idx2 = idx_list[1]
            is_down_idx1 = is_down(idx1)
            is_down_idx2 = is_down(idx2)
            for i in range(get_dimension()):
                if is_down_idx1:
                    idx10 = down_indices[i]
                else:
                    idx10 = up_indices[i]
                for j in range(get_dimension()):
                    if i == j:
                        continue
                    if is_down_idx2:
                        idx20 = down_indices[j]
                    else:
                        idx20 = up_indices[j]
                    result = mk_sten({idx1: idx10, idx2: idx20}, expr)
                    self.funs2[(func, idx10, idx20)] = result

        return func

    class DeclOptionalArgs(TypedDict, total=False):
        centering: Centering
        declare_as_temp: bool
        rhs: IndexedBase
        from_thorn: str
        parity: TensorParity
        group_name: str
        symmetries: List[Tuple[Idx, Idx]]
        anti_symmetries: List[Tuple[Idx, Idx]]
        substitution_rule: MkSubstType | None

    def get_state(self) -> List[IndexedBase]:
        return [self.gfs[k] for k in self.rhs]

    # noinspection PyIncorrectDocstring
    def decl(self, basename: str, indices: List[Idx], **kwargs: Unpack[DeclOptionalArgs]) -> IndexedBase:
        """
        Declares a new scalar or tensor variable.

        :param basename: The symbolic name of the variable.
        :param indices: The indices of the variable. If the variable is a scalar, this should be an empty list.
        :param rhs: Specifies the right-hand side of an implied PDE with d(the_var)/dt on the left.
                    Setting this argument implies that the variable to be declared is a state variable.
        :param centering: The centering of the variable. Defaults to VVV.
        :param group_name: Override the Cactus group name this variable (or its components) will be declared under.
        :param from_thorn: Specifies the thorn wherein this variable is declared. If this argument is present,
                           EmitCactus will not produce any declarations for the variable in the current thorn.
        :param declare_as_temp: If true, the variable will be declared as a temporary variable, i.e., one that only
                                exists in source code. EmitCactus will not produce any thorn-level declarations for
                                temporary variables.
        :param parity: Specifies the variable's reflectional symmetries.
        :param symmetries: Specifies the permutations of the variable's indices which are symmetric with the
                           canonical ordering given in the `indices` argument.
        :param anti_symmetries: Specifies the permutations of the variable's indices which are anti-symmetric
                                with the canonical ordering given in the `indices` argument.
        :param substitution_rule: Specifies the base substitution rule for the variable. If this argument is absent,
                                  a default substitution rule is applied. Pass `None` to suppress the default rule.
                                  The default substitution rule is determined as follows:
                                  1) If the variable is a scalar, the substitution rule is the identity function.
                                  2) If the variable is a tensor with `from_thorn` set to one of the thorns in
                                     `_xyz_subst_thorns`, then the substitution rule is `subst_tensor_xyz`.
                                  3) Otherwise, the substitution rule is `subst_tensor`.

        :return: A symbolic `IndexedBase` object which represents the declared variable.
        :raises DslException: If symmetries or anti-symmetries are applied to a scalar variable.
        """
        if (rhs := kwargs.get('rhs', None)) is not None:
            base_sym = rhs.args[0]
            assert isinstance(base_sym, Symbol)
            self.rhs[basename] = base_sym

        if (centering := kwargs.get('centering', None)) is None:
            centering = Centering.VVV

        the_symbol = mkIndexedBase(basename, shape=tuple([dimension] * len(indices)))

        if len(indices) != 0:
            indexed_symbol = mkIndexed(the_symbol, *tuple(indices))
        else:
            indexed_symbol = None

        assert basename not in self.gfs
        self.gfs[basename] = the_symbol
        self.defn[basename] = (basename, list(indices))
        self.centering[basename] = centering
        self.base2group[basename] = kwargs.get('group_name', basename)

        if (from_thorn := kwargs.get('from_thorn', None)) is not None:
            self.base2thorn[basename] = from_thorn

        if kwargs.get('declare_as_temp', False):
            self.temp.add(basename)

        if (parity := kwargs.get('parity', None)) is not None:
            self.base2parity[basename] = parity

        if (symmetries := kwargs.get('symmetries', None)) is not None:
            if indexed_symbol is None:
                raise DslException('Symmetries cannot be applied to a scalar variable')

            for sym in symmetries:
                self._add_sym(indexed_symbol, *sym, sgn=1)

        if (anti_symmetries := kwargs.get('anti_symmetries', None)) is not None:
            if indexed_symbol is None:
                raise DslException('Anti-symmetries cannot be applied to a scalar variable')

            for a_sym in anti_symmetries:
                self._add_sym(indexed_symbol, *a_sym, sgn=-1)

        if indexed_symbol is not None:
            default_subst = subst_tensor_xyz if from_thorn in self._xyz_subst_thorns else subst_tensor

            if (substitution_rule := kwargs.get('substitution_rule', default_subst)) is not None:
                self.add_substitution_rule(indexed_symbol, substitution_rule)

        return the_symbol

    def _add_symbol(self, the_symbol: Symbol, centering: Optional[Centering]) -> None:
        basename = str(the_symbol)

        assert basename not in self.gfs
        self.gfs[basename] = mkIndexedBase(basename, shape=())
        self.defn[basename] = (basename, list())
        self.centering[basename] = centering
        self.base2group[basename] = basename

    def find_indices(self, foo: Basic) -> List[Idx]:
        ret: List[Idx] = list()
        if type(foo) in [div, D]:
            ret = self.find_indices(foo.args[0])
        for arg in foo.args[1:]:
            assert isinstance(arg, Idx)
            ret.append(arg)
        return ret

    def find_symmetries(self, foo: Basic) -> List[Tuple[int, int, int]]:
        m_sym_list: List[Tuple[int, int, int]] = list()
        # noinspection PyUnresolvedReferences
        if foo.is_Function and hasattr(foo, "name") and foo.name in ["div", "D"]:
            # This is a derivative
            if len(foo.args) == 3:
                # This is a 2nd derivative, symmetric in the last 2 args
                foo_arg1 = len(foo.args[0].args) - 1
                foo_arg2 = foo_arg1 + 1
                m_sym: Tuple[int, int, int] = (foo_arg1, foo_arg2, 1)
                m_sym_list += [m_sym]
                m_sym_list += self.find_symmetries(foo.args[0])
            elif len(foo.args) == 2:
                m_sym_list += self.find_symmetries(foo.args[0])
            else:
                assert False, "Only handle 1st and 2nd derivatives"
        elif isinstance(foo, Indexed):
            k = foo.base
            return self.symmetries.sd.get(k, list())
        return m_sym_list

    def get_matrix(self, ind: Indexed) -> Matrix:
        values: Dict[Idx, Idx] = dict()
        result = mkZeros(*tuple([dimension] * (len(ind.args) - 1)))
        ind_args: List[Idx] = [checked_cast(x, Idx) for x in ind.args[1:]]
        while incr(ind_args, values):
            arr_idxs = tuple([to_num(checked_cast(do_subs(x, values), Idx)) for x in ind_args])
            r = self.do_subs(ind, idx_subs=values)
            result[arr_idxs] = r
        return result

    @staticmethod
    def get_indices(expr: Expr) -> List[Idx]:
        out: List[Idx] = list()
        if type(expr) in [div, D]:
            for arg in expr.args[0].args[1:]:
                assert isinstance(arg, Idx)
                out.append(arg)
        assert isinstance(expr, Indexed)
        for arg in expr.args[1:]:
            assert isinstance(arg, Idx)
            out.append(arg)
        return out

    def get_coords(self) -> List[Symbol]:
        return self.coords

    def get_params(self) -> Set[str]:
        return OrderedSet(self.params)

    @multimethod
    def add_substitution_rule(self, indexed: Indexed, f: Callable[[Indexed, int, int], Expr]) -> None:
        def f2(ix: Indexed, *n: int) -> Expr:
            return f(ix, n[0], n[1])

        self.add_substitution_rule(indexed, f2)

    @add_substitution_rule.register
    def _(self, indexed: Indexed, f: Callable[[Indexed, int], Expr]) -> None:
        def f2(ix: Indexed, *n: int) -> Expr:
            return f(ix, n[0])

        self.add_substitution_rule(indexed, f2)

    @add_substitution_rule.register
    def _(self, indexed: Indexed, f: Callable[[Indexed, int, int, int], Expr]) -> None:
        def f2(ix: Indexed, *n: int) -> Expr:
            return f(ix, n[0], n[1], n[2])

        self.add_substitution_rule(indexed, f2)

    @add_substitution_rule.register
    def _(self, indexed: Indexed, f: BaseIndexedSubstFnType = subst_tensor) -> None:
        indices: List[Idx]
        iter_var = indexed

        for tup in expand_free_indices(iter_var, self.symmetries):
            indexed_sym, ind_rep, _ = tup

            assert isinstance(indexed_sym, Indexed)

            idxs = indexed_sym.indices
            sub_val_ = f(indexed_sym, *idxs)

            if sub_val_.is_Number:
                pass
            elif sub_val_.is_Function:
                pass
            else:
                sub_val = str(sub_val_)
                out_str = str(indexed_sym.base)
                assert isinstance(sub_val_, Symbol)
                self.gfs[sub_val] = mkIndexedBase(sub_val, tuple())
                self.centering[sub_val] = self.centering[out_str]
                self.var2base[sub_val] = out_str
                if out_str not in self.groups:
                    self.groups[out_str] = list()
                members = self.groups[out_str]
                members.append(sub_val)
                print(colorize(indexed_sym, "red"), colorize("->", "magenta"), colorize(sub_val, "cyan"))
                self.subs[indexed_sym] = sub_val_

    @add_substitution_rule.register
    def _(self, indexedBase: IndexedBase, f: Expr) -> None:
        rhs = simplify(self.do_subs(f, idx_subs={}))
        self.subs[indexedBase] = rhs
        print(colorize(indexedBase,"cyan"), colorize("->","green"), colorize(rhs,"yellow"))

    @add_substitution_rule.register
    def _(self, indexed: Indexed, f: Expr) -> None:
        indices: List[Idx]
        iter_var = indexed

        if self.get_free_indices(iter_var) != self.get_free_indices(f):
            raise Exception(f"Free indices of '{indexed}' and '{f}' do not match.")
        for tup in expand_free_indices(iter_var, self.symmetries):
            indexed_sym, ind_rep, _ = tup
            assert isinstance(indexed_sym, Indexed)
            if self.run_simplify:
                self.subs[indexed_sym] = simplify(self.do_subs(f, idx_subs=ind_rep))
            else:
                self.subs[indexed_sym] = self.do_subs(f, idx_subs=ind_rep)
            print(colorize(indexed_sym, "red"), colorize("->", "magenta"), colorize(self.subs[indexed_sym], "cyan"))
        return None

    @add_substitution_rule.register
    def _(self, indexed: Indexed, f: ImmutableDenseMatrix) -> None:
        self._mk_subst_matrix(indexed, f)

    @add_substitution_rule.register
    def _(self, indexed: Indexed, f: MatrixBase) -> None:
        self._mk_subst_matrix(indexed, f)

    def _mk_subst_matrix(self, indexed: Indexed, f: MatrixBase) -> None:
        indices: List[Idx]
        iter_var = indexed

        set_matrix = f
        for tup in expand_free_indices(iter_var, self.symmetries):
            out, idx_rep, alist = tup
            assert isinstance(out, Indexed)
            arr_idxs = tuple([to_num(x) for x in out.indices])
            if self.run_simplify:
                n_array = len(arr_idxs)
                res = simplify(set_matrix[arr_idxs[0:2]])
                if n_array >= 3:
                    res = self.do_subs(res, idx_subs=idx_rep)
                self.subs[out] = res
            else:
                self.subs[out] = set_matrix[arr_idxs]
            print(colorize(out, "red"), colorize("->", "magenta"), colorize(self.subs[out], "cyan"))
        return None

    def expand_eqn(self, eqn: Eq) -> List[Eq]:
        result: List[Eq] = list()
        for tup in expand_free_indices(eqn.lhs, self.symmetries):
            lhs, idxs, _ = tup
            result += [mkEq(self.do_subs(lhs), self.do_subs(eqn.rhs, idxs))]
        return result

    def do_subs(self, arg: Expr, idx_subs: Optional[Dict[Idx, Idx]] = None) -> Expr:
        if idx_subs is None:
            idx_subs = dict()
        isub = IndexSubsVisitor(self.subs)
        arg1 = arg
        for i in range(20):
            new_arg = arg1
            new_arg = expand_contracted_indices(new_arg, self.symmetries)
            new_arg = cast(Expr, self.symmetries.apply(new_arg))

            isub.idx_subs = idx_subs
            new_arg = isub.visit(new_arg)
            new_arg = self.do_div(new_arg)
            if new_arg == arg1:
                return new_arg
            arg1 = new_arg
        raise Exception(arg)


def _parity_of(p: int | Parity) -> Parity:
    if isinstance(p, Parity):
        return p
    elif p == -1:
        return Parity.Negative
    elif p == 1:
        return Parity.Positive
    else:
        raise DslException(f"Parity must be -1 or +1")


def parities(*args: Parity | int) -> TensorParity:
    if len(args) == 0:
        raise DslException("Parities must not be empty")

    if len(args) % 3 != 0:
        raise DslException('Parities must come in groups of 3')

    parities: list[SingleIndexParity] = list()
    for i in range(0, len(args), 3):
        pars = [_parity_of(p) for p in args[i:i + 3]]
        parities.append(SingleIndexParity(*pars))

    return TensorParity(parities)
