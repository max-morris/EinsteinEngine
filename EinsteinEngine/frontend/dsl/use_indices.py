#  Copyright (C) 2024-2026 Max Morris, Steven R. Brandt, and other Einstein Engine contributors.
#
#  This file is part of the Einstein Engine (EinsteinEngine).
#
#  EinsteinEngine is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  EinsteinEngine is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Use the Sympy Indexed type for relativity expressions.
"""
import typing
from typing import *

import sympy.logic.boolalg
from multimethod import multimethod
from mypy_extensions import VarArg
# noinspection PyUnusedImports
# (MyPy needs Idx, Expr)
from sympy import Symbol, Indexed, IndexedBase, Basic, MatrixBase, ImmutableDenseMatrix, Idx, Expr
from sympy.core.relational import Relational

from EinsteinEngine.frontend.dsl.dsl_exception import DslException
from EinsteinEngine.frontend.definitions import *
from EinsteinEngine.frontend.dsl.symmetries import Sym
from EinsteinEngine.common.sympywrap import *
from EinsteinEngine.common.util import OrderedSet
from EinsteinEngine.common.util import checked_cast

__all__ = ["EinsteinNotationManager", "idx_to_int", "IndexedSubstFnType", "MkSubstType", "subst_tensor",
           "subst_tensor_xyz", "noop", "stencil", "DD", "DDI", "is_numeric_relativity_index"]

IndexPairLookup = Mapping[Idx, Idx]


class IndexedDeclaration(Protocol):
    indices: tuple[Idx, ...]


type IndexDeclarations = Mapping[str, tuple[Idx, ...] | IndexedDeclaration]

###
import sympy as sy


class InvalidIndexError(DslException):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(self.message)


class EinsteinNotationManager:
    _pair_tmp_name: str
    lookup_pair: Dict[Idx, Idx]
    dimensionality: int
    _lower_numeric_indices: tuple[Idx, ...]
    _upper_numeric_indices: tuple[Idx, ...]

    def __init__(self, dimensionality: int = 3) -> None:
        self.dimensionality = dimensionality
        self._pair_tmp_name = "A"
        self.lookup_pair = dict()
        self._lower_numeric_indices = tuple(mk_idx(f"l{i}") for i in range(dimensionality))
        self._upper_numeric_indices = tuple(mk_idx(f"u{i}") for i in range(dimensionality))

    def mk_pair(self, s: Optional[str] = None) -> tuple[Idx, Idx]:
        """
        Returns a tuple containing an upper/lower index pair.
        """

        if s is None:
            s = self._pair_tmp_name
            tmp_num = ord(self._pair_tmp_name[-1])
            if tmp_num == ord("Z"):
                self._pair_tmp_name += "A"
            else:
                self._pair_tmp_name = self._pair_tmp_name[0:-1] + chr(tmp_num + 1)
        u, l = mk_idxes(f"u{s} l{s}")
        self.lookup_pair[l] = u
        self.lookup_pair[u] = l
        return u, l

    def check_indices(
            self,
            rhs: Expr,
            declarations: IndexDeclarations
    ) -> 'IndexTracker':
        """
        Check indexed-expression validity and return free/contracted index tracking.
        """
        err = IndexContractionVisitor(declarations, dimensionality=self.dimensionality, lookup_pair=self.lookup_pair)
        ret: IndexTracker
        _, ret = err.visit(rhs)
        return ret

    def get_free_indices(
            self,
            expr: Expr,
            declarations: IndexDeclarations
    ) -> OrderedSet[Idx]:
        return self.check_indices(expr, declarations).free

    def expand_contracted_indices(self, expr: Expr, sym: Sym) -> Expr:
        viz = IndexContractionVisitor(dict(), dimensionality=self.dimensionality, lookup_pair=self.lookup_pair)
        out, _it = viz.visit(expr)
        out = sym.apply(out)
        assert isinstance(out, Expr)
        return out

    def expand_free_indices(self, expr: Expr, sym: Sym) -> List[Tuple[Expr, Dict[Idx, Idx], List[Idx]]]:
        index_list: List[Idx] = sorted(list(self.get_uncontracted_indices(expr)), key=str)
        output: List[Tuple[Expr, Dict[Idx, Idx], List[Idx]]] = list()
        expr = self.expand_contracted_indices(expr, sym)
        index_values: Dict[Idx, Idx] = dict()
        while self.incr(index_list, index_values):
            assert len(index_values) != 0, "Something very bad happened"
            if type(expr) == Indexed:
                result = do_subs(expr, index_values)
                sym_result = sym.apply(result)
                if result != sym_result:
                    continue
            out_expr = do_subs(expr, index_values, sym)
            output += [(out_expr, index_values.copy(), index_list)]
        return output

    def is_numeric_index(self, idx: Idx) -> bool:
        s = str(idx)
        assert idx in self.lookup_pair
        n = ord(s[1])
        return num0 <= n <= num9

    def is_letter_index(self, sym: Basic) -> bool:
        if type(sym) != Idx:
            return False
        s = str(sym)
        if sym not in self.lookup_pair:
            return False
        if s[0] not in ["u", "l"]:
            return False
        n = ord(s[1])
        return n < ord0 or n > ord9

    def get_indices(self, expr: Expr) -> OrderedSet[Idx]:
        ret: OrderedSet[Idx] = OrderedSet()
        for symbol in free_indexed(expr):
            if self.is_letter_index(symbol):
                ret.add(symbol)
        return ret

    def by_name(self, idx: Idx) -> str:
        s = str(idx)
        assert idx in self.lookup_pair
        return s[1:] + s[0]

    def get_contracted_indices(self, expr: Expr) -> OrderedSet[Idx]:
        indices = list(self.get_indices(expr))
        indices = sorted(indices, key=self.by_name)
        ret: OrderedSet[Idx] = OrderedSet()
        i = 0
        while i < len(indices):
            if i + 1 < len(indices) and self.is_pair(indices[i], indices[i + 1]):
                ret.add(indices[i])
                i += 2
            else:
                i += 1
        return ret

    def get_pair(self, idx: Idx) -> Tuple[Idx, Idx]:
        if self.is_lower(idx):
            return self.lookup_pair[idx], idx
        else:
            return idx, self.lookup_pair[idx]

    def is_pair(self, a: Idx, b: Idx) -> bool:
        sa = str(a)
        sb = str(b)
        assert a in self.lookup_pair
        assert b in self.lookup_pair
        if sa[1:] == sb[1:] and ((sa[0] == 'u' and sb[0] == 'l') or (sa[0] == 'l' and sb[0] == 'u')):
            return True
        else:
            return False

    def get_uncontracted_indices(self, expr: Expr) -> OrderedSet[Idx]:
        indices = list(self.get_indices(expr))
        indices = sorted(indices, key=self.by_name)
        ret: OrderedSet[Idx] = OrderedSet()
        i = 0
        while i < len(indices):
            if i + 1 < len(indices) and self.is_pair(indices[i], indices[i + 1]):
                i += 2
            else:
                ret.add(indices[i])
                i += 1
        return ret

    def incr(self, index_list: List[Idx], index_values: Dict[Idx, Idx]) -> bool:
        if len(index_list) == 0:
            return False
        ix = 0
        if len(index_values) == 0:
            for ind_ in index_list:
                u_ind, ind = self.get_pair(ind_)
                index_values[ind] = self._lower_numeric_indices[0]
                index_values[u_ind] = self._upper_numeric_indices[0]
            return True
        while True:
            if ix >= len(index_list):
                return False
            u_ind, ind = self.get_pair(index_list[ix])
            index_value = idx_to_int(index_values[ind])
            if index_value == self.dimensionality - 1:
                index_values[ind] = self._lower_numeric_indices[0]
                index_values[u_ind] = self._upper_numeric_indices[0]
                ix += 1
            else:
                index_values[ind] = self._lower_numeric_indices[index_value + 1]
                index_values[u_ind] = self._upper_numeric_indices[index_value + 1]
                break
        return True

    @staticmethod
    def is_lower(idx: Idx) -> bool:
        s = str(idx)
        return s[0] == 'l'

    @staticmethod
    def is_upper(idx: Idx) -> bool:
        s = str(idx)
        return s[0] == 'u'


class IndexTracker:
    def __init__(self, lookup_pair: IndexPairLookup) -> None:
        self.lookup_pair = lookup_pair
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
            ret.add(self.lookup_pair[a])
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
        if (idx in self.free) or (idx in self.contracted):
            return False
        # TODO: Factor this logic out elsewhere
        letter_or_num = ord(str(idx)[1])
        if ord('0') <= letter_or_num <= ord('9'):
            return True
        pdx = self.lookup_pair.get(idx, None)
        assert pdx is not None, f"{idx} not in {self.lookup_pair}"
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
    def __init__(
            self,
            declarations: IndexDeclarations,
            dimensionality: int,
            lookup_pair: IndexPairLookup
    ) -> None:
        self.declarations = declarations
        self.dimensionality = dimensionality
        self.lookup_pair = lookup_pair
        self._numeric_index_pairs: tuple[tuple[Idx, Idx], ...] = tuple(
            (mk_idx(f"l{i}"), mk_idx(f"u{i}")) for i in range(dimensionality)
        )

    def _tracker(self) -> IndexTracker:
        return IndexTracker(self.lookup_pair)

    def _decl_indices(self, basename: str) -> Optional[Tuple[Idx, ...]]:
        decl = self.declarations.get(basename, None)
        if decl is None:
            return None
        indices = decl if isinstance(decl, tuple) else decl.indices
        assert all(isinstance(idx, Idx) for idx in indices), f"Invalid index tuple for '{basename}': {indices}"
        return indices

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
            return new_expr, self._tracker()
        else:
            return new_expr, it

    def contract(self, expr: Expr, it: IndexTracker) -> Tuple[Expr, IndexTracker]:
        for lo_idx in it.contracted:
            new_expr: Expr = zero
            up_idx = self.lookup_pair[lo_idx]
            for lo_idx_val, up_idx_val in self._numeric_index_pairs:
                new_expr += do_isub(expr, dict(), {lo_idx: lo_idx_val, up_idx: up_idx_val})
            expr = new_expr
        it.used = it.contracted
        it.contracted = OrderedSet()
        return expr, it

    @visit.register
    def _(self, expr: sy.Mul) -> Tuple[Expr, IndexTracker]:
        it = self._tracker()
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
        it = self._tracker()
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
        it = self._tracker()
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
        return expr, self._tracker()

    @visit.register
    def _(self, expr: sy.Symbol) -> Tuple[Expr, IndexTracker]:
        return expr, self._tracker()

    @visit.register
    def _(self, expr: sy.Integer) -> Tuple[Expr, IndexTracker]:
        return expr, self._tracker()

    @visit.register
    def _(self, expr: sy.Rational) -> Tuple[Expr, IndexTracker]:
        return expr, self._tracker()

    @visit.register
    def _(self, expr: sy.Float) -> Tuple[Expr, IndexTracker]:
        return expr, self._tracker()

    @visit.register
    def _(self, expr: sy.Idx) -> Tuple[Expr, IndexTracker]:
        return expr, self._tracker()

    @visit.register
    def _(self, _expr: sympy.logic.boolalg.BooleanTrue) -> Tuple[Expr, IndexTracker]:
        return sympify(True), self._tracker()

    @visit.register
    def _(self, _expr: sympy.logic.boolalg.BooleanFalse) -> Tuple[Expr, IndexTracker]:
        return sympify(False), self._tracker()

    @visit.register
    def _(self, expr: sy.Indexed) -> Tuple[Expr, IndexTracker]:
        basename = str(expr.args[0])
        indices = self._decl_indices(basename)
        if indices is not None:
            if len(indices) + 1 != len(expr.args):
                raise InvalidIndexError(f"indices used on a non-indexed quantity '{expr}' in:")
        else:
            assert len(self.declarations) == 0
        it = self._tracker()
        for a in expr.args[1:]:
            _a_it = self.visit(a)
            assert isinstance(a, Idx)
            if not it.add(a):
                raise InvalidIndexError(str(expr))
        return self.contract(expr, it)

    @visit.register
    def _(self, expr: sy.Function) -> Tuple[Expr, IndexTracker]:
        it = self._tracker()
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
        return sy.Pow(*new_args), self._tracker()

    @visit.register
    def _(self, expr: sy.IndexedBase) -> Tuple[Expr, IndexTracker]:
        basename = str(expr)
        indices = self._decl_indices(basename)
        if indices is None:
            if len(self.declarations) == 0:
                n = 0
            else:
                raise InvalidIndexError(f"Undefined symbol in '{self.declarations}':")
        else:
            n = len(indices)
        if n != 0:
            if n == 1:
                msg = "1 index"
            else:
                msg = f"{n} indices"
            raise InvalidIndexError(
                f"Expression '{expr}' was declared with {msg}, but was used in this expression without indices: ")
        return expr, self._tracker()


class IndexSubsVisitor:
    def __init__(self, defn: Optional[dict[Indexed | IndexedBase, Expr]] = None, idx_subs: Optional[dict[Idx, Idx]] = None) -> None:
        self.defn = defn if defn is not None else dict()
        self.idx_subs = idx_subs if idx_subs is not None else dict()

    @multimethod
    def visit(self, expr: sy.Expr) -> Expr:
        raise DslException(f"Unexpected expression type in IndexSubsVisitor: {type(expr)}")

    @visit.register
    def _(self, expr: sy.Add) -> Expr:
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
            r = mk_indexed(expr.base, *indexes)
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
        return mk_piecewise(*args)

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


def do_isub(expr: Expr, subs: Optional[Dict[Indexed | IndexedBase, Expr]] = None,
            idx_subs: Optional[Dict[Idx, Idx]] = None) -> Expr:
    isub = IndexSubsVisitor(subs, idx_subs)
    # FIXME Why is this cast needed?
    return cast(Expr, isub.visit(expr))


def check_indices(
        rhs: Expr,
        declarations: IndexDeclarations,
        *,
        manager: EinsteinNotationManager
) -> IndexTracker:
    """
    This function not only checks the validity of indexed expressions, it returns
    all free and contracted indices.
    """

    return manager.check_indices(rhs, declarations)


###
# Need Expand Visitor
###


def is_lower_idx(ind: Idx) -> bool:
    s = str(ind)
    assert s[0] in ["u", "l"], f"ind={ind}"
    return s[0] == "l"


def idx_to_int(ind: Idx) -> int:
    s = str(ind)
    assert s[0] in ["u", "l"], f"ind={ind}"
    return int(s[1])

### dmv

TA = TypeVar("TA")


def sub_idxs(idx: Idx, values: Dict[Idx, Idx]) -> Idx:
    return checked_cast(do_subs(idx, values), Idx)


def to_num_tup_2(li: List[Idx], values: Dict[Idx, Idx]) -> Tuple[int, ...]:
    return tuple([idx_to_int(sub_idxs(x, values)) for x in li])


def to_num_tup(li: Tuple[Basic, ...], values: Dict[Idx, Idx]) -> Tuple[int, ...]:
    return to_num_tup_2([checked_cast(x, Idx) for x in li], values)


ord0 = ord('0')
ord9 = ord('9')


def is_letter_index(sym: Basic, manager: EinsteinNotationManager) -> bool:
    return manager.is_letter_index(sym)


def get_indices(xpr: Expr, manager: EinsteinNotationManager) -> OrderedSet[Idx]:
    """ Return all indices of IndexedBase objects in xpr. """
    return manager.get_indices(xpr)


def by_name(x: Idx, manager: EinsteinNotationManager) -> str:
    """ Return a string suitable for sorting a list of upper/lower indices. """
    return manager.by_name(x)


num0 = ord('0')
num9 = ord('9')


def is_numeric_relativity_index(idx: Idx, manager: Optional[EinsteinNotationManager] = None) -> bool:
    if manager is not None:
        return manager.is_numeric_index(idx)

    s = str(idx)
    n = ord(s[1])
    return s[0] in ('u', 'l') and (num0 <= n <= num9)

def is_lower(x: Idx) -> bool:
    return EinsteinNotationManager.is_lower(x)


def is_upper(x: Idx) -> bool:
    return EinsteinNotationManager.is_upper(x)


def get_pair(x: Idx, manager: EinsteinNotationManager) -> Tuple[Idx, Idx]:
    return manager.get_pair(x)


def is_pair(a: Idx, b: Idx, manager: EinsteinNotationManager) -> bool:
    return manager.is_pair(a, b)


def get_free_indices(xpr: Expr, manager: EinsteinNotationManager) -> OrderedSet[Idx]:
    """ Return all uncontracted indices in xpr. """
    return manager.get_uncontracted_indices(xpr)


def get_contracted_indices(xpr: Expr, manager: EinsteinNotationManager) -> OrderedSet[Idx]:
    """ Return all contracted indices in xpr. """
    return manager.get_contracted_indices(xpr)


def incr(
        index_list: List[Idx],
        index_values: Dict[Idx, Idx],
        manager: EinsteinNotationManager
) -> bool:
    """ Increment the indices in index_list, creating an index_values table with all possible permutations. """
    return manager.incr(index_list, index_values)


def expand_contracted_indices(
        in_expr: Expr,
        sym: Sym,
        manager: EinsteinNotationManager
) -> Expr:
    return manager.expand_contracted_indices(in_expr, sym)


def expand_free_indices(
        xpr: Expr,
        sym: Sym,
        manager: EinsteinNotationManager
) -> List[Tuple[Expr, Dict[Idx, Idx], List[Idx]]]:
    return manager.expand_free_indices(xpr, sym)


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
        base_name += str(idx_to_int(ind))

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

    return mk_symbol(_mk_name_for_tensor(sym))


def _mk_name_for_tensor_xyz(sym: Indexed, *_args: Idx) -> str:
    base_name = str(sym.base)
    for ind in sym.args[1:]:
        assert isinstance(ind, Idx)
        base_name += ["x", "y", "z"][idx_to_int(ind)]
    return base_name


def subst_tensor_xyz(sym: Indexed, *_idxs: int) -> Symbol:
    """
    Defines a symbol for a tensor using standard Cactus rules.
    Don't distinguish up/down indices. Use suffixes based on
    x, y, and z at the end.

    :param sym: The tensor expression with integer indices.

    :return: A new sympy symbol
    """
    return mk_symbol(_mk_name_for_tensor_xyz(sym))


BaseIndexedSubstFnType = Callable[[Indexed, VarArg(int)], Expr]
IndexedSubstFnType = (
        Callable[[Indexed, int], Expr] |
        Callable[[Indexed, int, int], Expr] |
        Callable[[Indexed, int, int, int], Expr] |
        BaseIndexedSubstFnType
)
MkSubstType = IndexedSubstFnType | Expr | ImmutableDenseMatrix | MatrixBase
