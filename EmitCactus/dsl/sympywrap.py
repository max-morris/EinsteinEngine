from typing import Tuple, List, Dict, Any, Union, cast, Mapping, Callable, Set, Optional
from sympy import Expr
cbrt : Callable[[Expr],Expr]
from sympy import cse as cse_, IndexedBase, Idx, Symbol, Eq, Basic, sympify, Mul, Indexed, \
    Function, Matrix, zeros, Wild, diff, simplify, sqrt, cbrt
import re
from abc import ABC, abstractmethod
from sympy.core.function import UndefinedFunction as UFunc
from EmitCactus.dsl.dsl_exception import DslException

from EmitCactus.util import OrderedSet

from multimethod import multimethod
from here import here


class Applier(ABC):

    @abstractmethod
    def apply(self, Basic: Basic) -> Basic:
        ...


IndexType = Union[Idx, Mul]

cse_return = Tuple[List[Tuple[Symbol, Expr]], List[Expr]]

def do_inv(e:Matrix)->Matrix:
    return cast(Matrix, e.inv()) # type: ignore[no-untyped-call]

def do_sqrt(e:Expr)->Expr:
    return cast(Expr, sqrt(e)) # type: ignore[no-untyped-call]

def do_det(e:Matrix)->Symbol:
    return cast(Symbol, e.det()) # type: ignore[no-untyped-call]

def do_sympify(e:Union[int,Expr])->Expr:
    return cast(Expr, sympify(e)) # type: ignore[no-untyped-call]

def do_simplify(e:Union[int,Expr])->Expr:
    return cast(Expr, simplify(e)) # type: ignore[no-untyped-call]

def cse(arg: List[Expr]) -> cse_return:
    return cast(cse_return, cse_(arg))  # type: ignore[no-untyped-call]


def mkIdx(name: str) -> Idx:
    return Idx(name)  # type: ignore[no-untyped-call]


def mkSymbol(name: str) -> Symbol:
    return Symbol(name)  # type: ignore[no-untyped-call]

def mkWild(name: str, exclude:List[Any]=list(), properties:List[Any]=list()) -> Wild:
    return Wild(name, exclude=exclude, properties=properties)  # type: ignore[no-untyped-call]

symar = List[List[Expr | int]]
def mkMatrix(array: symar) -> Matrix:
    return Matrix(array)  # type: ignore[no-untyped-call]


def mkZeros(*tup: int) -> Matrix:
    res = zeros(*tup) # type: ignore[no-untyped-call]
    assert isinstance(res, Matrix)
    return res

def mkFunction(name: str) -> UFunc:
    return Function(name)  # type: ignore[no-any-return]


def mkEq(a: Basic, b: Basic) -> Eq:
    return Eq(a, b)  # type: ignore[no-untyped-call]


def mkIdxs(names: str) -> Tuple[Idx, ...]:
    return tuple([Idx(name) for name in re.split(r'\s+', names)])  # type: ignore[no-untyped-call]


def mkIndexedBase(basename: str, shape: Tuple[int, ...]) -> IndexedBase:
    return IndexedBase(basename, shape=shape)  # type: ignore[no-untyped-call]


def mkIndexed(base: IndexedBase, *args: Union[int, IndexType]) -> Indexed:
    return Indexed(base, *args)  # type: ignore[no-untyped-call]


do_subs_table_type = Union[
    Mapping[Idx, Idx],
    Mapping[Indexed, Indexed],
    Mapping[Symbol, Expr],
    Mapping[Symbol, Symbol],
    Applier
]


def do_subs(sym: Expr, *tables: do_subs_table_type) -> Expr:
    result = sym
    for table in tables:
        if isinstance(table, Applier):
            result = cast(Expr, table.apply(result))
        else:
            result = cast(Expr, result.subs(table))  # type: ignore[no-untyped-call]
    return result

def mat_trans(mat:Matrix, tr:Callable[[Expr],Expr])->Matrix:
    table : List[List[Expr|int]] = list()
    for i in range(mat.rows):
        row : List[Expr|int] = list()
        for j in range(mat.cols):
            row += [tr(mat[i,j])]
        table += [row]
    return mkMatrix(table)

def do_matrix_subs(mat: Matrix, *tables: do_subs_table_type) -> Matrix:
    def do_sub(x:Expr)->Expr:
        return do_subs(x, *tables)
    return mat_trans(mat, do_sub)


call_match = Union[
    Callable[[Expr], bool],
    Callable[[IndexedBase], bool],
    Callable[[Symbol], bool]]

call_replace = Union[
    Callable[[Expr], Expr],
    Callable[[IndexedBase], Expr],
    Callable[[Symbol], Expr]]


def do_replace(sym: Expr, func_m: call_match, func_r: call_replace) -> Expr:
    ret = sym.replace(func_m, func_r)  # type: ignore[no-untyped-call]
    assert isinstance(ret, Expr)
    return ret

def do_diff(expr:Expr, sym:Symbol)->Expr:
    return cast(Expr, diff(expr, sym)) # type: ignore[no-untyped-call]

def do_match(expr:Expr, pat:Wild)->Optional[Dict[Wild, Expr]]:
    return cast(Optional[Dict[Wild, Expr]], expr.match(pat)) # type: ignore[no-untyped-call]

###
@multimethod
def add_free_indexed(arg: IndexedBase, rhs: Set[Idx])->None:
    pass

@add_free_indexed.register
def _(arg: Idx, rhs: Set[Idx])->None:
    rhs.add(arg)

@add_free_indexed.register
def _(arg: Basic, rhs: Set[Idx])->None:
    for arg in arg.args:
        add_free_indexed(arg, rhs)

def free_indexed(expr: Expr) -> Set[Idx]:
    rhs: Set[Idx] = set()
    add_free_indexed(expr, rhs)
    return rhs
###
@multimethod
def add_free_symbol(arg: Symbol, rhs: Set[Symbol])->None:
    rhs.add(arg)

@add_free_symbol.register
def _(arg: IndexedBase, rhs: Set[Symbol])->None:
    add_free_symbol(arg.args[0], rhs)

@add_free_symbol.register
def _(arg: Indexed, rhs: Set[Symbol])->None:
    raise DslException(f"Not a symbol: {arg}")

@add_free_symbol.register
def _(arg: Idx, rhs: Set[Symbol])->None:
    pass #raise DslException(f"Not a symbol: {arg}")

@add_free_symbol.register
def _(arg: Basic, rhs: Set[Symbol])->None:
    for arg in arg.args:
        add_free_symbol(arg, rhs)

def free_symbols(expr: Expr) -> Set[Symbol]:
    rhs: Set[Symbol] = set()
    add_free_symbol(expr, rhs)
    return rhs
