"""
Use the Sympy Indexed type for relativity expressions.
"""
import sys
from enum import auto
from typing import *
from mypy_extensions import Arg, VarArg, KwArg
import re
from multimethod import multimethod

from nrpy.finite_difference import setup_FD_matrix__return_inverse_lowlevel
from nrpy.helpers.coloring import coloring_is_enabled as colorize
from sympy import Integer, Number, Pow, Expr, Eq, Symbol, Indexed, IndexedBase, Matrix, Idx, Basic, Mul, MatrixBase, exp, ImmutableDenseMatrix
from sympy.core.function import UndefinedFunction as UFunc

from EmitCactus.dsl.coef import coef
from EmitCactus.dsl.dsl_exception import DslException
from EmitCactus.dsl.eqnlist import EqnList, DXI, DYI, DZI, DX, DY, DZ
from EmitCactus.dsl.symm import Sym
from EmitCactus.dsl.sympywrap import *
from EmitCactus.emit.ccl.interface.interface_tree import TensorParity, Parity, SingleIndexParity
from EmitCactus.emit.ccl.schedule.schedule_tree import ScheduleBlock, GroupOrFunction
from EmitCactus.emit.tree import Centering
from EmitCactus.util import OrderedSet, ScheduleBinEnum

__all__ = ["D", "div", "to_num", "mk_subst_type", "Param", "ThornFunction", "ScheduleBin", "ThornDef",
           "set_dimension", "get_dimension", "lookup_pair", "mksymbol_for_tensor_xyz", "mkPair",
           "stencil","DD","DDI",
           "ui", "uj", "uk", "ua", "ub", "uc", "ud", "u0", "u1", "u2", "u3", "u4", "u5",
           "li", "lj", "lk", "la", "lb", "lc", "ld", "l0", "l1", "l2", "l3", "l4", "l5"]

one = do_sympify(1)
zero = do_sympify(0)

lookup_pair:Dict[Idx,Idx] = dict()

###
def mk_mk_subst(s : str)->str:
    nextsub = 'a'
    pos = 0
    new_s = ""
    for g in re.finditer(r'\b([ul])([0-9])\b', s):
        new_s += s[pos:g.start()]
        pos = g.end()
        updn = g.group(1)
        index = g.group(2)
        new_s += updn
        new_s += nextsub
        nextsub = chr(ord(nextsub)+1)
    new_s += s[pos:]
    return new_s
        
###
import sympy as sy

class InvalidIndexError(DslException):
    def __init__(self, message:str)->None:
        self.message = message
        super().__init__(self.message)

class IndexTracker:
    def __init__(self)->None:
        self.free:OrderedSet[Idx] = OrderedSet()
        self.contracted:OrderedSet[Idx] = OrderedSet()
        self.used:OrderedSet[Idx] = OrderedSet()

    def all(self)->OrderedSet[Idx]:
        """
        The set of all contracted and free.
        """
        ret:OrderedSet[Idx] = OrderedSet()
        for a in self.free:
            ret.add(a)
        for a in self.contracted:
            ret.add(a)
            ret.add(lookup_pair[a])
        return ret

    def used_overlap(self, used:OrderedSet[Idx])->bool:
        for u in self.used:
            if u in used:
                return True
        for u in used:
            if u in self.used:
                return True
        return False

    def add(self, idx:Idx)->bool:
        """
        We keep single indices. So if we get ua and la,
        only la goes in contracted. If we get get ua and lc,
        both indices go in free. Used should not be added
        here.
        """
        global lookup_pair
        if (idx in self.free) or (idx in self.contracted):
            return False
        # TODO: Factor this logic out elsewhere
        letter_or_num = ord(str(idx)[1])
        if letter_or_num >= ord('0') and letter_or_num <= ord('9'):
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
    def __repr__(self)->str:
        return "(free:"+repr(self.free)+", contracted:"+repr(self.contracted)+", used:"+repr(self.used)+")"

class IndexContractionVisitor:
    def __init__(self, defn:Dict[str, Tuple[str, List[Idx]]])->None:
        self.defn = defn

    @multimethod
    def visit(self, expr: sy.Basic) -> Tuple[Expr,IndexTracker]:
        raise Exception(str(expr)+" "+str(type(expr)))

    @visit.register
    def _(self, expr: sy.Add) -> Tuple[Expr,IndexTracker]:
        it:Optional[IndexTracker] = None
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
            return (new_expr, IndexTracker())
        else:
            return (new_expr, it)

    def contract(self, expr:Expr, it:IndexTracker)->Tuple[Expr,IndexTracker]:
        #print("contract:",expr)
        if len(it.contracted) > 0:
            #print(colorize("CONTRACTED:", "cyan"), expr, it)
            for loidx in it.contracted:
                new_expr:Expr = zero
                upidx = lookup_pair[loidx]
                for i in range(get_dimension()): 
                    loidx_val = [l0,l1,l2,l3,l4,l5][i]
                    upidx_val = [u0,u1,u2,u3,u4,u5][i]
                    new_expr += do_isub(expr, dict(), {loidx: loidx_val, upidx: upidx_val})
                expr = new_expr
            it.used = it.contracted
            it.contracted = OrderedSet()
            return new_expr, it
        else:
            return expr, it

    @visit.register
    def _(self, expr: sy.Mul) -> Tuple[Expr,IndexTracker]:
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
    def _(self, expr: sy.Symbol) -> Tuple[Expr,IndexTracker]:
        return expr, IndexTracker()

    @visit.register
    def _(self, expr: sy.Integer) -> Tuple[Expr,IndexTracker]:
        return expr, IndexTracker()

    @visit.register
    def _(self, expr: sy.Rational) -> Tuple[Expr,IndexTracker]:
        return expr, IndexTracker()

    @visit.register
    def _(self, expr: sy.Float) -> Tuple[Expr,IndexTracker]:
        return expr, IndexTracker()

    @visit.register
    def _(self, expr: sy.Idx) -> Tuple[Expr,IndexTracker]:
        return expr, IndexTracker()

    @visit.register
    def _(self, expr: sy.Indexed) -> Tuple[Expr,IndexTracker]:
        basename = str(expr.args[0])
        if basename in self.defn:
            bn, indices = self.defn[basename]
            if len(indices)+1 != len(expr.args):
                raise InvalidIndexError(f"indices used on a non-indexed quantity '{expr}' in:")
        else:
            assert len(self.defn) == 0
        it = IndexTracker()
        for a in expr.args[1:]:
            a_it = self.visit(a)
            assert isinstance(a, Idx)
            if not it.add(a):
                raise InvalidIndexError(str(expr))
        return self.contract(expr, it)

    @visit.register
    def _(self, expr: sy.Function) -> Tuple[Expr,IndexTracker]:
        it = IndexTracker()
        new_args : List[Expr] = list()
        for a in expr.args:
            if isinstance(a, Idx):
                if not it.add(a):
                    raise InvalidIndexError(repr(expr))
                new_args.append(a)
            else:
                a_expr , a_it = self.visit(a)
                new_args.append(a_expr)
                for idx in a_it.all():
                    it.add(idx)
        ret = self.contract(expr.func(*new_args), it)
        return ret

    @visit.register
    def _(self, expr: sy.Pow) -> Tuple[Expr,IndexTracker]:
        new_args: List[Expr] = list()
        for a in expr.args:
            new_arg, it = self.visit(a)
            #print("-->", a, expr, it)
            new_args += [new_arg]
            if len(it.free) != 0 or len(it.contracted) != 0:
                raise InvalidIndexError(repr(expr))
        return sy.Pow(*new_args), IndexTracker()

    @visit.register
    def _(self, expr: sy.IndexedBase) -> Tuple[Expr,IndexTracker]:
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
            raise InvalidIndexError(f"Expression '{expr}' was declared with {msg}, but was used in this expression without indices: ")
        return expr, IndexTracker()


### ind subs
class IndexSubsVisitor:
    def __init__(self, defn:Dict[Indexed, Expr])->None:
        self.defn = defn
        self.idxsubs: Dict[Idx, Idx]  = dict()

    @multimethod
    def visit(self, expr: sy.Add) -> Expr:
        r = do_sympify(0)
        for a in expr.args:
            r += self.visit(a)
        return r

    @visit.register
    def _(self, expr: sy.Mul) -> Expr:
        r = do_sympify(1)
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
    def _(self, expr: sy.Idx) -> Expr:
        res = self.idxsubs.get(expr, None)
        if res is None:
            return expr
        else:
            return res

    @visit.register
    def _(self, expr: sy.Indexed) -> Expr:
        r: Indexed = expr
        if len(self.idxsubs) > 0:
            indexes : List[Idx] = list()
            for a in expr.args[1:]:
                assert isinstance(a, Idx)
                indexes.append(self.idxsubs.get(a,a))
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
    def _(self, expr: sy.Pow) -> Expr:
        return cast(Expr, sy.Pow(self.visit(expr.args[0]), self.visit(expr.args[1])))

    @visit.register
    def _(self, expr: sy.IndexedBase) -> Expr:
        return expr

def do_isub(expr:Expr, subs:Optional[Dict[Indexed,Expr]]=None, idxsubs:Optional[Dict[Idx,Idx]]=None)->Expr:
    if subs is None:
        subs = dict()
    if idxsubs is None:
        idxsubs = dict()
    isub = IndexSubsVisitor(subs)
    isub.idxsubs = idxsubs
    # FIXME Why is this cast needed?
    return cast(Expr, isub.visit(expr))

def check_indices(rhs:Expr, defn:Optional[Dict[str, Tuple[str, List[Idx]]]]=None)->IndexTracker:
    """
    This function not only checks the validity of indexed expressions, it returns
    all free and contracted indices.
    """
    if defn is None:
        defn = dict()
    err = IndexContractionVisitor(defn)
    ret : IndexTracker
    try:
        _, ret = err.visit(rhs)
        die = False
        msg = ''
    finally:
        pass
    #except InvalidIndexError as sie:
    #    die = True
    #    msg = sie.message
    #if die:
    #    raise InvalidIndexError(msg+str(rhs))
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
def mkPair(s: Optional[str]=None) -> Tuple[Idx, Idx]:
    """
    Returns a tuple containing an upper/lower index pair.
    """
    global pair_tmp_name
    if s is None:
        s = pair_tmp_name
        tmpnum = ord(pair_tmp_name[-1])
        if tmpnum == ord("Z"):
            pair_tmp_name += "A"
        else:
            pair_tmp_name = pair_tmp_name[0:-1] + chr(tmpnum+1)
        tmpnum += 1
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
ui, li = mkPair('i')
uj, lj = mkPair('j')
uk, lk = mkPair('k')
ua, la = mkPair('a')
ub, lb = mkPair('b')
uc, lc = mkPair('c')
ud, ld = mkPair('d')
ui, li = mkPair('i')
uj, lj = mkPair('j')
uk, lk = mkPair('k')
u0, l0 = mkPair('0')
u1, l1 = mkPair('1')
u2, l2 = mkPair('2')
u3, l3 = mkPair('3')
u4, l4 = mkPair('4')
u5, l5 = mkPair('5')
up_indices = u0, u1, u2, u3, u4, u5
down_indices = l0, l1, l2, l3, l4, l5

### dmv
from sympy import sin, cos
x = mkSymbol("x")
y = mkSymbol("y")
z = mkSymbol("z")
noidx = mkIdx("noidx")
dummy = mkSymbol("_dummy_")

def mkdiv(div_fun:UFunc, expr:Expr, *args:Idx)->Expr:
    r = div_fun(expr, *args)
    assert isinstance(r, Expr)
    return r

class DivMakerVisitor:
    def __init__(self, div_fun:UFunc, coords: Optional[List[Symbol]]=None)->None:
        self.div_func = div_fun
        self.div_name = str(div_fun)
        self.params:Set[Symbol] = set()
        if coords is None:
            coords = [x,y,z]
        self.coords = coords
        self.idxmap = dict()
        for i in range(len(coords)):
            self.idxmap[coords[i]] = down_indices[i]

    @multimethod
    def visit(self, expr: sy.Basic, idx: sy.Idx) -> Expr:
        raise Exception(str(expr)+" "+str(type(expr)))

    @visit.register
    def _(self, expr: sy.Add, idx: sy.Idx) -> Expr:
        r = zero
        for a in expr.args:
            r += self.visit(a, idx)
        return r

    @visit.register
    def _(self, expr: sy.Mul, idx: sy.Idx) -> Expr:
        if idx is not noidx:
            s = zero
            for i in range(len(expr.args)):
                term = one
                for j in range(len(expr.args)):
                    a = expr.args[j]
                    if i==j:
                        term *= self.visit(a, idx)
                    else:
                        term *= self.visit(a, noidx)
                s += term
            return s
        else:
            s = one
            for a in expr.args:
                s *= self.visit(a, noidx)
            return s

    @visit.register
    def _(self, expr: sy.Symbol, idx: sy.Idx) -> Expr:
        if idx is noidx:
            return expr
        ####
        # TODO: generalize for other dimensions than 3
        #assert get_dimension()==3
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

        return mkdiv(self.div_func, expr, idx)

    @visit.register
    def _(self, expr: sy.Integer, idx: sy.Idx) -> Expr:
        if idx is noidx:
            return expr
        return zero

    @visit.register
    def _(self, expr: sy.Rational, idx: sy.Idx) -> Expr:
        if idx is noidx:
            return expr
        return zero

    @visit.register
    def _(self, expr: sy.Float, idx: sy.Idx) -> Expr:
        if idx is noidx:
            return expr
        return zero

    @visit.register
    def _(self, expr: sy.Idx, idx: sy.Idx) -> Expr:
        raise Exception("Derivative of Index")

    @visit.register
    def _(self, expr: sy.Indexed, idx: sy.Idx) -> Expr:
        if idx is noidx:
            return expr
        return mkdiv(self.div_func, expr, idx)

    @visit.register
    def _(self, expr: sy.IndexedBase, idx: sy.Idx) -> Expr:
        if idx is noidx:
            return expr
        return mkdiv(self.div_func, expr, idx)

    @visit.register
    def _(self, expr: sy.Function, idx: sy.Idx) -> Expr:
        r = expr.args[0]
        name = expr.func.__name__
        if name == self.div_name:

            # Handle div of div
            assert isinstance(expr.args[0], Expr)
            sub: Expr = expr.args[0]
            sub = self.visit(sub, noidx)
            if len(expr.args) > 2:
                for idx1 in expr.args[1:]:
                    sub = self.visit(sub, idx1)
                return sub
            if isinstance(sub, sy.Function) and sub.func.__name__ == self.div_name:
                args = sorted(sub.args[1:] + expr.args[1:],key=lambda x : str(x))
                return mkdiv(self.div_func, sub.args[0], *args)

            for idx1 in expr.args[1:]:
                sub = self.visit(sub, idx1)

            if idx is not noidx:
                sub = self.visit(self.div_func(sub, idx), noidx)

            return sub
        elif idx is noidx:
            return expr
        else:
            if name == "sin":
                f = cos(r)*self.visit(r, idx)
            elif name == "cos":
                f = -sin(r)*self.visit(r, idx)
            elif name == "exp":
                f = exp(r)*self.visit(r, idx)
            else:
                raise Exception("unknown func")
            assert isinstance(f, Expr)
            return f

    @visit.register
    def _(self, expr: sy.Pow, idx:sy.Idx) -> Expr:
        if idx is noidx:
            return expr
        else:
            r = expr.args[0]
            n = expr.args[1]
            ret = n*r**(n-1)*self.visit(r, idx)
            assert isinstance(ret, Expr)
            return ret

dmv = DivMakerVisitor(div)
dmv2 = DivMakerVisitor(D)

def assert_eq(a: Expr ,b: Expr)->None:
    assert a is not None
    r =  do_simplify(a - b)
    assert r == 0, f"{a} minus {b} !=0, instead {r}"

def do_div(expr: Basic)->Expr:
    r = dmv.visit(expr, noidx)
    r = dmv2.visit(r, noidx)
    assert isinstance(r, Expr)
    return r

if __name__ == "__main__":
    foo = mkIndexedBase("foo",(1,))
    gxx = mkSymbol("gxx")
    gxy = mkSymbol("gxy")
    gyy = mkSymbol("gyy")
    gzz = mkSymbol("gzz")
    gyz = mkSymbol("gyz")
    gxz = mkSymbol("gxz")

    expr1 = div(gxx**2, l0, l0)
    #expr2 = 2*div(gxx*div(gxx, l0),l0)
    expr2 = 2*div(gxx, l0)**2 + 2*gxx*div(gxx, l0, l0)
    assert_eq( do_div(expr1), expr2 )

    expr1 = - gyy*div(-gxz, l0) - gyy*div(gxz, l0)
    expr2 = zero
    assert_eq( do_div(expr1), expr2 )

    expr1 = -2*gxy*div(gxy, l2) + div(gxy**2, l2)
    expr2 = zero
    assert_eq( do_div(expr1), expr2 )

    expr1 = div(-gxy**2, l2)
    expr2 = -2*gxy*div(gxy, l2)
    assert_eq( do_div(expr1), expr2 )

    expr1 = div(gxx*gyy - gxy**2, l2)
    expr2 = gxx*div(gyy,l2)+div(gxx,l2)*gyy-2*gxy*div(gxy,l2)
    assert_eq( do_div(expr1), expr2 )

    assert_eq(do_div(div(x,l0)),one)
    assert_eq(do_div(div(y,l0)),zero)
    assert_eq(do_div(div(x**3,l0)),3*x**2)
    assert_eq(do_div(div(sin(x),l0)),cos(x))
    assert_eq(do_div(div(x**2+x**3,l0)),2*x + 3*x**2)
    assert_eq(do_div(div(x**2+x**3,l1)),zero)
    assert_eq(do_div(div(1/(2+x**2),l0)),-2*x/(2+x**2)**2)
    assert_eq(do_div(div(x**3/(2+x**2),l0)),-2*x**4/(2+x**2)**2 + 3*x**2/(2+x**2))
    assert_eq(do_div(div(x**2*sin(x),l0)),x*(x*cos(x)+2*sin(x)))
    assert_eq(do_div(div(sin(x**3),l0)),cos(x**3)*3*x**2)
    assert_eq(do_div(div(foo[la],l0)),div(foo[la],l0))
    assert_eq(do_div(div(div(foo[la],lb),lc)), div(foo[la],lb,lc))
    assert_eq(do_div(div(div(foo[la],lc),lb)), div(foo[la],lb,lc))
    assert_eq(do_div(div(div(foo[la],l0),l1)), div(foo[la],l0,l1))
    assert_eq(do_div(div(div(foo[la],l1),l0)), div(foo[la],l0,l1))
    assert_eq(do_div(x*div(x,l0)), x)
    assert_eq(do_div(x+div(x,l0)), x+1)
    assert_eq(do_div(x+x*div(x,l0)), 2*x)
    assert_eq(do_div(x*(x+x*div(x,l0))), 2*x**2)
    assert_eq(do_div(x*(x/2+3*x*div(x,l0))), (7/2)*x**2)
    assert_eq(do_div(div(foo,l0)), div(foo,l0))
    assert_eq( do_div(div(gxx, l1)/2 + div(gxy, l0)), div(gxx, l1)/2 + div(gxy, l0) )
    assert_eq( do_div(div(exp(x),l0)), exp(x))
    assert_eq( do_div(div(exp(x)/2,l0)), exp(x)/2)
    expr = (gxy*gyz - gxz*gyy)*(-div(gxx, l2)/2 + div(gxz, l0))
    assert_eq(do_div(expr), expr)
    expr1 = (gxy*gyz - gxz*gyy + gzz)*(-div(gxx, l2)/2 + div(gxz+gzz, l0))
    expr2 = (gxy*gyz - gxz*gyy + gzz)*(-div(gxx, l2)/2 + div(gxz, l0) +div(gzz, l0))
    assert_eq(do_div(expr1), expr2)
    expr = (gxx*gyz - gxy*gxz)*(div(gxx, l1) - 2*div(gxy, l0))/2 + (gxy*gyz - gxz*gyy)*div(gxx, l0)/2
    assert_eq( do_div(expr), expr )
    expr1 = div(gxx*gyy, l0)
    expr2 = div(gxx, l0)*gyy + div(gyy, l0)*gxx
    assert_eq( do_div(expr1), expr2 )
    expr1 = x**6/3 + sin(x)/x
    expr2 = 2*x**5 + cos(x)/x - sin(x)/x**2
    assert_eq( do_div(div(expr1, l0)), expr2 )
    expr1 = (x+sin(x))*(1/x+cos(x))
    expr2 = (1+cos(x))*(1/x+cos(x))-(1/x**2+sin(x))*(x+sin(x))
    assert_eq( do_div(div(expr1, l0)), expr2 )
    expr1 = 1/(x+sin(x))
    expr2 = -(1+cos(x))/(x+sin(x))**2
    assert_eq( do_div(div(expr1, l0)), expr2 )
    assert_eq( do_div(div(sqrt(x), l0)), 1/sqrt(x)/2 )
### dmv

TA = TypeVar("TA")


def chkcast(obj: Any, typ: Type[TA]) -> TA:
    """
    Checked cast
    """
    assert isinstance(obj, typ), f"expected type {typ} found type {type(obj)}"
    return obj


def sub_inds(idx: Idx, values: Dict[Idx, Idx]) -> Idx:
    return chkcast(do_subs(idx, values), Idx)


def toNumTup2(li: List[Idx], values: Dict[Idx, Idx]) -> Tuple[int, ...]:
    return tuple([to_num(sub_inds(x, values)) for x in li])


def toNumTup(li: Tuple[Basic, ...], values: Dict[Idx, Idx]) -> Tuple[int, ...]:
    return toNumTup2([chkcast(x, Idx) for x in li], values)


stencil = mkFunction("stencil")
DD = mkFunction("DD")
DDI = mkFunction("DDI")
noop = mkFunction("noop")

multype = Mul  # type(i*j)
addtype = type(ui + uj)
eqtype = Eq
powtype = type(ui ** uj)

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
    ###
    if type(xpr) in [multype, addtype, powtype]:
        for arg in xpr.args:
            ret.update(get_indices(arg))
        return ret
    elif hasattr(xpr, "indices"):
        ret.update(xpr.indices)
    return ret


def byname(x: Idx) -> str:
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
    return num0 <= n and n <= num9


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
    indices = sorted(indices, key=byname)
    ret: OrderedSet[Idx] = OrderedSet()
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and is_pair(indices[i], indices[i + 1]):
            i += 2
        else:
            ret.add(indices[i])
            i += 1
    return ret


if __name__ == "__main__":
    M = mkIndexedBase('M', (dimension, dimension))
    print(get_free_indices(M[ui, uj] * M[lj, lk]))
    #assert sorted(list(get_free_indices(M[ui, uj] * M[lj, lk])), key=byname) == [ui, lk]


def get_contracted_indices(xpr: Expr) -> OrderedSet[Idx]:
    """ Return all contracted indices in xpr. """
    indices = list(get_indices(xpr))
    indices = sorted(indices, key=byname)
    ret: OrderedSet[Idx] = OrderedSet()
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and is_pair(indices[i], indices[i + 1]):
            ret.add(indices[i])
            i += 2
        else:
            i += 1
    return ret


if __name__ == "__main__":
    assert sorted(list(get_contracted_indices(M[ui, uj] * M[lj, lk])), key=byname) == [lj]


def incr(index_list: List[Idx], index_values: Dict[Idx, Idx]) -> bool:
    """ Increment the indices in index_list, creating an index_values table with all possible permutations. """
    if len(index_list) == 0:
        return False
    ix = 0
    if len(index_values) == 0:
        for ind_ in index_list:
            uind, ind = get_pair(ind_)
            index_values[ind] = l0
            index_values[uind] = u0
        return True
    while True:
        if ix >= len(index_list):
            return False
        uind, ind = get_pair(index_list[ix])
        index_value = to_num(index_values[ind])
        if index_value == dimension - 1:
            index_values[ind] = l0
            index_values[uind] = u0
            ix += 1
        else:
            index_values[ind] = down_indices[index_value + 1]
            index_values[uind] = up_indices[index_value + 1]
            break
    return True


# Check
ilist = [ui, lj]
dvals: Dict[Idx, Idx] = dict()
valscount = 0
while incr(ilist, dvals):
    valscount += 1
assert valscount == dimension ** 2


def expand_contracted_indices(in_expr:Expr, sym:Sym)->Expr:
    viz = IndexContractionVisitor(dict())
    expr, it = viz.visit(in_expr)
    expr = sym.apply(expr)
    assert isinstance(expr, Expr)
    return expr


# Check
if __name__ == "__main__":
    sym = Sym()
    assert expand_contracted_indices(M[ui, li], sym) == M[u0, l0] + M[u1, l1] + M[u2, l2]
    assert expand_contracted_indices(M[ui, lj] * M[li, uk], sym) == M[l0, uk] * M[u0, lj] + M[l1, uk] * M[u1, lj] + M[
    l2, uk] * M[u2, lj]


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


def _mksymbol_for_tensor(sym: Indexed, *args: Idx) -> Expr:
    newstr = str(sym.base)
    for ind in sym.args[1:]:
        assert isinstance(ind, Idx)
        if is_lower(ind):
            newstr += "D"
        elif is_upper(ind):
            newstr += "U"
        else:
            assert False
    for ind in sym.args[1:]:
        assert isinstance(ind, Idx)
        newstr += str(to_num(ind))
    return mkSymbol(newstr)


def mksymbol_for_tensor(sym: Indexed) -> Expr:
    """
    Define a symbol for a tensor using standard NRPy+ rules.
    For an upper index put a U, for a lower index put a D.
    Follow the string of U's and D's with the integer value
    of the up/down index.

    :param out: The tensor expression with integer indices.

    :return: a new sympy symbol
    """
    if sym.is_Function and hasattr(sym, "name") and sym.name == "div":
        assert isinstance(sym.args[0], Indexed)
        newstr = _mksymbol_for_tensor(sym.args[0]) + "_d"
        for ind in sym.args[1:]:
            assert isinstance(ind, Idx)
            if is_lower(ind):
                newstr += "D"
            elif is_upper(ind):
                newstr += "U"
            else:
                assert False
        for ind in sym.args[1:]:
            assert isinstance(ind, Idx)
            newstr += str(to_num(ind))
        return mkSymbol(newstr)
    else:
        return _mksymbol_for_tensor(sym)


## mksymbol_for_tensor_xyz

def _mksymbol_for_tensor_xyz(sym: Indexed, *args: Idx) -> str:
    newstr = str(sym.base)
    for ind in sym.args[1:]:
        assert isinstance(ind, Idx)
        newstr += ["x", "y", "z"][to_num(ind)]
    return newstr


def mksymbol_for_tensor_xyz(sym: Indexed, *idxs: int) -> Symbol:
    """
    Define a symbol for a tensor using standard Cactus rules.
    Don't distinguish up/down indices. Use suffixes based on
    x, y, and z at the end.

    :param out: The tensor expression with integer indices.

    :return: a new sympy symbol
    """
    if sym.is_Function and hasattr(sym, "name") and sym.name == "div":
        assert isinstance(sym.args[0], Indexed)
        newstr = _mksymbol_for_tensor_xyz(sym.args[0]) + "_d"
        for ind in sym.args[1:]:
            assert isinstance(ind, Idx)
            newstr += str(to_num(ind))
        return mkSymbol(newstr)
    else:
        return mkSymbol(_mksymbol_for_tensor_xyz(sym))


mk_subst_type = Callable[[Indexed, VarArg(int)], Expr]


def mk_subst_default(out: Indexed, *inds: int) -> Expr:
    return mksymbol_for_tensor(out)

param_default_type = Union[float, int, str, bool]
param_values_type = Optional[Union[Tuple[float, float], Tuple[int, int], Tuple[bool, bool], str, Set[str]]]
min_max_type = Union[Tuple[float, float], Tuple[int, int]]


class Param:
    def __init__(self, name: str, default: param_default_type, desc: str, values: param_values_type) -> None:
        self.name = name
        self.values = values
        self.desc = desc
        self.default = default

    def get_min_max(self) -> min_max_type:
        ty = self.get_type()
        if ty == int:
            if self.values is not None:
                return cast(min_max_type, self.values)
            return (-2 ** 31, 2 ** 31 - 1)
        elif ty == float:
            if self.values is not None:
                return cast(min_max_type, self.values)
            return (sys.float_info.min, sys.float_info.max)
        else:
            assert False

    def get_values(self) -> param_values_type:
        if self.values is not None:
            return self.values
        ty = self.get_type()
        if ty == bool:
            return (False, True)
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
    divnm = "div" + "xyz"[i]
    globals()[divnm] = mkFunction(divnm)

# Second derivatives
for i in range(dimension):
    for j in range(i, dimension):
        divnm = "div" + "xyz"[i] + "xyz"[j]
        globals()[divnm] = mkFunction(divnm)


def to_div(out: Expr) -> Expr:
    nm = "div"
    for k in out.args[1:]:
        assert isinstance(k, Idx)
        nm += "xyz"[to_num(k)]
    divnn = mkFunction(nm)
    arg = out.args[0]  # div(v, i, j) -> v
    return cast(Expr, divnn(arg))


class ApplyDiv(Applier):
    def __init__(self) -> None:
        self.val: Optional[Expr] = None

    def m(self, expr: Expr) -> bool:
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

    def r(self, expr: Basic) -> Optional[Expr]:
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

    def __init__(self, n: int, funs1:Dict[Tuple[UFunc,Idx],Expr], funs2:Dict[Tuple[UFunc,Idx,Idx],Expr], fun_args:Dict[str,int]) -> None:
        self.val: Optional[Expr] = None
        self.n = n
        self.fd_matrix = setup_FD_matrix__return_inverse_lowlevel(n, 0)
        self.funs1 = funs1
        self.funs2 = funs2
        self.fun_args = fun_args

    def is_user_func(self, f:Expr)->Optional[Expr]:
        if not f.is_Function:
            return None
        if hasattr(f,"name") and f.name in self.fun_args:
            nargs = self.fun_args[f.name]
            if len(f.args) != nargs:
                raise DslException(f"function {f.name} called with wrong number of args. Expected {nargs}, got {len(f.args)}. Expr: {f}")
            return None
        elif len(f.args) == 2:
            #assert isinstance(f.func, UFunc)
            assert isinstance(f.args[1], Idx), f"f={f}"
            return self.funs1.get((f.func, f.args[1]),None)
        elif len(f.args) == 3:
            assert isinstance(f.args[1], Idx)
            assert isinstance(f.args[2], Idx)
            if f.args[1] == f.args[2]:
                return self.funs1.get((f.func, f.args[1]),None)
            else:
                return self.funs2.get((f.func, f.args[1], f.args[2]),None)
        return None

    def m(self, expr: Expr) -> bool:
        if (fundef := self.is_user_func(expr)) is not None:
            assert isinstance(expr.args[0], Expr)
            self.val = do_subs(fundef, {dummy:expr.args[0]})
            return True
        elif expr.is_Function and hasattr(expr, "name") and expr.name == "stencil":
            new_expr1: List[int]= list() 
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
            dxt = do_sympify(1)
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
                        termi = coefs[i]
                        for j in range(len(coefs)):
                            term = coefs[j] * termi
                            new_expr += [(term, mkterm(expr.args[0], i - len(coefs) // 2, j - len(coefs) // 2, 0))]
                    dxt = DXI * DYI
                elif expr.args[1:] in ((l0, l2), (l2, l0)):
                    coefs = self.fd_matrix.col(1)
                    for i in range(len(coefs)):
                        termi = coefs[i]
                        for j in range(len(coefs)):
                            term = coefs[j] * termi
                            new_expr += [(term, mkterm(expr.args[0], i - len(coefs) // 2, 0, j - len(coefs) // 2))]
                    dxt = DXI * DZI
                elif expr.args[1:] in ((l1, l2), (l2, l1)):
                    coefs = self.fd_matrix.col(1)
                    for i in range(len(coefs)):
                        termi = coefs[i]
                        for j in range(len(coefs)):
                            term = coefs[j] * termi
                            new_expr += [(term, mkterm(expr.args[0], 0, i - len(coefs) // 2, j - len(coefs) // 2))]
                    dxt = DYI * DZI
                else:
                    raise Exception()

            if len(new_expr) > 0:
                new_expr = sorted(new_expr, key=sort_exprs)
                self.val = do_sympify(0)
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

    def r(self, expr: Expr) -> Optional[Expr]:
        return self.val

    def apply(self, arg: Basic) -> Basic:
        return cast(Basic, arg.replace(self.m, self.r))  # type: ignore[no-untyped-call]


val = mkSymbol("val")
x = mkSymbol("x")
y = mkSymbol("y")
z = mkSymbol("z")


class ScheduleBin(ScheduleBinEnum):
    Evolve = auto(), 'Evolve', False
    Init = auto(), 'Init', True
    Analysis = auto(), 'Analysis', True
    EstimateError = auto(), 'EstimateError', False
    DriverInit = auto(), 'ODESolvers_Initial', False
    PostStep = auto(), 'ODESolvers_PostStep', False


ScheduleTarget = ScheduleBin | ScheduleBlock

class ThornFunction:
    def __init__(self,
                 name: str,
                 schedule_target: ScheduleTarget,
                 thorn_def: "ThornDef",
                 schedule_before: Optional[Collection[str]],
                 schedule_after: Optional[Collection[str]]) -> None:
        self.schedule_target = schedule_target
        self.name = name
        self.thorn_def = thorn_def
        self.eqn_list: EqnList = EqnList(thorn_def.is_stencil)
        self.been_baked: bool = False
        self.schedule_before: Collection[str] = schedule_before or list()
        self.schedule_after: Collection[str] = schedule_after or list()

        if isinstance(schedule_target, ScheduleBlock) and schedule_target.group_or_function is GroupOrFunction.Function:
            raise DslException("Cannot schedule into this schedule block because it is not a schedule group.")

    def _add_eqn2(self, lhs2: Symbol, rhs2: Expr) -> None:
        rhs2 = self.thorn_def.do_subs(expand_contracted_indices(rhs2, self.thorn_def.symmetries))
        if str(lhs2) in self.thorn_def.gfs and str(lhs2) not in self.thorn_def.temp:
            self.eqn_list.add_output(lhs2)
        for item in free_symbols(rhs2):
            if str(item) in self.thorn_def.gfs:
                # assert item.is_Symbol
                if str(item) not in self.thorn_def.temp:
                    self.eqn_list.add_input(item)
            elif str(item) in self.thorn_def.params:
                assert item.is_Symbol
                self.eqn_list.add_param(item)
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
        self.eqn_list.add_eqn(lhs2, rhs2)
        print(colorize("Add eqn:","green"),lhs2,colorize("->","cyan"),rhs2)

    def get_free_indices(self, expr : Expr) -> OrderedSet[Idx]:
        it = check_indices(expr, self.thorn_def.defn)
        return it.free

    #def expand_free_indices(self, expr : Expr)->Expr

    @multimethod
    def add_eqn(self, lhs: Indexed, rhs: Expr) -> None:
        check_indices(rhs, self.thorn_def.defn)

        if self.been_baked:
            raise Exception("add_eqn should not be called on a baked ThornFunction")

        lhs2: Symbol
        if self.get_free_indices(lhs) != self.get_free_indices(rhs):
            raise Exception(f"Free indices of '{lhs}' and '{rhs}' do not match.")
        count = 0
        for tup in expand_free_indices(lhs, self.thorn_def.symmetries):
            count += 1
            lhsx, inds, _ = tup
            lhs2_: Basic = do_isub(lhsx, self.thorn_def.subs) #.thorn_def.do_subs(lhsx, self.thorn_def.subs)
            if not isinstance(lhs2_, Symbol):
                mms = mk_mk_subst(repr(lhs2_))
                raise Exception(f"'{lhs2_}' does not evaluate a Symbol. Did you forget to call mk_subst({mms},...)?")
            lhs2 = lhs2_
            rhs0 = rhs
            rhs2 = self.thorn_def.do_subs(rhs0, self.thorn_def.subs, inds)
            # rhs2 = self.thorn_def.do_subs(rhs2, inds, self.thorn_def.subs)
            self._add_eqn2(lhs2, rhs2)
        if count == 0:
            # TODO: Understand what's going on with arg 0
            for ind in lhs.args[1:]:
                assert isinstance(ind, Idx)
                assert is_numeric_index(ind)
            lhs2 = cast(Symbol, self.thorn_def.do_subs(lhs, self.thorn_def.subs))
            rhs2 = self.thorn_def.do_subs(rhs, self.thorn_def.subs)
            self._add_eqn2(lhs2, rhs2)

    @add_eqn.register
    def _(self, lhs: IndexedBase, rhs: Expr) -> None:

        if self.been_baked:
            raise Exception("add_eqn should not be called on a baked ThornFunction")

        lhs2 = cast(Symbol, self.thorn_def.do_subs(lhs, self.thorn_def.subs))
        eci = expand_contracted_indices(rhs, self.thorn_def.symmetries)
        rhs2 = do_isub(eci)
        self._add_eqn2(lhs2, rhs2)
        
    @add_eqn.register
    def _(self, lhs: Indexed, rhs: Matrix) -> None:

        if self.been_baked:
            raise Exception("add_eqn should not be called on a baked ThornFunction")

        count = 0
        for tup in expand_free_indices(lhs, self.thorn_def.symmetries):
            count += 1
            lhsx, inds, _ = tup
            lhs2_ = do_isub(lhsx, self.thorn_def.subs) #.thorn_def.do_subs(lhsx, self.thorn_def.subs)
            lhs2 = lhs2_
            arr_inds = toNumTup(lhs.args[1:], inds)
            rhs0 = rhs[arr_inds]
            rhs2 = self.thorn_def.do_subs(rhs0, self.thorn_def.subs, inds)
            # rhs2 = self.thorn_def.do_subs(rhs2, inds, self.thorn_def.subs)
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
            lhsx, inds, idx = tup
            num_idx = to_num(inds[idx[0]])
            lhs2_ = do_isub(lhsx, self.thorn_def.subs) #.thorn_def.do_subs(lhsx, self.thorn_def.subs)
            lhs2 = lhs2_
            rhs2 = rhs[num_idx]
            assert isinstance(lhs2, Symbol)
            self._add_eqn2(lhs2, rhs2)
        assert count > 0

    def madd(self) -> None:
        self.eqn_list.madd()

    def cse(self) -> None:
        self.eqn_list.cse()

    def dump(self) -> None:
        self.eqn_list.dump()

    def eqn_bake(self) -> None:
        self.eqn_list.bake()

    def recycle_temporaries(self) -> None:
        self.eqn_list.recycle_temporaries()

    def split_output_eqns(self) -> None:
        self.eqn_list.split_output_eqns()

    def bake(self, *,
             do_cse: bool = True,
             do_madd: bool = False,
             do_recycle_temporaries: bool = True,
             do_split_output_eqns: bool = True) -> None:
        """
        Finalize this function in preparation to be passed to a generator.
        :param do_cse: If true, perform SymPy's common subexpression elimination.
        :param do_madd: If true, attempt to generate fused multiply-add function calls where appropriate.
        :param do_recycle_temporaries: If true, attempt to conserve register use by recycling temporary variables.
        :param do_split_output_eqns: If true, split apart equations whose LHSes are output variables.
        :return:
        """
        if self.been_baked:
            raise Exception("bake should not be called more than once")
        print(f"*** {self.name} ***")

        if do_madd:
            self.madd()
        if do_cse:
            self.cse()

        self.eqn_bake()

        if do_split_output_eqns:
            self.split_output_eqns()

        if do_recycle_temporaries:
            self.recycle_temporaries()

        self.been_baked = True

    def show_tensortypes(self) -> None:
        keys: Set[str] = OrderedSet()
        for k1 in self.eqn_list.inputs:
            keys.add(str(k1))
        for k2 in self.eqn_list.outputs:
            keys.add(str(k2))
        for k in keys:
            group, indices, members = self.get_tensortype(k)
            print(colorize(k, "green"), "is a member of", colorize(group, "green"), "with indices",
                  colorize(indices, "cyan"), "and members", colorize(members, "magenta"))

    def get_tensortype(self, item: Union[str, Symbol]) -> Tuple[str, List[Idx], List[str]]:
        return self.thorn_def.get_tensortype(item)


class ThornDef:
    def __init__(self, arr: str, name: str, run_simplify: bool = True) -> None:
        self.fun_args:Dict[str,int] = dict()
        self.run_simplify = run_simplify
        self.coords: List[Symbol] = list()
        self.apply_div: Applier = ApplyDiv()
        self.arrangement = arr
        self.name = name
        self.symmetries = Sym()
        self.base2group: Dict[str, str] = dict()
        self.gfs: Dict[str, IndexedBase] = dict()
        self.subs: Dict[Indexed, Expr] = dict()
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
        self.funs1: Dict[Tuple[UFunc,Idx],Expr] = dict()
        self.funs2: Dict[Tuple[UFunc,Idx,Idx],Expr] = dict()
        self.dmvs:Dict[str,DivMakerVisitor]=dict()
        self.set_derivative_stencil(5)
        self.dmvs["div"] = DivMakerVisitor(div)
        self.dmvs["D"] = DivMakerVisitor(D)
        for dmv in self.dmvs.values():
            dmv.params = self.mk_param_set()

    def get_free_indices(self, expr : Expr) -> OrderedSet[Idx]:
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

    def add_param(self, name: str, default: param_default_type, desc: str, values: param_values_type = None) -> Symbol:
        self.params[name] = Param(name, default, desc, values)
        return mkSymbol(name)

    def _add_sym(self, tens: Indexed, ix1: Idx, ix2: Idx, sgn: int = 1) -> None:
        base: IndexedBase = cast(IndexedBase, tens.args[0])
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

    def declfun(self, funname: str, args:int=1, is_stencil: bool=False) -> UFunc:
        fun = mkFunction(funname)
        self.fun_args[funname] = args
        # self.eqnlist.add_func(fun, is_stencil)
        self.is_stencil[fun] = is_stencil

        return fun

    def declscalar(self, basename: str) -> Symbol:
        ret = mkIndexedBase(basename, tuple())
        self.gfs[basename] = ret
        self.defn[basename] = (basename, list())

        assert isinstance(ret.args[0], Symbol)
        return ret.args[0]

    def mk_coords(self, with_time: bool = False) -> List[Symbol]:
        # Note that x, y, and z are special symbols
        if dimension == 3:
            if with_time:
                self.coords = [self.declscalar("t"), self.declscalar("x"), self.declscalar("y"), self.declscalar("z")]
            else:
                self.coords = [self.declscalar("x"), self.declscalar("y"), self.declscalar("z")]
        elif dimension == 4:
            # TODO: No idea whether this works
            self.coords = [self.declscalar("t"), self.declscalar("x"), self.declscalar("y"), self.declscalar("z")]
        else:
            assert False
        return self.coords

    def mk_param_set(self)->Set[Symbol]:
        ret: Set[Symbol] = set()
        for k in self.params:
            ret.add(mkSymbol(k))
        return ret

    def do_div(self, expr:Expr)->Expr:
        params = self.mk_param_set()
        r = expr
        for k, v in self.dmvs.items():
            v.params = params
            r = v.visit(r, noidx)
        return r

    @multimethod
    def mk_stencil(self, func_name:str, idx:Idx, expr:Expr)->UFunc:
        result = self.mk_stencil(func_name, expr, [idx])
        assert isinstance(result, UFunc)
        self.dmvs[func_name] = DivMakerVisitor(result)
        return result

    @mk_stencil.register
    def _(self, func_name:str, idxa1:Idx, idxa2:Idx, expra:Expr,
                               idxb1:Idx, idxb2:Idx, exprb:Expr)->UFunc:
        self.mk_stencil(func_name, idxa1, idxa2, expra)
        result = self.mk_stencil(func_name, idxb1, idxb2, exprb)
        assert isinstance(result, UFunc)
        return result

    @mk_stencil.register
    def _(self, func_name:str, idx1:Idx, idx2:Idx, expr:Expr)->UFunc:
        result = self.mk_stencil(func_name, expr, [idx1,idx2])
        assert isinstance(result, UFunc)
        self.dmvs[func_name] = DivMakerVisitor(result)
        return result

    @mk_stencil.register
    def _(self, func_name:str, expr:Expr, idxlist:List[Idx])->UFunc:

        @multimethod
        def mk_sten(idxmap:Dict[Idx,Idx], expr:sy.Function)->Expr:
            # TODO: Rewrite so it does not require 3 dimensions
            assert get_dimension() == 3
            if expr.func == stencil:
                if len(expr.args) != 1:
                    raise DslException(expr)
                arg = mk_sten(idxmap, expr.args[0])
                c0 = coef(l0, arg)
                c1 = coef(l1, arg)
                c2 = coef(l2, arg)
                ret = stencil(dummy, c0, c1, c2)
                assert isinstance(ret, Expr)
                return ret
            elif expr.func == DD:
                if len(expr.args) != 1:
                    raise DslException(expr)
                arg = mk_sten(idx, idx0, expr.args[0])
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
                arg = mk_sten(idxmap, expr.args[0])
                if arg == l0:
                    return DXI
                elif arg == l1:
                    return DYI
                elif arg == l2:
                    return DZI
                assert False
            else:
                raise DslException("Bad Func")

        @mk_sten.register
        def _(idxmap:Dict[Idx,Idx], expr:sy.Float)->Expr:
            return expr

        @mk_sten.register
        def _(idxmap:Dict[Idx,Idx], expr:sy.Integer)->Expr:
            return expr

        @mk_sten.register
        def _(idxmap:Dict[Idx,Idx], expr:sy.Rational)->Expr:
            return expr

        @mk_sten.register
        def _(idxmap:Dict[Idx,Idx], expr:sy.Pow)->Expr:
            result:Expr = Pow(mk_sten(idxmap,expr.args[0]),expr.args[1])
            return result

        @mk_sten.register
        def _(idxmap:Dict[Idx,Idx], expr:Idx)->Expr:
            retval = idxmap.get(expr, expr)
            return retval

        @mk_sten.register
        def _(idxmap:Dict[Idx,Idx], expr:sy.Add)->Expr:
            ret = zero
            for a in expr.args:
                term = mk_sten(idxmap, a)
                ret += term
            return ret

        @mk_sten.register
        def _(idxmap:Dict[Idx,Idx], expr:sy.Mul)->Expr:
            ret = one
            for a in expr.args:
                term = mk_sten(idxmap, a)
                ret *= term
            return ret

        func = mkFunction(func_name)
        repl: Dict[int,Expr] = dict()
        if len(idxlist)==1 or (len(idxlist)==2 and idxlist[0] == idxlist[1]):
            idx = idxlist[0]
            is_down_idx = is_down(idx)
            for i in range(get_dimension()):
                if is_down_idx:
                    idx0 = down_indices[i]
                else:
                    idx0 = up_indices[i]
                result = mk_sten({idx: idx0}, expr)
                self.funs1[(func, idx0)] = result
        elif len(idxlist)==2:
            idx1 = idxlist[0]
            idx2 = idxlist[1]
            is_down_idx1 = is_down(idx1)
            is_down_idx2 = is_down(idx2)
            for i in range(get_dimension()):
                if is_down_idx1:
                    idx10 = down_indices[i]
                else:
                    idx10 = up_indices[i]
                for j in range(get_dimension()):
                    if i==j:
                        continue
                    if is_down_idx2:
                        idx20 = down_indices[j]
                    else:
                        idx20 = up_indices[j]
                    result = mk_sten({idx1: idx10, idx2: idx20}, expr)
                    self.funs2[(func, idx10, idx20)] = result

        return mkFunction(func_name)

    class DeclOptionalArgs(TypedDict, total=False):
        centering: Centering
        declare_as_temp: bool
        rhs: IndexedBase
        from_thorn: str
        parity: TensorParity
        group_name: str
        symmetries: List[Tuple[Idx, Idx]]
        anti_symmetries: List[Tuple[Idx, Idx]]

    def get_state(self)->List[IndexedBase]:
        result : List[IndexedBase] = list()
        for k in self.rhs:
            result.append(self.gfs[k])
        return result

    def decl(self, basename: str, indices: List[Idx], **kwargs: Unpack[DeclOptionalArgs]) -> IndexedBase:
        if (rhs := kwargs.get('rhs', None)) is not None:
            base_sym = rhs.args[0]
            assert isinstance(base_sym, Symbol)
            self.rhs[basename] = base_sym

        if (centering := kwargs.get('centering', None)) is None:
            centering = Centering.VVV

        the_symbol = mkIndexedBase(basename, shape=tuple([dimension] * len(indices)))
        indexed_symbol = mkIndexed(the_symbol, *tuple(indices))

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
            for sym in symmetries:
                self._add_sym(indexed_symbol, *sym, sgn=1)

        if (anti_symmetries := kwargs.get('anti_symmetries', None)) is not None:
            for a_sym in anti_symmetries:
                self._add_sym(indexed_symbol, *a_sym, sgn=-1)

        return the_symbol

    def find_indices(self, foo: Basic) -> List[Idx]:
        ret: List[Idx] = list()
        if type(foo) in [div, D]:
            ret = self.find_indices(foo.args[0])
        for arg in foo.args[1:]:
            assert isinstance(arg, Idx)
            ret.append(arg)
        return ret

    def find_symmetries(self, foo: Basic) -> List[Tuple[int, int, int]]:
        msym_list: List[Tuple[int, int, int]] = list()
        if foo.is_Function and hasattr(foo, "name") and foo.name in ["div", "D"]:
            # This is a derivative
            if len(foo.args) == 3:
                # This is a 2nd derivative, symmetric in the last 2 args
                foo_arg1 = len(foo.args[0].args) - 1  # chkcast(foo.args[1], int)
                foo_arg2 = foo_arg1 + 1  # chkcast(foo.args[2], int)
                msym: Tuple[int, int, int] = (foo_arg1, foo_arg2, 1)
                msym_list += [msym]
                msym_list += self.find_symmetries(foo.args[0])
            elif len(foo.args) == 2:
                msym_list += self.find_symmetries(foo.args[0])
            else:
                assert False, "Only handle 1st and 2nd derivs"
        elif isinstance(foo, Indexed):
            k = foo.base
            return self.symmetries.sd.get(k, list())
        return msym_list

    def get_matrix(self, ind: Indexed) -> Matrix:
        values: Dict[Idx, Idx] = dict()
        result = mkZeros(*tuple([dimension] * (len(ind.args) - 1)))
        ind_args: List[Idx] = [chkcast(x, Idx) for x in ind.args[1:]]
        while incr(ind_args, values):
            arr_inds = tuple([to_num(chkcast(do_subs(x, values), Idx)) for x in ind_args])
            r = self.do_subs(ind, idxsubs=values)
            result[arr_inds] = r
        return result

    def get_indices(self, expr: Expr) -> List[Idx]:
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
    def mk_subst(self, indexed: Indexed, f: Callable[[Indexed, int, int], Expr]) -> None:
        def f2(ix: Indexed, *n:int)->Expr:
            return f(ix, n[0], n[1])
        self.mk_subst(indexed, f2)

    @mk_subst.register
    def _(self, indexed: Indexed, f: Callable[[Indexed, int], Expr]) -> None:
        def f2(ix: Indexed, *n:int)->Expr:
            return f(ix, n[0])
        self.mk_subst(indexed, f2)

    @mk_subst.register
    def _(self, indexed: Indexed, f: Callable[[Indexed, int, int, int], Expr]) -> None:
        def f2(ix: Indexed, *n:int)->Expr:
            return f(ix, n[0], n[1], n[2])
        self.mk_subst(indexed, f2)

    @mk_subst.register
    def _(self, indexed: Indexed, f: mk_subst_type = mk_subst_default) -> None:
        indices: List[Idx]
        iter_var = indexed
        iter_syms = self.find_symmetries(indexed)
        indices = self.find_indices(indexed)

        for tup in expand_free_indices(iter_var, self.symmetries):
            out, indrep, _ = tup
            assert isinstance(out, Indexed)
            inds = out.indices
            subj: Expr
            subj = out
            subval_ = f(subj, *inds)

            if subval_.is_Number:
                pass
            elif subval_.is_Function:
                pass
            else:
                subval = str(subval_)
                outstr = str(out.base)
                assert isinstance(subval_, Symbol)
                self.gfs[subval] = mkIndexedBase(subval, tuple())
                self.centering[subval] = self.centering[outstr]
                self.var2base[subval] = outstr
                if outstr not in self.groups:
                    self.groups[outstr] = list()
                members = self.groups[outstr]
                members.append(subval)
            print(colorize(subj, "red"), colorize("->", "magenta"), colorize(subval, "cyan"))
            self.subs[subj] = subval_

    @mk_subst.register
    def _(self, indexed: Indexed, f: Expr) -> None:
        indices: List[Idx]
        iter_var = indexed
        iter_syms = self.find_symmetries(indexed)
        indices = self.find_indices(indexed)

        if self.get_free_indices(iter_var) != self.get_free_indices(f):
            raise Exception(f"Free indices of '{indexed}' and '{f}' do not match.")
        for tup in expand_free_indices(iter_var, self.symmetries):
            out, indrep, _ = tup
            assert isinstance(out, Indexed)
            arr_inds = tuple([to_num(x) for x in out.indices])
            if self.run_simplify:
                self.subs[out] = do_simplify(self.do_subs(f, idxsubs=indrep))
            else:
                self.subs[out] = self.do_subs(f, idxsubs=indrep)
            print(colorize(out, "red"), colorize("->", "magenta"), colorize(self.subs[out], "cyan"))
        return None

    @mk_subst.register
    def _(self, indexed: Indexed, f: ImmutableDenseMatrix) -> None:
        self.mk_subst_matrix(indexed, f)

    @mk_subst.register
    def _(self, indexed: Indexed, f: MatrixBase) -> None:
        self.mk_subst_matrix(indexed, f)

    def mk_subst_matrix(self, indexed: Indexed, f: MatrixBase)->None:
        indices: List[Idx]
        iter_var = indexed
        iter_syms = self.find_symmetries(indexed)
        indices = self.find_indices(indexed)

        set_matrix = f
        for tup in expand_free_indices(iter_var, self.symmetries):
            out, indrep, alist = tup
            assert isinstance(out, Indexed)
            arr_inds = tuple([to_num(x) for x in out.indices])
            if self.run_simplify:
                narray = len(arr_inds)
                res = do_simplify(set_matrix[arr_inds[0:2]])
                if narray >= 3:
                    res = self.do_subs(res, idxsubs=indrep)
                self.subs[out] = res
            else:
                self.subs[out] = set_matrix[arr_inds]
            print(colorize(out, "red"), colorize("->", "magenta"), colorize(self.subs[out], "cyan"))
        return None

    def expand_eqn(self, eqn: Eq) -> List[Eq]:
        result: List[Eq] = list()
        for tup in expand_free_indices(eqn.lhs, self.symmetries):
            lhs, inds, _ = tup
            result += [mkEq(self.do_subs(lhs, self.subs), self.do_subs(eqn.rhs, self.subs, inds))]
        return result

    def do_subs(self, arg: Expr, subs: Optional[Dict[Indexed, Expr]]=None, idxsubs: Optional[Dict[Idx,Idx]]=None) -> Expr:
        if subs is None:
            subs = dict()
        if idxsubs is None:
            idxsubs = dict()
        isub = IndexSubsVisitor(self.subs)
        arg1 = arg
        for i in range(20):
            new_arg = arg1
            new_arg = expand_contracted_indices(new_arg, self.symmetries)
            new_arg = cast(Expr, self.symmetries.apply(new_arg))


            isub.idxsubs = idxsubs
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
        pars = [_parity_of(p) for p in args[i:i+3]]
        parities.append(SingleIndexParity(*pars))

    return TensorParity(parities)

if __name__ == "__main__":
    gf = ThornDef("ARR", "TST")
    B = gf.decl("B", [lc, lb])
    gf._add_sym(B[la, lb], la, lb, 1)
    M = gf.decl("M", [la, lb])
    gf._add_sym(M[la, lb], la, lb, -1)
    V = gf.decl("V", [la])
    A = gf.decl("A", [ub,la,lc])
    gf._add_sym(A[ua,lb,lc], lb, lc, 1)

    ####
    fail_expr = mkSymbol("fail_expr")
    def testerr(gf:ThornDef, in_expr:Expr, result_expr:Expr)->None:
        viz = IndexContractionVisitor(dict())
        try:
            expr, it = viz.visit(in_expr)
            expr = gf.do_subs(expr)
        except InvalidIndexError as iie:
            print(iie)
            it = IndexTracker()
            expr = fail_expr
        zero_expr = do_simplify(expr - result_expr)
        if zero_expr == 0:
            result_color : Literal['red', 'green', 'yellow', 'blue', 'magenta', 'cyan']
            if result_expr == fail_expr:
                result_color = "red"
            else:
                result_color = "green"
            print(colorize("success:","green"),in_expr,colorize("->","cyan"),colorize(result_expr,result_color))
        else:
            print(colorize("FAIL","red"))
            print(colorize(in_expr,"yellow"))
            print(colorize(result_expr,"cyan"))
            print(colorize(expr,"green"))
            print(colorize(it,"blue"))
            raise Exception(colorize("Fail","red"))
    testerr(gf, M[la,ub]*B[lb,uc], M[la,u0]*B[l0,uc] + M[la,u1]*B[l1,uc] + M[la,u2]*B[l2,uc])
    testerr(gf, M[la,ua]*B[ub,lb] ,(M[l0,u0]+M[l1,u1]+M[l2,u2])*(B[u0,l0]+B[u1,l1]+B[u2,l2]))
    testerr(gf, sqrt(M[la,ua])*B[ub,lb] ,sqrt(M[l0,u0]+M[l1,u1]+M[l2,u2])*(B[u0,l0]+B[u1,l1]+B[u2,l2]))
    testerr(gf, M[la,ua]*(1+B[ub,lb]), (M[l0,u0]+M[l1,u1]+M[l2,u2])*(1+B[u0,l0]+B[u1,l1]+B[u2,l2]))
    testerr(gf, M[la,ua]*B[la,ua],fail_expr)
    testerr(gf, M[ua,lb] + 1, fail_expr)
    testerr(gf, sqrt(V[ua]), fail_expr)
    testerr(gf, sqrt(V[ua]*V[la]), sqrt(V[u0]*V[l0]+V[u1]*V[l1]+V[u2]*V[l2]))
    testerr(gf, B[la,lb]*V[ua]*V[ub], B[l0,l0]*V[u0]**2 + B[l1,l1]*V[u1]**2 + B[l2,l2]*V[u2]**2 + 2*B[l0,l1]*V[u0]*V[u1] + 2*B[l1,l2]*V[u1]*V[u2] + 2*B[l0,l2]*V[u0]*V[u2])
    testerr(gf, M[la,lb]*V[ua]*V[ub], zero)
    testerr(gf, A[ua,lb,la]*V[ub], (A[u0, l0, l0] + A[u1, l0, l1] + A[u2, l0, l2])*V[u0] + (A[u0, l0, l1] + A[u1, l1, l1] + A[u2, l1, l2])*V[u1] + (A[u0, l0, l2] + A[u1, l1, l2] + A[u2, l2, l2])*V[u2])

    #testerr(gf, div(V[la],lb)*B[ua,ub],
    #    div(V[l0],l0)*B[u0,u0] + div(V[l1],l0)*B[u0,u1] + div(V[l2],l0)*B[u0,u2]+
    #    div(V[l0],l1)*B[u0,u1] + div(V[l1],l1)*B[u1,u1] + div(V[l2],l1)*B[u1,u2]+
    #    div(V[l0],l2)*B[u0,u2] + div(V[l1],l2)*B[u1,u2] + div(V[l2],l2)*B[u2,u2])
    testerr(gf, div(A[ua, l0, l0], la), div(A[u0, l0, l0], l0)+div(A[u1, l0, l0], l1)+div(A[u2, l0, l0], l2))
    testerr(gf, div(A[ua, la, l0], l0), div(A[u0, l0, l0], l0)+div(A[u1, l0, l1], l0)+div(A[u2, l0, l2], l0))
    ####

    # Anti-Symmetric

    n = 0
    for out in gf.expand_eqn(mkEq(M[la, lb], B[la, lb])):
        print(out)
        n += 1
    assert n == dimension, f"n = {n}"

    # Symmetric
    N = gf.decl("N", [la, lb])
    gf._add_sym(N[la, lb], la, lb, 1)

    n = 0
    for out in gf.expand_eqn(mkEq(N[la, lb], B[la, lb])):
        print(out)
        n += 1
    assert n == dimension * (dimension - 1)

    # Non-Symmetric
    Q = gf.decl("Q", [la, lb])

    n = 0
    for out in gf.expand_eqn(mkEq(Q[la, lb], B[la, lb])):
        print(out)
        n += 1
    assert n == dimension ** 2

    a = gf.decl("a", [], declare_as_temp=True)
    b = gf.decl("b", [])
    c = gf.decl("c", [])
    k = gf.decl("k", [la])
    gf.mk_subst(k[la])
    foofunc = gf.create_function("foo", ScheduleBin.Analysis)
    foofunc.add_eqn(a, do_sympify(dimension))
    foofunc.add_eqn(b, a + do_sympify(2))
    
    # Test of custom derivative operation mdiv
    mdiv = gf.mk_stencil("mdiv", la, (stencil(la)-stencil(0))*DDI(la))
    foofunc.add_eqn(k[la], mdiv(a**5*b,la))
    kd0eqn = foofunc.eqn_list.eqns.get(mkSymbol("kD0"),None)
    assert kd0eqn == 5*DXI*(-stencil(a, 0, 0, 0) + stencil(a, 1, 0, 0))*a**4*b + DXI*(-stencil(b, 0, 0, 0) + stencil(b, 1, 0, 0))*a**5

    def getsym(a: IndexedBase)->Symbol:
        b = a.args[0]
        assert isinstance(b, Symbol)
        return b

    # Now test functions
    fmax = gf.declfun("fmax", 2)
    foofunc.add_eqn(c, fmax(a,b))
    foofunc.bake()
    assert foofunc.eqn_list.depends_on(getsym(c), getsym(a))
    assert foofunc.eqn_list.depends_on(getsym(c), getsym(b))
