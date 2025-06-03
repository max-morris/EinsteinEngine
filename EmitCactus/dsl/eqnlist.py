import typing
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property

from nrpy.helpers.coloring import coloring_is_enabled as colorize
from sympy import symbols, Basic, IndexedBase, Expr, Symbol, Integer
from sympy.core.function import UndefinedFunction as UFunc, Function
from typing import cast, Dict, List, Tuple, Optional, Set
from EmitCactus.util import OrderedSet, incr_and_get

from EmitCactus.dsl.sympywrap import *
from EmitCactus.emit.ccl.schedule.schedule_tree import IntentRegion
from EmitCactus.util import get_or_compute, ProgressBar
from EmitCactus.dsl.dsl_exception import DslException

from multimethod import multimethod

# These symbols represent the inverse of the
# spatial discretization.
DXI = mkSymbol("DXI")
DYI = mkSymbol("DYI")
DZI = mkSymbol("DZI")
DX = mkSymbol("DX")
DY = mkSymbol("DY")
DZ = mkSymbol("DZ")


@dataclass
class TemporaryLifetime:
    symbol: Symbol
    prime: int
    read_at: OrderedSet[int]
    written_at: int
    replaces: Optional["TemporaryLifetime"]
    is_superseded: bool

    def __str__(self) -> str:
        ticks = "'" * self.prime
        return f'{self.symbol}{ticks}'

    def __hash__(self) -> int:
        return (self.symbol, self.prime).__hash__()

    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, TemporaryLifetime)
                and self.symbol.__eq__(__value.symbol)  # type: ignore[no-untyped-call]
                and self.prime.__eq__(__value.prime))

    @cached_property
    def final_read(self) -> int:
        return max(self.read_at)


@dataclass(frozen=True)
class TemporaryReplacement:
    old: Symbol
    new: Symbol
    begin_eqn: int
    end_eqn: int


class EqnList:
    """
    This class models a generic list of equations. As such, it knows nothing about the rest of NRPy+.
    Ultimately, the information in this class will be used to generate a loop to be output by NRPy+.
    All it knows are the following things:
    (1) params - These are quantities that are generated outside the loop.
    (2) inputs - These are quantities which are read by equations but never written by them.
    (3) outputs - These are quantities which are written by equations but never read by them.
    (4) equations - These relate inputs to outputs. These may contain temporary variables, i.e.
                    quantities that are both read and written by equations.

    This class can remove equations and parameters that are not needed, but will complain
    about inputs that are not needed. It can also detect errors in the classification of
    symbols as inputs/outputs/params.
    """

    def __init__(self, is_stencil: Dict[UFunc, bool]) -> None:
        self.eqns: Dict[Symbol, Expr] = dict()
        self.params: Set[Symbol] = OrderedSet()
        self.inputs: Set[Symbol] = OrderedSet()
        self.outputs: Set[Symbol] = OrderedSet()
        self.temps: Set[Symbol] = OrderedSet()
        self.order: List[Symbol] = list()
        self.sublists: List[List[Symbol]] = list()
        self.verbose = True
        self.read_decls: Dict[Symbol, IntentRegion] = dict()
        self.write_decls: Dict[Symbol, IntentRegion] = dict()
        # TODO: need a better default
        self.default_read_write_spec: IntentRegion = IntentRegion.Everywhere  #Interior
        self.is_stencil: Dict[UFunc, bool] = is_stencil
        self.temporaries: Set[Symbol] = OrderedSet()
        self.temporary_replacements: Set[TemporaryReplacement] = OrderedSet()
        self.split_lhs_prime_count: Dict[Symbol, int] = dict()
        self.provides : Dict[Symbol,Set[Symbol]] = dict() # vals require key
        self.requires : Dict[Symbol,Set[Symbol]] = dict() # key requires vals

        # The modeling system treats these special
        # symbols as parameters.
        self.add_param(DXI)
        self.add_param(DYI)
        self.add_param(DZI)

    @cached_property
    def variables(self) -> Set[Symbol]:
        return self.inputs | self.outputs | self.temporaries

    def add_param(self, lhs: Symbol) -> None:
        assert lhs not in self.outputs, f"The symbol '{lhs}' is already in outputs"
        assert lhs not in self.inputs, f"The symbol '{lhs}' is already in outputs"
        self.params.add(lhs)

    @multimethod
    def add_input(self, lhs: Symbol) -> None:
        # TODO: Automatically assign temps?
        #assert lhs not in self.outputs, f"The symbol '{lhs}' is already in outputs"
        if lhs in self.outputs:
            self.temps.add(lhs)
        assert lhs not in self.params, f"The symbol '{lhs}' is already in outputs"
        assert isinstance(lhs, Symbol)
        self.inputs.add(lhs)
    @add_input.register
    def _(self, lhs: IndexedBase) -> None:
        self.add_input(lhs.args[0])
    @add_input.register
    def _(self, lhs: Basic) -> None:
        raise DslException("bad input")

    def add_output(self, lhs: Symbol) -> None:
        # TODO: Automatically assign temps?
        #assert lhs not in self.inputs, f"The symbol '{lhs}' is already in outputs"
        if lhs in self.inputs:
            self.temps.add(lhs)
        assert lhs not in self.params, f"The symbol '{lhs}' is already in outputs"
        self.outputs.add(lhs)

    def add_eqn(self, lhs: Symbol, rhs: Expr) -> None:
        assert lhs not in self.eqns, f"Equation for '{lhs}' is already defined"
        self.eqns[lhs] = rhs

    def _prepend_split_subeqn(self, target_lhs: Symbol, new_lhs: Symbol, new_rhs: Expr) -> None:
        """
        Insert a new equation into the list. Said equation will represent one subexpression of another equation
        which it precedes.
        :param target_lhs: The LHS of the equation of which ``new_rhs`` is a subexpression.
        :param new_lhs: The LHS of the equation to be inserted.
        :param new_rhs: The RHS of the equation to be inserted.
        :return:
        """
        assert len(self.order) > 0, "Called prepend_split_subeqn before order was set."
        assert new_lhs not in self.eqns
        assert new_lhs not in self.order
        assert target_lhs in self.eqns
        assert target_lhs in self.order

        self.eqns[new_lhs] = new_rhs
        self.order.insert(self.order.index(target_lhs), new_lhs)
        self.temporaries.add(new_lhs)

    def _split_sympy_expr(self, lhs: Symbol, expr: Expr) -> Tuple[Expr, Dict[Symbol, Expr]]:
        subexpressions: Dict[Symbol, Expr] = OrderedDict()

        for subexpression in expr.args:
            subexpression_lhs = f'{lhs}_{incr_and_get(self.split_lhs_prime_count, lhs)}'
            subexpressions[Symbol(subexpression_lhs)] = typing.cast(Expr, subexpression)  # type: ignore[no-untyped-call]

        new_expr = expr.func(*subexpressions.keys())
        return new_expr, subexpressions

    def split_eqn(self, target_lhs: Symbol) -> None:
        assert target_lhs in self.eqns

        expr = self.eqns[target_lhs]

        # Can't split unary expression
        if len(expr.args) < 2:
            return

        # Can't split IndexedBase (it appears to have two args, but the second is an empty tuple)
        if isinstance(expr, IndexedBase):
            assert len(expr.args) == 2
            assert expr.args[1] == ()
            return

        new_rhs, subexpressions = self._split_sympy_expr(target_lhs, expr)
        self.eqns[target_lhs] = new_rhs

        for sub_lhs, sub_rhs in subexpressions.items():
            self._prepend_split_subeqn(target_lhs, sub_lhs, sub_rhs)

    def split_output_eqns(self) -> None:
        for output in self.outputs:
            self.split_eqn(output)

    def recycle_temporaries(self) -> None:
        temp_reads: Dict[Symbol, OrderedSet[int]] = OrderedDict()
        temp_writes: Dict[Symbol, OrderedSet[int]] = OrderedDict()

        for lhs, rhs in self.eqns.items():
            eqn_i = self.order.index(lhs)

            if lhs in self.temporaries:
                get_or_compute(temp_writes, lhs, lambda _: OrderedSet()).add(eqn_i)

            if len(temps_read := free_symbols(rhs).intersection(self.temporaries)) > 0:
                temp_var : Symbol
                for temp_var in temps_read:
                    get_or_compute(temp_reads, temp_var, lambda _: OrderedSet()).add(eqn_i)

        lifetimes: Set[TemporaryLifetime] = OrderedSet()

        for temp_var in self.temporaries:
            print(f'Temporary {temp_var}:')
            assert len(temp_writes[temp_var]) == 1

            reads_str = [str(x) for x in temp_reads[temp_var]]
            writes_str = [str(x) for x in temp_writes[temp_var]]

            print(f'    Read in EQNs: {", ".join(reads_str)}')
            print(f'    Written in EQNs: {", ".join(writes_str)}')

            lifetimes.add(TemporaryLifetime(
                symbol=temp_var,
                prime=0,
                read_at=temp_reads[temp_var],
                written_at=temp_writes[temp_var].pop(),
                replaces=None,
                is_superseded=False
            ))

        for eqn_i in range(len(self.order)):
            def is_assigned_here(lt: TemporaryLifetime) -> bool:
                return lt.written_at == eqn_i

            def is_stale(lt: TemporaryLifetime) -> bool:
                return lt.final_read < eqn_i and not lt.is_superseded

            if not (assigned_here := cast(TemporaryLifetime, next(filter(is_assigned_here, lifetimes), None))):
                continue

            stale_temporaries: List[TemporaryLifetime] = sorted(filter(is_stale, lifetimes), key=lambda lt: lt.final_read)

            if len(stale_temporaries) == 0:
                continue

            candidate = stale_temporaries[0]

            lifetimes.add(TemporaryLifetime(
                symbol=candidate.symbol,
                prime=candidate.prime + 1,
                read_at=assigned_here.read_at,
                written_at=eqn_i,
                replaces=assigned_here,
                is_superseded=False
            ))

            lifetimes.remove(assigned_here)
            candidate.is_superseded = True

            self.temporary_replacements.add(TemporaryReplacement(
                old=assigned_here.symbol,
                new=candidate.symbol,
                begin_eqn=eqn_i,
                end_eqn=assigned_here.final_read
            ))

            print(f'Will replace the declaration of {assigned_here.symbol} with reassignment to {candidate.symbol} in equation {eqn_i}.')

        print("*** Dumping temporary lifetimes ***")
        for lifetime in sorted(lifetimes, key=lambda lt: (str(lt.symbol), lt.prime)):
            print(f'{lifetime} [{lifetime.written_at}, {max(lifetime.read_at)}]')

    def uses_dict(self)->Dict[Symbol,int]:
        uses : Dict[Symbol,int] = dict()
        for k,v in self.eqns.items():
            for k2 in free_symbols(v):
                old = uses.get(k2,0)
                uses[k2] = old + 1
        return uses

    def apply_order(self, k:Symbol, provides:Dict[Symbol,Set[Symbol]], requires:Dict[Symbol,Set[Symbol]])->List[Symbol]:
        result = list()
        if k not in self.params and k not in self.inputs:
            self.order.append(k)
        for v in provides.get(k,set()):
            req = requires[v]
            if k in req:
                req.remove(k)
            if len(req) == 0:
                result.append(v)
        return result
            
    def order_builder(self, complete:Dict[Symbol, int], cno:int)->None:
        provides : Dict[Symbol,Set[Symbol]] = OrderedDict() # vals require key
        requires : Dict[Symbol,Set[Symbol]] = OrderedDict() # key requires vals
        self.requires = OrderedDict()
        # Thus for
        #   u_t = v
        #   v_t = div(u,la,lb) g[ua,ub]
        # provides = {v:{u_t}, u:{v_t}}
        # requires = {u_t:{v}, v_t:{u}}
        for k in self.eqns:
            if k not in requires:
                requires[k] = OrderedSet()
                self.requires[k] = OrderedSet()
            for v in free_symbols(self.eqns[k]):
                if v not in provides:
                    provides[v] = OrderedSet()
                provides[v].add(k)
                requires[k].add(v)
                self.requires[k].add(v)
        self.order = list()
        self.sublists = list()
        result = list()
        for k,v2 in requires.items():
            if len(v2) == 0:
                result += self.apply_order(k, provides, requires)
                complete[k] = cno
        for k in self.inputs:
            result += self.apply_order(k, provides, requires)
            complete[k] = cno
        for k in self.params:
            result += self.apply_order(k, provides, requires)
            complete[k] = cno
        self.sublists.append(result)
        while len(result) > 0:
            cno += 1
            new_result = list()
            for r in result:
                new_result += self.apply_order(r, provides, requires)
                complete[r] = cno
            if len(new_result) > 0:
                self.sublists.append(new_result)
            result = new_result
        for k,v2 in requires.items():
            for vv in v2:
                if vv not in self.params:
                    raise DslException(f"Unsatisfied {k} <- {vv} : {self.params}")
        self.provides = provides


    def bake(self) -> None:
        """ Discover inconsistencies and errors in the param/input/output/equation sets. """
        needed: Set[Symbol] = OrderedSet()
        complete: Dict[Symbol, int] = OrderedDict()
        self.order = list()

        read: Set[Symbol] = OrderedSet()
        written: Set[Symbol] = OrderedSet()

        self.read_decls.clear()
        self.write_decls.clear()

        self.lhs: Symbol

        # TODO: Automatically assign temps?
        for temp in self.temps:
            if temp in self.outputs:
                self.outputs.remove(temp)
            if temp in self.inputs:
                self.inputs.remove(temp)
                

        def ftrace(sym: Symbol) -> bool:
            if sym.is_Function:
                # The noop function should be treated
                # mathematically as parenthesis
                if str(sym.func) == "noop":
                    sym.args[0].replace(ftrace, noop) # type: ignore[no-untyped-call]
                elif self.is_stencil.get(sym.func, False):
                    for arg in sym.args:
                        if arg.is_Number:
                            continue
                        sym2 = cast(Symbol, arg)
                        self.read_decls[sym2] = IntentRegion.Everywhere
                        self.write_decls[self.lhs] = IntentRegion.Interior
                else:
                    for arg in sym.args:
                        if arg.is_Number:
                            continue
                        sym2 = cast(Symbol, arg)
                        self.read_decls[sym2] = IntentRegion.Everywhere
                        self.write_decls[self.lhs] = IntentRegion.Everywhere
            return False

        def noop(x: Symbol) -> Symbol:
            return x

        for lhs in self.eqns:
            self.lhs = lhs
            rhs = self.eqns[lhs]
            rhs.replace(ftrace, noop)  # type: ignore[no-untyped-call]

        for arg in self.inputs:
            assert isinstance(arg, Symbol), f"{arg}, type={type(arg)}"
            if arg not in self.read_decls:
                self.read_decls[arg] = self.default_read_write_spec

        for lhs in self.eqns:
            assert isinstance(lhs, Symbol), f"{lhs}, type={type(lhs)}"
            rhs = self.eqns[lhs]
            print(colorize("EQN:", "cyan"), lhs, colorize("=", "cyan"), rhs)

        if self.verbose:
            print(colorize("Inputs:", "green"), self.inputs)
            print(colorize("Outputs:", "green"), self.outputs)
            print(colorize("Params:", "green"), self.params)

        for k in self.eqns:
            assert isinstance(k, Symbol), f"{k}, type={type(k)}"
            written.add(k)
            for q in free_symbols(self.eqns[k]):
                read.add(q)

        if self.verbose:
            print(colorize("Read:", "green"), read)
            print(colorize("Written:", "green"), written)

        for k in self.inputs:
            assert isinstance(k, Symbol), f"{k}, type={type(k)}"
            if k not in read:
                for kk in self.eqns:
                    print(">>",kk,"=",self.eqns[kk])
            assert k in read, f"Symbol '{k}' is in inputs, but it is never read. {read}"
            assert k not in written, f"Symbol '{k}' is in inputs, but it is assigned to."

        for k in self.outputs:
            assert isinstance(k, Symbol)
            assert k in written, f"Symbol '{k}' is in outputs, but it is never written"

        for k in written:
            assert isinstance(k, Symbol)
            if k not in self.outputs:
                self.temporaries.add(k)

        for k in read:
            assert isinstance(k, Symbol), f"{k}, type={type(k)}"
            if k not in self.inputs and k not in self.params:
                self.temporaries.add(k)

        if self.verbose:
            print(colorize("Temps:", "green"), self.temporaries)

        for k in self.temporaries:
            assert k in read, f"Temporary variable '{k}' is never read"
            assert k in written, f"Temporary variable '{k}' is never written"
            #assert k not in self.outputs, f"Temporary variable '{k}' in outputs"
            assert k not in self.inputs, f"Temporary variable '{k}' in inputs"

        for k in read:
            assert k in self.inputs or self.params or self.temporaries, f"Symbol '{k}' is read, but it is not a temp, parameter, or input."

        self.order_builder(complete, 1)
        print(colorize("Order:", "green"), self.order)

        default_read_spec = IntentRegion.Interior
        default_write_spec = IntentRegion.Everywhere
        for var in self.inputs:
            for val in self.read_decls.values():
                if val == IntentRegion.Everywhere:
                    default_read_spec = val
                    break
        for key in self.read_decls.keys():
            self.read_decls[key] = default_read_spec
        for var in self.outputs:
            for val in self.write_decls.values():
                if val == IntentRegion.Interior:
                    default_write_spec = val
                    break
        for key in self.write_decls.keys():
            self.write_decls[key] = default_write_spec

        # Figure out the rest of the READ/WRITEs
        spec: IntentRegion
        for var in self.order:
            if var in self.inputs and var not in self.read_decls:
                if default_read_spec is None:
                    default_read_spec = IntentRegion.Everywhere
                self.read_decls[var] = default_read_spec
            elif var in self.outputs and var not in self.write_decls:
                if default_write_spec is None:
                    default_write_spec = IntentRegion.Everywhere
                spec = default_write_spec
                # if there are variables only valid in the interior on the RHS of the
                # equation where the var is assigned, then it must have spec = Interior
                for rvar in free_symbols(self.eqns[var]):
                    if self.read_decls.get(rvar, IntentRegion.Everywhere) == IntentRegion.Interior:
                        spec = IntentRegion.Interior
                        break
                self.write_decls[var] = spec
            elif var in self.temporaries and var in self.write_decls and var not in self.read_decls:
                # temporaries don't really have reads/writes, we figure out
                # what they would be just to connect the reads/writes of the inputs/outputs.
                self.read_decls[var] = self.write_decls[var]
            elif var in self.temporaries and var not in self.write_decls:
                if default_write_spec is None:
                    default_write_spec = IntentRegion.Everywhere
                spec = default_write_spec
                if spec != IntentRegion.Interior:
                    for rvar in free_symbols(self.eqns[var]):
                        if self.read_decls.get(rvar, IntentRegion.Everywhere) == IntentRegion.Interior:
                            spec = IntentRegion.Interior
                            break
                self.write_decls[var] = spec
                self.read_decls[var] = spec

        if self.verbose:
            print(colorize("READS:", "green"), end="")
            for var, spec in self.read_decls.items():
                if var in self.inputs:
                    print(" ", var, "=", colorize(repr(spec), "yellow"), sep="", end="")
            print()
            print(colorize("WRITES:", "green"), end="")
            for var, spec in self.write_decls.items():
                if var in self.outputs:
                    print(" ", var, "=", colorize(repr(spec), "yellow"), sep="", end="")
            print()

        for k, v in self.eqns.items():
            assert k in complete, f"Eqn '{k} = {v}' does not contribute to the output."
            val1: int = complete[k]
            for k2 in free_symbols(v):
                val2: Optional[int] = complete.get(k2, None)
                assert val2 is not None, f"k2={k2}"
                assert val1 >= val2, f"Symbol '{k}' is part of an assignment cycle."
        for k in needed:
            if k not in complete:
                print(f"Symbol '{k}' needed but could not be evaluated. Cycle in assignment?")
        for k in self.inputs:
            assert k in complete, f"Symbol '{k}' appears in inputs but is not complete"
        for k in self.eqns:
            assert k in complete, f"Equation '{k} = {self.eqns[k]}' is never complete"

        class FindBad:
            def __init__(self,outer:EqnList)->None:
                self.outer = outer
                self.msg: Optional[str] = None
            def m(self, expr:Expr)->bool:
                if expr.is_Function:
                    if self.outer.is_stencil.get(expr.func, False):
                        for arg in expr.args:
                            if arg in self.outer.temporaries:
                                self.msg = f"Temporary passed to stencil: call='{expr}' arg='{arg}'"
                            break # only check the first arg
                return False
            def exc(self)->None:
                if self.msg is not None:
                    raise Exception(self.msg)
            def r(self, expr:Expr)->Expr:
                return expr
        fb = FindBad(self)
        for eqn in self.eqns.items():
            do_replace(eqn[1], fb.m, fb.r)
            fb.exc()

    def trim(self) -> None:
        """ Remove temporaries of the form "a=b". They are clutter. """
        subs: Dict[Symbol, Symbol] = dict()
        for k, v in self.eqns.items():
            if v.is_symbol:
                # k is not not needed
                subs[k] = cast(Symbol, v)
                print(f"Warning: equation '{k} = {v}' can be trivially eliminated")

        new_eqns: Dict[Symbol, Expr] = dict()
        for k in self.eqns:
            if k not in subs:
                v = self.eqns[k]
                v2 = do_subs(v, subs)
                new_eqns[k] = v2

        self.eqns = new_eqns

    def uncse(self) -> None:
        print("Call UnCSE")
        class UndoCSE:
            def __init__(self, outer:EqnList)->None:
                self.outer = outer
                self.value:Optional[Expr] = None
            def m(self, expr:Expr)->bool:
                self.value = None
                if expr.is_Function and self.outer.is_stencil.get(expr.func, False) and len(expr.args)>0:
                    arg = expr.args[0]
                    assert arg is not None
                    while arg in self.outer.temporaries:
                        assert arg is not None
                        assert isinstance(arg, Symbol)
                        print("UNDO:",arg,'->',end=' ')
                        arg = self.outer.eqns[arg]
                        print(arg)
                        args_list = list(expr.args)
                        args_list[0] = arg
                        args = tuple(args_list)
                        self.value = expr.func(*args)
                        print("UNDO:",expr,"->",self.value)
                return self.value is not None
            def r(self, expr:Expr)->Expr:
                assert self.value is not None
                return self.value
        undo = UndoCSE(self)
        for eqn in self.eqns.items():
            self.eqns[eqn[0]] = do_replace(eqn[1], undo.m, undo.r)

    def madd(self) -> None:
        """ Insert fused multiply add instructions """
        muladd = mkFunction("muladd")
        p0 = mkWild("p0", exclude=[0, 1, 2, -1, -2])
        p1 = mkWild("p1", exclude=[0, 1, 2, -1, -2])
        p2 = mkWild("p2", exclude=[0])

        class make_madd:
            def __init__(self)->None:
                self.value:Optional[Expr] = None
            def m(self, expr:Expr)->bool:
                self.value = None
                g = do_match(expr, p0*p1+p2)
                if g:
                    q0, q1, q2 = g[p0], g[p1], g[p2]
                    self.value = muladd(self.repl(q0), self.repl(q1), self.repl(q2))
                return self.value is not None
            def r(self, expr:Expr)->Expr:
                assert self.value is not None
                return self.value
            def repl(self, expr:Expr)->Expr:
                for iter in range(20):
                    nexpr = do_replace(expr, self.m, self.r)
                    if nexpr == expr:
                        return nexpr
                    expr = nexpr
                return expr
        mm = make_madd()
        for k, v in self.eqns.items():
            self.eqns[k] = mm.repl(v)

    def cse(self) -> None:
        """ Invoke Sympy's CSE method, but ensure that the order of the resulting assignments is correct. """
        print("Call CSE")
        indexes: List[Symbol] = list()
        old_eqns: List[Expr] = list()
        for k in self.eqns:
            indexes.append(k)
            old_eqns.append(self.eqns[k])
        new_eqns, mod_eqns = cse(old_eqns)
        for eqn in new_eqns:
            self.temporaries.add(eqn[0])
        e: Tuple[Symbol, Expr]
        for e in new_eqns:
            assert e[0] not in self.inputs and e[0] not in self.params and e[0] not in self.eqns
            self.add_eqn(e[0], e[1])
        for i in range(len(indexes)):
            k = indexes[i]
            v = old_eqns[i]
            m = mod_eqns[i]
            self.eqns[k] = m
        self.uncse()

    def stencil_limits(self)->typing.Tuple[int,int,int]:
        result = [0,0,0]
        for eqn in self.eqns.values():
            self.stencil_limits_(result, eqn)
        return (result[0], result[1], result[2])

    def stencil_limits_(self, result:List[int], expr:Expr) -> None:
        for arg in expr.args:
            if str(type(arg)) == "stencil":
                for i in range(3):
                    ivar = arg.args[i+1]
                    assert isinstance(ivar, Integer), f"ivar={ivar}, type={type(ivar)}"
                    result[i] = max(result[i], abs(int(ivar)))
            else:
                if isinstance(arg, Expr):
                    self.stencil_limits_(result, arg)

    def dump(self) -> None:
        print(colorize("Dumping Equations:", "green"))
        for k in self.order:
            print(" ", colorize(k, "cyan"), "=", self.eqns[k])

    def depends_on(self, a:Symbol, b:Symbol) -> bool:
        """
        Dependency checker. Assumes no cycles.
        """
        for c in self.requires:
            if c == b:
                return True
            else:
                return self.depends_on(a, c)
        return False


if __name__ == "__main__":
    a, b, c, d, e, f, g, q, r = symbols("a b c d e f g q r")
    try:
        div = mkFunction("div")
        dmap: Dict[UFunc, bool] = {div: True}
        el = EqnList(dmap)
        el.default_read_write_spec = IntentRegion.Interior
        # el.add_func(div, True)
        el.add_input(a)
        el.add_eqn(c, do_sympify(3))
        el.add_eqn(b, 2 * c + do_sympify(8))
        el.add_eqn(f, div(a) + c)
        el.add_eqn(d, b + c + f)
        el.add_output(c)
        el.add_output(d)

        el.bake()
        assert el.read_decls[a] == IntentRegion.Everywhere
        assert el.write_decls[d] == IntentRegion.Interior
        assert el.write_decls[c] == IntentRegion.Interior

        el.default_read_write_spec = IntentRegion.Everywhere
        # el.add_func(div, True)
        el.bake()
        assert el.read_decls[a] == IntentRegion.Everywhere
        assert el.write_decls[d] == IntentRegion.Interior
        assert el.write_decls[c] == IntentRegion.Interior

        el.default_read_write_spec = IntentRegion.Everywhere
        # el.add_func(div, False)
        el.bake()
        assert el.read_decls[a] == IntentRegion.Everywhere
        #assert el.write_decls[d] == IntentRegion.Everywhere
        #assert el.write_decls[c] == IntentRegion.Everywhere
    finally:
        print("Test zero passed")
        print()

    try:
        el = EqnList(dict())
        el.add_eqn(r, q)  # cycle
        el.add_eqn(q, r)  # cycle
        el.add_eqn(a, r)
        el.add_output(a)
        el.bake()
    except DslException as ae:
        assert "Unsatisfied" in str(ae)
        print("Test one passed")
        print()

    try:
        el = EqnList(dict())
        el.add_input(a)
        el.add_input(f)
        el.add_input(b)
        el.add_output(d)
        el.add_eqn(c, g + f + a ** 3 + q)
        el.add_eqn(e, a ** 2)
        el.add_eqn(g, e)
        el.add_eqn(d, c * b + a ** 3)
        el.bake()
    except AssertionError as ae:
        assert "'q' is never written" in str(ae)
        print("Test two passed")
        print()

    try:
        el = EqnList(dict())
        el.add_input(a)
        el.add_input(f)
        el.add_output(d)
        el.add_eqn(a, f + 2)
        el.add_eqn(d, a)
        el.bake()
    except AssertionError as ae:
        assert "Symbol 'a' is in inputs, but it is assigned to." in str(ae)
        print("Test three passed")
        print()

    try:
        el = EqnList(dict())
        el.add_input(a)
        el.add_input(f)
        el.add_output(d)
        el.add_output(e)
        el.add_eqn(d, a + f)
        el.bake()
    except AssertionError as ae:
        assert "Symbol 'e' is in outputs, but it is never written" in str(ae)
        print("Test four passed")
        print()
