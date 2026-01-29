import typing
from collections import OrderedDict
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from typing import cast, Dict, List, Tuple, Optional, Set

from multimethod import multimethod
from nrpy.helpers.coloring import coloring_is_enabled as colorize
from sympy import Basic, IndexedBase, Expr, Symbol, Integer

from EmitCactus.dsl.dsl_exception import DslException
from EmitCactus.dsl.sympywrap import *
from EmitCactus.dsl.functions import *
from EmitCactus.dsl.util import require_baked
from EmitCactus.emit.ccl.schedule.schedule_tree import IntentRegion
from EmitCactus.generators.sympy_complexity import SympyComplexityVisitor, calculate_complexities
from EmitCactus.util import OrderedSet, incr_and_get, consolidate
from EmitCactus.util import get_or_compute

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


class EqnComplex:
    eqn_lists: list['EqnList']
    is_stencil: dict[UFunc, bool]
    been_baked: bool
    _tile_temporaries: set[Symbol]

    def __init__(self, is_stencil: Dict[UFunc, bool]) -> None:
        self.is_stencil = is_stencil
        self.eqn_lists = [EqnList(self, is_stencil)]
        self.been_baked = False
        self._tile_temporaries = OrderedSet()

    def new_eqn_list(self) -> 'EqnList':
        new_list = EqnList(self, self.is_stencil)
        self.eqn_lists.append(new_list)
        return new_list

    def bake(self) -> None:
        if self.been_baked:
            raise DslException("Can't bake an EqnComplex that has already been baked.")
        self.been_baked = True

        for eqn_list in self.eqn_lists:
            eqn_list.bake()

    def get_active_eqn_list(self) -> 'EqnList':
        return self.eqn_lists[-1]

    def _grid_variables(self) -> set[Symbol]:
        gv: set[Symbol] = set()
        for eqn_list in self.eqn_lists:
            gv |= eqn_list._grid_variables()
        return gv

    def do_madd(self) -> None:
        for eqn_list in self.eqn_lists:
            eqn_list.madd()

    def do_cse(self) -> None:
        old_shape: list[int] = list()
        old_lhses: list[Symbol] = list()
        old_rhses: list[Expr] = list()

        for el in self.eqn_lists:
            old_shape.append(0)
            for lhs, rhs in el.eqns.items():
                old_lhses.append(lhs)
                old_rhses.append(rhs)
                old_shape[-1] += 1

        substitutions_list: list[tuple[Symbol, Expr]]
        new_rhses: list[Expr]
        substitutions_list, new_rhses = cse(old_rhses)

        substitutions = {lhs: rhs for lhs, rhs in substitutions_list}
        substitutions_order = {lhs: idx for idx, (lhs, _) in enumerate(substitutions_list)}

        new_temp_reads: dict[Symbol, set[int]] = {sym: set() for sym in substitutions.keys()}
        new_temp_dependencies: dict[Symbol, set[Symbol]] = {sym: set() for sym in substitutions.keys()}


        # We need to figure out exactly which loops use which temporaries.
        # By doing this, we can determine which temporaries need to be promoted to tile temporaries and which loop each
        #  temporary should be computed in.
        # We will also populate the temporary-related bookkeeping fields on EqnList and EqnComplex.

        global_eqn_idx = 0
        for el_idx, el_shape in enumerate(old_shape):
            eqn_list = self.eqn_lists[el_idx]
            el_new_free_symbols: set[Symbol] = set(chain(*[free_symbols(rhs) for rhs in new_rhses[global_eqn_idx:global_eqn_idx + el_shape]]))
            new_temps = el_new_free_symbols.intersection(substitutions.keys())

            for new_temp, temp_rhs in [(new_temp, substitutions[new_temp]) for new_temp in new_temps]:
                assert new_temp not in eqn_list.inputs
                assert new_temp not in eqn_list.params
                assert new_temp not in eqn_list.outputs
                assert new_temp not in eqn_list.eqns

                new_temp_reads[new_temp].add(el_idx)

                # Temps might be substituted for expressions which contain other temps.
                # We need to recursively check the RHSes to ensure we compute the dependencies in the appropriate loops.
                def drill(lhs: Symbol, rhs: Expr) -> None:
                    temp_dependencies = free_symbols(rhs).intersection(substitutions.keys())
                    assert lhs not in temp_dependencies
                    for td in temp_dependencies:
                        new_temp_dependencies[lhs].add(td)
                        drill(td, substitutions[td])

                drill(new_temp, temp_rhs)

            for lhs in old_lhses[global_eqn_idx:global_eqn_idx + el_shape]:
                assert lhs in eqn_list.eqns
                eqn_list.eqns[lhs] = new_rhses[global_eqn_idx]
                global_eqn_idx += 1

            eqn_list.uncse()  # todo: Kept this around from the previous implementation, but do we need it anymore?

        for new_temp, temp_dependencies in sorted(new_temp_dependencies.items(),
                                                  key=lambda kv: substitutions_order[kv[0]],
                                                  reverse=True):
            el_idx = min(new_temp_reads[new_temp])
            for td in temp_dependencies:
                new_temp_reads[td].add(el_idx)

        for new_temp, el_list in new_temp_reads.items():
            if (seen_count := len(el_list)) == 0:
                continue

            primary_el = self.eqn_lists[primary_idx := min(el_list)]
            primary_el.add_eqn(new_temp, substitutions[new_temp])

            if seen_count == 1:
                primary_el.temporaries.add(new_temp)
            else:
                self._tile_temporaries.add(new_temp)
                primary_el.uninitialized_tile_temporaries.add(new_temp)
                for eqn_list in [self.eqn_lists[el_idx] for el_idx in el_list if el_idx != primary_idx]:
                    eqn_list.preinitialized_tile_temporaries.add(new_temp)

    def dump(self) -> None:
        for idx, eqn_list in enumerate(self.eqn_lists):
            print(f'=== Loop {idx} ===')
            eqn_list.dump()
            print()

    def recycle_temporaries(self) -> None:
        for eqn_list in self.eqn_lists:
            eqn_list.recycle_temporaries()

    def split_output_eqns(self) -> None:
        for eqn_list in self.eqn_lists:
            eqn_list.split_output_eqns()

    @property
    @require_baked(msg="Can't get tile_temporaries before baking the EqnComplex.")
    def tile_temporaries(self) -> set[Symbol]:
        assert hasattr(self, '_tile_temporaries')
        return self._tile_temporaries

    @cached_property
    @require_baked(msg="Can't get inputs before baking the EqnComplex.")
    def inputs(self) -> set[Symbol]:
        ret: set[Symbol] = OrderedSet()
        for eqn_list in self.eqn_lists:
            ret |= eqn_list.inputs
        return ret

    @cached_property
    @require_baked(msg="Can't get outputs before baking the EqnComplex.")
    def outputs(self) -> set[Symbol]:
        ret: set[Symbol] = OrderedSet()
        for eqn_list in self.eqn_lists:
            ret |= eqn_list.outputs
        return ret

    @cached_property
    @require_baked(msg="Can't get temporaries before baking the EqnComplex.")
    def temporaries(self) -> set[Symbol]:
        ret: set[Symbol] = OrderedSet()
        for eqn_list in self.eqn_lists:
            ret |= eqn_list.temporaries
        return ret

    @cached_property
    @require_baked(msg="Can't get read_decls before baking the EqnComplex.")
    def read_decls(self) -> dict[Symbol, IntentRegion]:
        ret: dict[Symbol, IntentRegion] = OrderedDict()
        for eqn_list in self.eqn_lists:
            consolidate(ret, eqn_list.read_decls, lambda r1, r2: r1.consolidate(r2))
        return ret

    @cached_property
    @require_baked(msg="Can't get write_decls before baking the EqnComplex.")
    def write_decls(self) -> dict[Symbol, IntentRegion]:
        ret: dict[Symbol, IntentRegion] = OrderedDict()
        for eqn_list in self.eqn_lists:
            consolidate(ret, eqn_list.write_decls, lambda r1, r2: r1.consolidate(r2))
        return ret

    @cached_property
    @require_baked(msg="Can't get variables before baking the EqnComplex.")
    def variables(self) -> set[Symbol]:
        ret: set[Symbol] = OrderedSet()
        for eqn_list in self.eqn_lists:
            ret |= eqn_list.variables
        return ret

    @cached_property
    @require_baked(msg="Can't get stencil_limits before baking the EqnComplex.")
    def stencil_limits(self) -> tuple[int, int, int]:
        result = [0, 0, 0]

        for eqn_list in self.eqn_lists:
            for eqn_rhs in eqn_list.eqns.values():
                # noinspection PyProtectedMember
                eqn_list._stencil_limits(result, eqn_rhs)

        return result[0], result[1], result[2]

    @cached_property
    @require_baked(msg="Can't get stencil_idxes before baking the EqnComplex.")
    def stencil_idxes(self) -> set[tuple[int, int, int]]:
        result: set[tuple[int, int, int]] = set()

        for eqn_list in self.eqn_lists:
            for eqn_rhs in eqn_list.eqns.values():
                # noinspection PyProtectedMember
                eqn_list._stencil_idxes(result, eqn_rhs)

        return result


class EqnList:
    """
    This class models a generic list of equations. As such, it knows nothing about the rest of EmitCactus.
    Ultimately, the information in this class will be used to generate a loop to be output by EmitCactus.
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

    def __init__(self, parent: EqnComplex, is_stencil: Dict[UFunc, bool]) -> None:
        self.eqns: Dict[Symbol, Expr] = dict()
        self.params: Set[Symbol] = OrderedSet()
        self.inputs: Set[Symbol] = OrderedSet()
        self.outputs: Set[Symbol] = OrderedSet()
        self.order: List[Symbol] = list()
        self.sublists: List[List[Symbol]] = list()
        self.verbose = True
        self.read_decls: Dict[Symbol, IntentRegion] = OrderedDict()
        self.write_decls: Dict[Symbol, IntentRegion] = OrderedDict()
        # TODO: need a better default
        self.default_read_write_spec: IntentRegion = IntentRegion.Everywhere  # Interior
        self.is_stencil: Dict[UFunc, bool] = is_stencil
        self.temporaries: Set[Symbol] = OrderedSet()
        self.uninitialized_tile_temporaries: Set[Symbol] = OrderedSet()
        self.preinitialized_tile_temporaries: Set[Symbol] = OrderedSet()
        self.temporary_replacements: Set[TemporaryReplacement] = OrderedSet()
        self.split_lhs_prime_count: Dict[Symbol, int] = dict()
        self.provides: Dict[Symbol, Set[Symbol]] = dict()  # vals require key
        self.requires: Dict[Symbol, Set[Symbol]] = dict()  # key requires vals
        self.been_baked: bool = False
        self.parent = parent
        self.complexity: dict[Symbol, int] = dict()

        # The modeling system treats these special
        # symbols as parameters.
        self.add_param(DXI)
        self.add_param(DYI)
        self.add_param(DZI)

    #@cached_property
    @property
    @require_baked(msg="Can't get variables before baking the EqnList.")
    def variables(self) -> Set[Symbol]:
        return self.inputs | self.outputs | self.temporaries

    #@cached_property
    @property
    @require_baked(msg="Can't get sorted_eqns before baking the EqnList.")
    def sorted_eqns(self) -> list[tuple[Symbol, Expr]]:
        return sorted(self.eqns.items(), key=lambda kv: self.order.index(kv[0]))

    def _grid_variables(self) -> set[Symbol]:
        return {s for s in (self.inputs | self.outputs) if str(s) not in {'t', 'x', 'y', 'z', 'DXI', 'DYI', 'DZI'}}

    #@cached_property
    @property
    @require_baked(msg="Can't get grid_variables before baking the EqnList.")
    def grid_variables(self) -> set[Symbol]:
        return self._grid_variables()

    def add_param(self, lhs: Symbol) -> None:
        assert lhs not in self.outputs, f"The symbol '{lhs}' is already in outputs"
        assert lhs not in self.inputs, f"The symbol '{lhs}' is already in outputs"
        self.params.add(lhs)

    @multimethod
    def add_input(self, lhs: Symbol) -> None:
        # TODO: Automatically assign temps?
        return
        assert lhs not in self.outputs, f"The symbol '{lhs}' is already in outputs"
        if lhs in self.outputs:
            self.temporaries.add(lhs)
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
        # assert lhs not in self.inputs, f"The symbol '{lhs}' is already in outputs"
        return
        if lhs in self.inputs:
            self.temporaries.add(lhs)
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
        self._run_complexity_analysis(new_lhs)
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
        self._run_complexity_analysis(target_lhs)

        for sub_lhs, sub_rhs in subexpressions.items():
            self._prepend_split_subeqn(target_lhs, sub_lhs, sub_rhs)

    def split_output_eqns(self) -> None:
        for output in self.outputs:
            self.split_eqn(output)

    def recycle_temporaries(self) -> None:
        temp_reads: Dict[Symbol, OrderedSet[int]] = OrderedDict()
        temp_writes: Dict[Symbol, OrderedSet[int]] = OrderedDict()

        local_temporaries = self.temporaries - self.parent.tile_temporaries

        for lhs, rhs in self.eqns.items():
            eqn_i = self.order.index(lhs)

            if lhs in local_temporaries:
                get_or_compute(temp_writes, lhs, lambda _: OrderedSet()).add(eqn_i)

            if len(temps_read := free_symbols(rhs).intersection(local_temporaries)) > 0:
                temp_var: Symbol
                for temp_var in temps_read:
                    get_or_compute(temp_reads, temp_var, lambda _: OrderedSet()).add(eqn_i)

        lifetimes: Set[TemporaryLifetime] = OrderedSet()

        for temp_var in local_temporaries:
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

            stale_temporaries: List[TemporaryLifetime] = sorted(filter(is_stale, lifetimes),
                                                                key=lambda lt: lt.final_read)

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

    def uses_dict(self) -> Dict[Symbol, int]:
        uses: Dict[Symbol, int] = dict()
        for k, v in self.eqns.items():
            for k2 in free_symbols(v):
                old = uses.get(k2, 0)
                uses[k2] = old + 1
        return uses

    def apply_order(self, k: Symbol, provides: Dict[Symbol, Set[Symbol]], requires: Dict[Symbol, Set[Symbol]]) -> List[Symbol]:
        result = list()
        if k not in self.params and k not in self.inputs and k not in self.preinitialized_tile_temporaries:
            self.order.append(k)
        for v in provides.get(k, set()):
            req = requires[v]
            if k in req:
                req.remove(k)
            if len(req) == 0:
                result.append(v)
        return result

    def order_builder(self, complete: Dict[Symbol, int], cno: int) -> None:
        provides: Dict[Symbol, Set[Symbol]] = OrderedDict()  # vals require key
        requires: Dict[Symbol, Set[Symbol]] = OrderedDict()  # key requires vals
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
        for k, v2 in requires.items():
            if len(v2) == 0:
                result += self.apply_order(k, provides, requires)
                complete[k] = cno
        for k in self.inputs:
            result += self.apply_order(k, provides, requires)
            complete[k] = cno
        for k in self.params:
            result += self.apply_order(k, provides, requires)
            complete[k] = cno
        for k in self.preinitialized_tile_temporaries:
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
        for k, v2 in requires.items():
            for vv in v2:
                if vv not in self.params and vv not in self.preinitialized_tile_temporaries:
                    raise DslException(f"Unsatisfied {k} <- {vv} : {self.params}")
        self.provides = provides

    def _run_preliminary_complexity_analysis(self) -> None:
        grid_vars = self._grid_variables()
        complexity_visitor = SympyComplexityVisitor(lambda s: s in grid_vars)
        for lhs, rhs in self.eqns.items():
            self.complexity[lhs] = complexity_visitor.complexity(rhs)

    def _run_main_complexity_analysis(self) -> None:
        complexity_visitor = SympyComplexityVisitor(lambda s: s in self._grid_variables())
        for lhs, rhs in self.eqns.items():
            self.complexity[lhs] = complexity_visitor.complexity(rhs)

    def _run_complexity_analysis(self, *lhses: Symbol) -> None:
        complexity_visitor = SympyComplexityVisitor(lambda s: s in self._grid_variables())
        for lhs in lhses:
            self.complexity[lhs] = complexity_visitor.complexity(self.eqns[lhs])

    def bake(self, *, force_rebake: bool = False) -> None:
        """ Discover inconsistencies and errors in the param/input/output/equation sets. """
        if self.been_baked and not force_rebake:
            raise DslException("Can't bake an EqnList that has already been baked.")
        self.been_baked = True

        rd_overwrites = OrderedSet()
        wr_overwrites = OrderedSet()
        def process_overwrite(s:Symbol)->None:
            if "'" in str(s):
                rd = mkSymbol(str(s).replace("'", ""))
                wr = s
                rd_overwrites.add(rd)
                wr_overwrites.add(wr)

        # Bake now regenerates inputs and outputs but not parameters
        self.inputs.clear()
        self.outputs.clear()
        self.temporaries.clear()
        for lhs, rhs in self.eqns.items():
            assert lhs not in self.params, f"Symbol '{lhs}' is a parameter, but we are assigning to it."
            self.outputs.add(lhs)
            process_overwrite(lhs)
            for symb in rhs.free_symbols:

                if symb not in self.params:
                    assert isinstance(symb, Symbol), f"{symb} should be an instance of Symbol, but type={type(symb)}"
                    self.inputs.add(symb)
                    process_overwrite(symb)

        for lhs in self.outputs:
            if lhs in self.inputs:
                self.temporaries.add(lhs)
        for lhs in self.temporaries:
            self.inputs.remove(lhs)
            self.outputs.remove(lhs)

        for rd in rd_overwrites:
            if rd in self.outputs:
                raise DslException(f"Overwrite source symbol {rd} should not be in outputs")
            if rd in self.temporaries:
                raise DslException(f"Overwrite source symbol {rd} should not be in temporaries")

        for wr in wr_overwrites:
            if wr in self.inputs:
                raise DslException(f"Overwrite destination symbol {wr} should not be in inputs")
            if wr in self.temporaries:
                raise DslException(f"Overwrite destination symbol {wr} should not be in temporaries")

        needed: Set[Symbol] = OrderedSet()
        complete: Dict[Symbol, int] = OrderedDict()
        self.order = list()

        read: Set[Symbol] = OrderedSet()
        written: Set[Symbol] = OrderedSet()

        self.read_decls.clear()
        self.write_decls.clear()

        self.lhs: Symbol

        for temp in self.temporaries:
            if temp in self.outputs:
                self.outputs.remove(temp)
            if temp in self.inputs:
                self.inputs.remove(temp)

        def ftrace(sym: Symbol) -> bool:
            if sym.is_Function:
                # The noop function should be treated
                # mathematically as parenthesis
                if str(sym.func) == "noop":
                    sym.args[0].replace(ftrace, noop)  # type: ignore[no-untyped-call]
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
            # With loop splitting, it can arise that an input symbol ends up in the RHS of a tile temp assigned
            #  in the previous loop, so we can just quietly fix the inconsistency.
            if k not in read:
                self.inputs.remove(k)
            assert k not in written, f"Symbol '{k}' is in inputs, but it is assigned to."

        for arg in self.inputs:
            assert isinstance(arg, Symbol), f"{arg}, type={type(arg)}"
            if arg not in self.read_decls:
                self.read_decls[arg] = self.default_read_write_spec

        for k in self.outputs:
            assert isinstance(k, Symbol)
            assert k in written, f"Symbol '{k}' is in outputs, but it is never written"

        for k in written:
            assert isinstance(k, Symbol)
            if k not in self.outputs and k not in self.uninitialized_tile_temporaries and k not in self.preinitialized_tile_temporaries:
                self.temporaries.add(k)

        for k in read:
            assert isinstance(k, Symbol), f"{k}, type={type(k)}"
            if k not in self.inputs and k not in self.params and k not in self.uninitialized_tile_temporaries  and k not in self.preinitialized_tile_temporaries:
                self.temporaries.add(k)

        if self.verbose:
            print(colorize("Temps:", "green"), self.temporaries)
            print(colorize("Uninitialized Tile Temps:", "green"), self.uninitialized_tile_temporaries)
            print(colorize("Preinitialized Tile Temps:", "green"), self.preinitialized_tile_temporaries)

        for k in self.temporaries:
            assert k in read, f"Temporary variable '{k}' is never read"
            assert k in written, f"Temporary variable '{k}' is never written"
            # assert k not in self.outputs, f"Temporary variable '{k}' in outputs"
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
            def __init__(self, outer: EqnList) -> None:
                self.outer = outer
                self.msg: Optional[str] = None

            def m(self, expr: Expr) -> bool:
                if expr.is_Function:
                    if self.outer.is_stencil.get(expr.func, False):
                        for arg in expr.args:
                            if arg in self.outer.temporaries:
                                self.msg = f"Temporary passed to stencil: call='{expr}' arg='{arg}'"
                            break  # only check the first arg
                return False

            def exc(self) -> None:
                if self.msg is not None:
                    raise Exception(self.msg)

            def r(self, expr: Expr) -> Expr:
                return expr

        fb = FindBad(self)
        for eqn in self.eqns.items():
            do_replace(eqn[1], fb.m, fb.r)
            fb.exc()

        self._run_main_complexity_analysis()

        for lhs in self.eqns:
            assert isinstance(lhs, Symbol), f"{lhs}, type={type(lhs)}"
            rhs = self.eqns[lhs]
            print(colorize("EQN:", "cyan"), lhs, colorize("=", "cyan"), rhs, " ", colorize(f"[complexity = {self.complexity[lhs]}]", "magenta"))

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
            def __init__(self, outer: EqnList) -> None:
                self.outer = outer
                self.value: Optional[Expr] = None

            def m(self, expr: Expr) -> bool:
                self.value = None
                if expr.is_Function and self.outer.is_stencil.get(expr.func, False) and len(expr.args) > 0:
                    arg = expr.args[0]
                    assert arg is not None
                    while arg in self.outer.temporaries:
                        assert arg is not None
                        assert isinstance(arg, Symbol)
                        print("UNDO:", arg, '->', end=' ')
                        arg = self.outer.eqns[arg]
                        print(arg)
                        args_list = list(expr.args)
                        args_list[0] = arg
                        args = tuple(args_list)
                        self.value = expr.func(*args)
                        print("UNDO:", expr, "->", self.value)
                return self.value is not None

            def r(self, expr: Expr) -> Expr:
                assert self.value is not None
                return self.value

        undo = UndoCSE(self)
        for eqn in self.eqns.items():
            self.eqns[eqn[0]] = do_replace(eqn[1], undo.m, undo.r)

    def madd(self) -> None:
        """ Insert fused multiply add instructions """
        p0 = mkWild("p0", exclude=[0, 1, 2, -1, -2])
        p1 = mkWild("p1", exclude=[0, 1, 2, -1, -2])
        p2 = mkWild("p2", exclude=[0])

        class make_madd:
            def __init__(self) -> None:
                self.value: Optional[Expr] = None

            def m(self, expr: Expr) -> bool:
                self.value = None
                g = do_match(expr, p0 * p1 + p2)
                if g:
                    q0, q1, q2 = g[p0], g[p1], g[p2]
                    self.value = muladd(self.repl(q0), self.repl(q1), self.repl(q2))
                return self.value is not None

            def r(self, expr: Expr) -> Expr:
                assert self.value is not None
                return self.value

            def repl(self, expr: Expr) -> Expr:
                for iter in range(20):
                    nexpr = do_replace(expr, self.m, self.r)
                    if nexpr == expr:
                        return nexpr
                    expr = nexpr
                return expr

        mm = make_madd()
        for k, v in self.eqns.items():
            self.eqns[k] = mm.repl(v)

    def stencil_limits(self) -> typing.Tuple[int, int, int]:
        result = [0, 0, 0]
        for eqn in self.eqns.values():
            self._stencil_limits(result, eqn)
        return result[0], result[1], result[2]

    def _stencil_limits(self, result: List[int], expr: Expr) -> None:
        for arg in expr.args:
            if str(type(arg)) == "stencil":
                for i in range(3):
                    ivar = arg.args[i + 1]
                    assert isinstance(ivar, Integer), f"ivar={ivar}, type={type(ivar)}"
                    result[i] = max(result[i], abs(int(ivar)))
            else:
                if isinstance(arg, Expr):
                    self._stencil_limits(result, arg)

    def stencil_idxes(self) -> set[tuple[int, int, int]]:
        result: set[tuple[int, int, int]] = set()
        for eqn in self.eqns.values():
            self._stencil_idxes(result, eqn)
        return result

    def _stencil_idxes(self, result: set[tuple[int, int, int]], expr: Expr) -> None:
        stencil_calls: set[Basic] = expr.find(lambda x: hasattr(x, 'func') and self.is_stencil.get(x.func, False))  # type: ignore[no-untyped-call]
        for call in stencil_calls:
            assert len(call.args) == 4, "Stencil function should have 4 arguments"
            result.add(tuple(int(typing.cast(Expr, a).evalf()) for a in call.args[1:]))  # type: ignore[arg-type, no-untyped-call]

    def dump(self) -> None:
        print(colorize("Dumping Equations:", "green"))
        for k in self.order:
            print(" ", colorize(k, "cyan"), "=", self.eqns[k])

    def depends_on(self, a: Symbol, b: Symbol) -> bool:
        """
        Dependency checker. Assumes no cycles.
        """
        for c in self.requires:
            if c == b:
                return True
            else:
                return self.depends_on(a, c)
        return False
