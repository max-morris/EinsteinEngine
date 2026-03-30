from functools import cache
import typing
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from functools import cached_property
from itertools import chain
from statistics import mean, median
from typing import cast, Dict, List, Tuple, Optional, Set, Iterator

from multimethod import multimethod
from nrpy.helpers.coloring import coloring_is_enabled as colorize
from sortedcontainers import SortedDict
from sympy import Basic, IndexedBase, Expr, Symbol, Integer
import sympy as sy

from EinsteinEngine.dsl.dsl_exception import DslException
from EinsteinEngine.dsl.eqn_ordering import maximize_symbol_reuse, EqnOrderingFn
from EinsteinEngine.dsl.stencil_idx import StencilIdxWithName, StencilIdx
from EinsteinEngine.dsl.sympywrap import *
from EinsteinEngine.dsl.functions import *
from EinsteinEngine.dsl.intent_override import IntentOverride
from EinsteinEngine.dsl.util import require_baked
from EinsteinEngine.emit.ccl.schedule.schedule_tree import IntentRegion
from EinsteinEngine.generators.sympy_complexity import SympyComplexityVisitor, calculate_complexities
from EinsteinEngine.util import OrderedSet, incr_and_get, consolidate, vprint, wprint, ProgressBarImpl, ProgressBar
from EinsteinEngine.util import get_or_compute
from EinsteinEngine.dsl.symbify import symbify
from EinsteinEngine.dsl.analytic_function_checker import AnalyticFunctionChecker

# These symbols represent the inverse of the
# spatial discretization.
DXI = mkSymbol("DXI")
DYI = mkSymbol("DYI")
DZI = mkSymbol("DZI")
DX = mkSymbol("DX")
DY = mkSymbol("DY")
DZ = mkSymbol("DZ")

stencil = mkFunction("stencil")

@dataclass
class TemporaryLifetime:
    symbol: Symbol
    prime: int
    read_at: OrderedSet[int]
    written_at: int
    replaces: Optional["TemporaryLifetime"]
    is_superseded: bool
    is_dead: bool

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
    intent_override: Optional[IntentOverride]
    been_baked: bool

    _tile_temporaries: set[Symbol]
    _inputs: set[Symbol]
    _outputs: set[Symbol]
    _temporaries: set[Symbol]
    _read_decls: dict[Symbol, IntentRegion]
    _write_decls: dict[Symbol, IntentRegion]
    _variables: set[Symbol]

    def __init__(self, is_stencil: Dict[UFunc, bool], intent_override: Optional[IntentOverride] = None) -> None:
        self.is_stencil = is_stencil
        self.intent_override = intent_override
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

    @cache
    def _calc_tile_temps(self) -> None:
        # Don't clear out self._tile_temporaries because it will already be populated by global_cse

        for temp in self.temporaries:
            written_el: Optional[int] = None
            read_els: set[int] = set()

            for el_idx, eqn_list in enumerate(self.eqn_lists):
                if temp in eqn_list.eqns:
                    written_el = el_idx
                    continue

                for lhs, rhs in eqn_list.eqns.items():
                    if temp in free_symbols(rhs):
                        read_els.add(el_idx)
                        break

            if written_el is not None and len(read_els) > 0:
                assert all(read_el > written_el for read_el in read_els), f"Determined {temp} should be a tile-temp in {self}, but it is written ({written_el}) after is is read ({read_els})"

                self._tile_temporaries.add(temp)
                self.eqn_lists[written_el].uninitialized_tile_temporaries.add(temp)
                for el_idx in read_els:
                    self.eqn_lists[el_idx].preinitialized_tile_temporaries.add(temp)

    @cache
    def _calc_vars(self) -> None:
        self._inputs = OrderedSet()
        self._outputs = OrderedSet()
        self._temporaries = OrderedSet()
        self._variables = OrderedSet()

        for eqn_list in self.eqn_lists:
            self._inputs |= eqn_list.inputs
            self._outputs |= eqn_list.outputs
            self._temporaries |= eqn_list.temporaries
            self._variables |= eqn_list.variables

        self._temporaries.update(self._inputs.intersection(self._outputs))
        self._inputs.difference_update(self._temporaries)
        self._outputs.difference_update(self._temporaries)

    @cache
    def _calc_decls(self) -> None:
        self._read_decls = OrderedDict()
        self._write_decls = OrderedDict()

        for eqn_list in self.eqn_lists:
            consolidate(self._read_decls, eqn_list.read_decls, lambda r1, r2: r1.consolidate(r2))
            consolidate(self._write_decls, eqn_list.write_decls, lambda r1, r2: r1.consolidate(r2))

        for t in self.temporaries:
            if t in self._read_decls:
                del self._read_decls[t]
            if t in self._write_decls:
                del self._write_decls[t]

    @property
    @require_baked(msg="Can't get tile_temporaries before baking the EqnComplex.")
    def tile_temporaries(self) -> set[Symbol]:
        assert hasattr(self, '_tile_temporaries')
        self._calc_tile_temps()
        return self._tile_temporaries

    @property
    @require_baked(msg="Can't get inputs before baking the EqnComplex.")
    def inputs(self) -> set[Symbol]:
        self._calc_vars()
        return self._inputs

    @property
    @require_baked(msg="Can't get outputs before baking the EqnComplex.")
    def outputs(self) -> set[Symbol]:
        self._calc_vars()
        return self._outputs

    @property
    @require_baked(msg="Can't get temporaries before baking the EqnComplex.")
    def temporaries(self) -> set[Symbol]:
        self._calc_vars()
        return self._temporaries

    @property
    @require_baked(msg="Can't get read_decls before baking the EqnComplex.")
    def read_decls(self) -> dict[Symbol, IntentRegion]:
        self._calc_decls()
        return self._read_decls

    @property
    @require_baked(msg="Can't get write_decls before baking the EqnComplex.")
    def write_decls(self) -> dict[Symbol, IntentRegion]:
        self._calc_decls()
        return self._write_decls

    @property
    @require_baked(msg="Can't get variables before baking the EqnComplex.")
    def variables(self) -> set[Symbol]:
        self._calc_decls()
        return self._variables

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
    def stencil_idxes(self) -> set[StencilIdxWithName]:
        result: set[StencilIdxWithName] = set()

        for eqn_list in self.eqn_lists:
            for eqn_rhs in eqn_list.eqns.values():
                # noinspection PyProtectedMember
                eqn_list._stencil_idxes(result, eqn_rhs)

        return result


class EqnList:
    """
    This class models a generic list of equations. As such, it knows nothing about the rest of EinsteinEngine.
    Ultimately, the information in this class will be used to generate a loop to be output by EinsteinEngine.
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
        self.order_clumping: dict[int, set[Symbol]] = dict()
        self.order_clumping_counter: int = 0
        self.sublists: List[List[Symbol]] = list()
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
        self.ordering_fn: EqnOrderingFn = maximize_symbol_reuse

        # The modeling system treats these special
        # symbols as parameters.
        self.add_param(DXI)
        self.add_param(DYI)
        self.add_param(DZI)

    def _order_tag(self) -> int:
        x = self.order_clumping_counter
        self.order_clumping_counter += 1
        return x

    @property
    def tile_temporaries(self) -> set[Symbol]:
        return self.uninitialized_tile_temporaries.union(self.preinitialized_tile_temporaries)

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
        # Ensure we only have symbols in eqnlist
        self.eqns[symbify(lhs)] = symbify(rhs)

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
            vprint(f'Temporary {temp_var}:')
            assert len(temp_writes[temp_var]) == 1

            reads_str = [str(x) for x in temp_reads[temp_var]]
            writes_str = [str(x) for x in temp_writes[temp_var]]

            vprint(f'    Read in EQNs: {", ".join(reads_str)}')
            vprint(f'    Written in EQNs: {", ".join(writes_str)}')

            lifetimes.add(TemporaryLifetime(
                symbol=temp_var,
                prime=0,
                read_at=temp_reads[temp_var],
                written_at=temp_writes[temp_var].pop(),
                replaces=None,
                is_superseded=False,
                is_dead=False
            ))

        lifetimes_assigned_at = {lt.written_at: lt for lt in lifetimes}
        lifetimes_final_read: SortedDict[int, OrderedSet[TemporaryLifetime]] = SortedDict()
        for lt in lifetimes:
            if lt.final_read in lifetimes_final_read:
                lifetimes_final_read[lt.final_read].add(lt)
            else:
                lifetimes_final_read[lt.final_read] = OrderedSet([lt])
        lifetimes_final_read_keys = list(lifetimes_final_read.keys())

        # Attempt to find a temporary lifetime that is stale (last read was before eqn_idx), not superseded, and not dead.
        def find_candidate(eqn_idx: int) -> Optional[TemporaryLifetime]:
            eqn_probe = eqn_idx
            while eqn_probe > 0:
                # In the sorted list of keys, find the index to insert `eqn_probe`. This will either give us the
                # index of `eqn_probe` itself if it's a valid key, or the index of the smallest key which is GT it.
                # If we get 0 back, we are either the first key in the list or smaller than all valid keys, so abort.
                if (key_idx := lifetimes_final_read.bisect_left(eqn_probe)) == 0:
                    return None

                # Subtract one from `key_idx` to get the next-smallest valid key.
                # Now, `eqn_probe` holds the next-smallest valid key from its previous value.
                eqn_probe = lifetimes_final_read_keys[key_idx - 1]

                assert eqn_probe < eqn_idx
                assert eqn_probe in lifetimes_final_read

                # Inspect the lifetimes which expired in eqn number `eqn_probe`. If we find a live one, return it.
                lt: TemporaryLifetime
                for lt in lifetimes_final_read[eqn_probe]:
                    if not lt.is_superseded and not lt.is_dead:
                        return lt

            return None



        for eqn_i in range(len(self.order)):
            if not (assigned_here := lifetimes_assigned_at.get(eqn_i, None)):
                continue

            if not (candidate := find_candidate(eqn_i)):
                continue

            lifetimes.add(TemporaryLifetime(
                symbol=candidate.symbol,
                prime=candidate.prime + 1,
                read_at=assigned_here.read_at,
                written_at=eqn_i,
                replaces=assigned_here,
                is_superseded=False,
                is_dead=False
            ))

            assigned_here.is_dead = True
            candidate.is_superseded = True

            self.temporary_replacements.add(TemporaryReplacement(
                old=assigned_here.symbol,
                new=candidate.symbol,
                begin_eqn=eqn_i,
                end_eqn=assigned_here.final_read
            ))

            vprint(f'Will replace the declaration of {assigned_here.symbol} with reassignment to {candidate.symbol} in equation {eqn_i}.')

        vprint("*** Dumping temporary lifetimes ***")
        for lifetime in filter(lambda lt: not lt.is_dead, sorted(lifetimes, key=lambda lt: (str(lt.symbol), lt.prime))):
            vprint(f'{lifetime} [{lifetime.written_at}, {max(lifetime.read_at)}]')

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

    def order_builder(self, complete: Dict[Symbol, int]) -> None:
        for k in self.inputs:
            complete[k] = 0
        for k in self.params:
            complete[k] = 0

        ordering_fn = self.ordering_fn
        myself = self

        class Ord:
            def __init__(self, eqns: dict[Symbol, Expr]) -> None:
                self.ord: list[Symbol] = list()
                self.eqns = eqns
            def add(self, sym: Symbol) -> None:
                if sym in complete:
                    return
                for dep in ordering_fn({dep: self.eqns[dep] for dep in free_symbols(self.eqns[sym]) if dep in self.eqns}, myself):
                    self.add(dep)
                self.ord.append(sym)
                complete[sym] = len(self.ord)

        ord = Ord(self.eqns)

        for sym in ordering_fn(self.eqns, self):
            ord.add(sym)
        self.order = ord.ord

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

        rd_overwrites: OrderedSet[Symbol] = OrderedSet()
        wr_overwrites: OrderedSet[Symbol] = OrderedSet()
        def process_overwrite(s: Symbol) -> None:
            if "'" in (ss := str(s)):
                rd = mkSymbol(ss.replace("'", ""))
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

        for rhs in self.eqns.values():
            if "stencil" in rhs.free_symbols:
                raise DslException(f"Overwrite source symbol {rd} cannot be used inside a stencil")

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

        for temp in self.temporaries:
            if temp in self.outputs:
                self.outputs.remove(temp)
            if temp in self.inputs:
                self.inputs.remove(temp)

        self.read_decls.clear()
        self.write_decls.clear()

        override_e2e = self.parent.intent_override is IntentOverride.E2E
        override_2i = self.parent.intent_override is IntentOverride.WriteInterior

        # Figure out the read/writes
        for lhs in self.inputs:
            self.read_decls[lhs] = IntentRegion.Everywhere if override_e2e else IntentRegion.Interior
        for lhs in self.outputs:
            self.write_decls[lhs] = IntentRegion.Everywhere if override_e2e else IntentRegion.Interior

        for lhs, rhs in self.eqns.items():
            for sten in rhs.find(stencil):  # type: ignore[no-untyped-call]
                if sten.args[1] != 0 or sten.args[2] != 0 or sten.args[3] != 0:
                    if override_e2e:
                        raise DslException(f"Stencil '{sten}' found in the RHS for {lhs} cannot have nonzero offset in E2E mode.")
                    var = sten.args[0]
                    self.read_decls[var] = IntentRegion.Everywhere


        if not override_2i:
            checker = AnalyticFunctionChecker(self.params, self.eqns)
            for lhs in checker.analytic():
                if lhs in self.outputs:
                    self.write_decls[lhs] = IntentRegion.Everywhere

        vprint(colorize("Inputs:", "green"), self.inputs)
        vprint(colorize("Outputs:", "green"), self.outputs)
        vprint(colorize("Params:", "green"), self.params)

        for k in self.eqns:
            assert isinstance(k, Symbol), f"{k}, type={type(k)}"
            written.add(k)
            for q in free_symbols(self.eqns[k]):
                read.add(q)

        vprint(colorize("Read:", "green"), read)
        vprint(colorize("Written:", "green"), written)

        for k in self.inputs:
            assert isinstance(k, Symbol), f"{k}, type={type(k)}"
            # With loop splitting, it can arise that an input symbol ends up in the RHS of a tile temp assigned
            #  in the previous loop, so we can just quietly fix the inconsistency.
            if k not in read:
                self.inputs.remove(k)
            assert k not in written, f"Symbol '{k}' is in inputs, but it is assigned to."

        for arg in self.inputs:
            assert isinstance(arg, Symbol), f"{arg}, type={type(arg)}"

        for k in self.outputs:
            assert isinstance(k, Symbol)
            assert k in written, f"Symbol '{k}' is in outputs, but it is never written"

        for k in written:
            assert isinstance(k, Symbol)
            if (k not in self.outputs
                    and k not in self.uninitialized_tile_temporaries
                    and k not in self.preinitialized_tile_temporaries):
                self.temporaries.add(k)

        for k in read:
            assert isinstance(k, Symbol), f"{k}, type={type(k)}"
            if (k not in self.inputs
                    and k not in self.params
                    and k not in self.uninitialized_tile_temporaries
                    and k not in self.preinitialized_tile_temporaries):
                self.temporaries.add(k)

        vprint(colorize("Temps:", "green"), self.temporaries)
        vprint(colorize("Uninitialized Tile Temps:", "green"), self.uninitialized_tile_temporaries)
        vprint(colorize("Preinitialized Tile Temps:", "green"), self.preinitialized_tile_temporaries)

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

        self.order_builder(complete)
        vprint(colorize("Order:", "green"), self.order)

        memory_pressure = self._score_memory_pressure()
        vprint(colorize("Memory Pressure:", "magenta"))
        vprint(f"  Total: {sorted(memory_pressure.items(), key=lambda kv: kv[1], reverse=True)}")
        vprint(f"  Mean: {mean(memory_pressure.values())}")
        vprint(f"  Median: {median(memory_pressure.values())}")
        vprint(f"  Max: {max(memory_pressure.items(), key=lambda kv: kv[1])}")

        for k in self.temporaries:
            assert k in read, f"Temporary variable '{k}' is never read"
            assert k in written, f"Temporary variable '{k}' is never written"
            # assert k not in self.outputs, f"Temporary variable '{k}' in outputs"
            assert k not in self.inputs, f"Temporary variable '{k}' in inputs"

        for k in read:
            assert k in self.inputs or self.params or self.temporaries, f"Symbol '{k}' is read, but it is not a temp, parameter, or input."

        vprint(colorize("READS:", "green"), end="")
        for var, spec in self.read_decls.items():
            if var in self.inputs:
                vprint(" ", var, "=", colorize(repr(spec), "yellow"), sep="", end="")
        vprint()
        vprint(colorize("WRITES:", "green"), end="")
        for var, spec in self.write_decls.items():
            if var in self.outputs:
                vprint(" ", var, "=", colorize(repr(spec), "yellow"), sep="", end="")
        vprint()

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

        for lhs in self.eqns:
            assert isinstance(lhs, Symbol), f"{lhs}, type={type(lhs)}"
            rhs = self.eqns[lhs]
            vprint(colorize("EQN:", "cyan"), lhs, colorize("=", "cyan"), rhs, " ", colorize(f"[complexity = {self.complexity[lhs]}]", "magenta"))

    def trim(self) -> None:
        """ Remove temporaries of the form "a=b". They are clutter. """
        subs: Dict[Symbol, Symbol] = dict()
        for k, v in self.eqns.items():
            if v.is_symbol:
                # k is not not needed
                subs[k] = cast(Symbol, v)
                wprint(f"Equation '{k} = {v}' can be trivially eliminated")

        new_eqns: Dict[Symbol, Expr] = dict()
        for k in self.eqns:
            if k not in subs:
                v = self.eqns[k]
                v2 = do_subs(v, subs)
                new_eqns[k] = v2

        self.eqns = new_eqns

    def _score_memory_pressure(self) -> Dict[Symbol, int]:
        assert len(self.order) > 0

        first_read: Dict[Symbol, int] = dict()
        last_read: Dict[Symbol, int] = dict()

        for idx, (_, rhs) in enumerate(sorted(self.eqns.items(), key=lambda eqn: self.order.index(eqn[0]))):
            for sym in free_symbols(rhs):
                if sym not in first_read:
                    first_read[sym] = idx
                last_read[sym] = idx

        return {sym: last_read[sym] - first_read[sym] + 1 for sym in first_read}

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

    def stencil_idxes(self) -> set[StencilIdxWithName]:
        result: set['StencilIdxWithName'] = set()
        for eqn in self.eqns.values():
            self._stencil_idxes(result, eqn)
        return result

    def _stencil_idxes(self, result: set[StencilIdxWithName], expr: Expr) -> None:
        grid_vars = self._grid_variables()
        stencil_calls: set[Basic] = expr.find(lambda x: hasattr(x, 'func') and self.is_stencil.get(x.func, False))  # type: ignore[no-untyped-call]
        straight_accesses: set[Basic] = expr.xreplace({call: Symbol("_stencil_call") for call in stencil_calls}).find(lambda x: x in grid_vars)  # type: ignore[no-untyped-call]

        for access in straight_accesses:
            result.add(StencilIdxWithName(StencilIdx(0, 0, 0), str(access)))

        for store in self.outputs:
            result.add(StencilIdxWithName(StencilIdx(0, 0, 0), str(store)))

        for call in stencil_calls:
            assert len(call.args) == 4, "Stencil function should have 4 arguments"
            result.add(StencilIdxWithName(tuple(int(typing.cast(Expr, a).evalf()) for a in call.args[1:]), str(call.args[0])))  # type: ignore[arg-type, no-untyped-call]

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
