#  Copyright (C) 2026 Max Morris and other Einstein Engine contributors.
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

from collections import defaultdict
from functools import cache
from typing import Callable, Optional, cast

from sympy import Symbol, Expr


class Dependencies:
    """
    Helper class for calculating acyclic dependencies between symbols in an equation system.
    """

    eqns: dict[Symbol, Expr]
    dependencies: dict[Symbol, set[Symbol]]
    dependents: dict[Symbol, set[Symbol]]
    head_nodes: set[Symbol]
    explored: set[Symbol]
    free_symbols: Callable[[Expr], set[Symbol]]

    def __init__(self,
                 eqns: dict[Symbol, Expr],
                 free_symbols: Callable[[Expr], set[Symbol]] = lambda expr: cast(set[Symbol], expr.free_symbols)):
        self.eqns = eqns
        self.free_symbols = free_symbols
        self.dependencies = defaultdict(set)
        self.dependents = defaultdict(set)
        self.head_nodes = set()
        self.explored = set()

        for sym in eqns:
            self.explore_dependencies(sym)

    def add_dependency(self, dependent: Symbol, dependency: Symbol) -> None:
        self.dependencies[dependent].add(dependency)
        self.dependents[dependency].add(dependent)

        if dependent in self.head_nodes:
            self.head_nodes.remove(dependent)

    def explore_dependencies(self, sym: Symbol) -> None:
        if sym in self.explored or sym not in self.eqns:
            return

        self.explored.add(sym)
        self.head_nodes.add(sym)

        for dep in self.free_symbols(self.eqns[sym]):
            self.add_dependency(sym, dep)
            self.explore_dependencies(dep)

    @cache
    def get_transitive_dependencies(self, sym: Symbol) -> set[Symbol]:
        if sym not in self.dependencies:
            return set()

        deps = set(self.dependencies[sym])
        for dep in self.dependencies[sym]:
            deps.update(self.get_transitive_dependencies(dep))

        return deps

    def get_partial_order(self, syms: set[Symbol], in_memory: Optional[set[Symbol]] = None) -> list[set[Symbol]]:
        f_syms = frozenset(syms) if not isinstance(syms, frozenset) else syms
        f_in_memory = (frozenset(in_memory) if not isinstance(in_memory, frozenset) else in_memory) if in_memory is not None else frozenset()
        return self._get_partial_order(f_syms, f_in_memory)

    @cache
    def _get_partial_order(self, syms: frozenset[Symbol], in_memory: frozenset[Symbol]) -> list[set[Symbol]]:
        """
        Use Kahn's algorithm to find a partial ordering of the given symbols.
        """
        eqn_syms = set(self.eqns.keys())
        roots: set[Symbol] = {sym for sym in syms if sym in eqn_syms and sym not in in_memory}
        all_syms: set[Symbol] = set(roots)
        frontier: list[Symbol] = list(roots)

        while len(frontier) > 0:
            sym = frontier.pop()
            for dep in self.dependencies[sym]:
                if dep in in_memory:
                    continue
                if dep in eqn_syms and dep not in all_syms:
                    all_syms.add(dep)
                    frontier.append(dep)

        indegree = {
            sym: sum(1 for dep in self.dependencies[sym] if dep in all_syms)
            for sym in all_syms
        }

        partial_order: list[set[Symbol]] = list()
        frontier = [sym for sym, degree in indegree.items() if degree == 0]
        explored_count = 0

        while len(frontier) > 0:
            level = set(frontier)
            partial_order.append(level)
            explored_count += len(level)

            next_frontier: list[Symbol] = list()
            for sym in frontier:
                for dependent in self.dependents[sym]:
                    if dependent not in indegree:
                        continue
                    indegree[dependent] -= 1
                    if indegree[dependent] == 0:
                        next_frontier.append(dependent)

            frontier = next_frontier

        if explored_count != len(all_syms):
            raise ValueError("Cycle detected while building partial order")

        return partial_order
