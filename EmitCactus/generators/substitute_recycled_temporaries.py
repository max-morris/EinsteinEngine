from dataclasses import dataclass
from typing import Optional, cast

import sympy as sy

from EmitCactus.dsl.eqnlist import TemporaryReplacement, EqnList


@dataclass(frozen=True)
class SubstituteRecycledTemporariesResult:
    eqns: list[tuple[sy.Symbol, sy.Expr]]
    substituted_lhs_idxes: set[int]


def substitute_recycled_temporaries(eqn_list: EqnList) -> SubstituteRecycledTemporariesResult:
    eqns: list[tuple[sy.Symbol, sy.Expr]] = list()
    substituted_lhs_idxes: set[int] = set()

    for eqn_idx, (lhs, rhs) in enumerate(eqn_list.sorted_eqns):
        active_replacements = list(filter(
            lambda r: r.begin_eqn <= eqn_idx <= r.end_eqn,
            eqn_list.temporary_replacements
        ))

        current_line_replacement = cast(Optional[TemporaryReplacement],
                                        next(filter(lambda r: r.begin_eqn == eqn_idx, active_replacements), None))

        for replacement in active_replacements:
            rhs = rhs.replace(replacement.old, replacement.new)  # type: ignore[no-untyped-call]

        if current_line_replacement:
            assert lhs == current_line_replacement.old, "Current line replacement target doesn't match LHS"
            lhs = current_line_replacement.new
            substituted_lhs_idxes.add(eqn_idx)

        eqns.append((lhs, rhs))

    return SubstituteRecycledTemporariesResult(eqns, substituted_lhs_idxes)
