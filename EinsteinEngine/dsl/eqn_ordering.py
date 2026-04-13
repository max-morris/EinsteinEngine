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

from __future__ import annotations

import json
import os
import random
import re
import time
from collections import defaultdict
from typing import Iterator, Callable, TYPE_CHECKING
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sympy import Symbol, Expr

from EinsteinEngine.dsl.sympywrap import free_symbols
from util import pprint, wprint

if TYPE_CHECKING:
    from EinsteinEngine.dsl.eqnlist import EqnList

EqnOrderingFn = Callable[[dict[Symbol, Expr], 'EqnList'], Iterator[Symbol]]

def maximize_symbol_reuse(eqns: dict[Symbol, Expr], eqn_list: EqnList) -> Iterator[Symbol]:
    """
    Orders equations based on symbol reuse, prioritizing equations that use symbols already present in previous equations.
    Equations with higher complexity are given higher priority. The first equation is always the most complex.
    """

    if len(eqns) == 0:
        return

    eqns_remaining = eqns.copy()
    in_memory: set[Symbol] = set()

    disambiguation = sorted(eqns_remaining.keys(), key=str, reverse=True)

    lhs, rhs = max(eqns_remaining.items(), key=lambda kv: (eqn_list.complexity[kv[0]], disambiguation.index(kv[0])))
    del eqns_remaining[lhs]
    in_memory.update(free_symbols(rhs))
    yield lhs

    while len(eqns_remaining) > 0:
        lhs, rhs = max(eqns_remaining.items(),
                       key=lambda kv: (len(free_symbols(kv[1]).intersection(in_memory)), eqn_list.complexity[kv[0]],
                                       disambiguation.index(kv[0])))
        del eqns_remaining[lhs]
        in_memory.update(free_symbols(rhs))
        yield lhs

def prioritize_rare_symbols(eqns: dict[Symbol, Expr],
                            eqn_list: EqnList,
                            consider_frequency: bool = True,
                            complexity_factor: float = 0.0) -> Iterator[Symbol]:
    """
    Orders equations based on symbol rarity.
    Equations which use symbols that are less common in other equations are given higher priority.

    To determine the rarity of a symbol, multiple occurrences of the same symbol in an equation are treated as one.
    If `consider_frequency` is true, when evaluating the overall priority of an equation, the rarity of each symbol is weighted positively by the frequency of that symbol in the equation.

    The complexity score of an equation, scaled by `complexity_factor`, is added to the priority.
    """

    if len(eqns) == 0:
        return

    reciprocal_rarity: dict[Symbol, float] = defaultdict(int)
    frequency_by_eqn: dict[Symbol, dict[Symbol, float]] = defaultdict(dict)  # {lhs: {sym: freq}}
    for lhs, rhs in eqns.items():
        for sym in free_symbols(rhs):
            reciprocal_rarity[sym] += 1
            frequency_by_eqn[lhs][sym] = rhs.count(sym)  # type: ignore[no-untyped-call]

    def symbol_rarity(sym: Symbol) -> float:
        return 1 / reciprocal_rarity[sym]

    def symbol_score(sym: Symbol, lhs: Symbol) -> float:
        return frequency_by_eqn[lhs][sym] * symbol_rarity(sym) if consider_frequency else symbol_rarity(sym)

    def eqn_score(lhs: Symbol) -> float:
        return (complexity_factor * eqn_list.complexity[lhs]) + sum(symbol_score(sym, lhs) for sym in free_symbols(eqns[lhs]))

    disambiguation = sorted(eqns.keys(), key=str, reverse=True)
    ordered = sorted(eqns.keys(), key=lambda lhs: (eqn_score(lhs), eqn_list.complexity[lhs], disambiguation.index(lhs)), reverse=True)

    yield from ordered.__iter__()


def _normalize_llm_key_token(token: str) -> str:
    cleaned = token.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"', "`"}:
        return cleaned[1:-1].strip()
    return cleaned


def _extract_order_from_llm_content(content: str, allowed_keys: set[str]) -> list[str]:
    tagged_match = re.search(r"<ORDER>(.*?)</ORDER>", content, flags=re.DOTALL)
    parse_target = tagged_match.group(1) if tagged_match else content

    tokens = [_normalize_llm_key_token(x) for x in parse_target.replace("\n", ",").split(",") if x.strip()]
    if len(tokens) == len(allowed_keys) and set(tokens) == allowed_keys:
        return tokens

    filtered = [tok for tok in tokens if tok in allowed_keys]
    n = len(allowed_keys)

    for start in range(len(filtered) - n, -1, -1):
        window = filtered[start:start + n]
        if len(set(window)) == n and set(window) == allowed_keys:
            return window

    seen: set[str] = set()
    deduped: list[str] = []
    for tok in filtered:
        if tok not in seen:
            seen.add(tok)
            deduped.append(tok)
    if len(deduped) == n and set(deduped) == allowed_keys:
        return deduped

    return tokens


def _repair_order(candidate: list[str], baseline_order: list[str], allowed_keys: set[str]) -> list[str]:
    seen: set[str] = set()
    repaired: list[str] = []
    for tok in candidate:
        if tok in allowed_keys and tok not in seen:
            seen.add(tok)
            repaired.append(tok)
    for tok in baseline_order:
        if tok not in seen:
            repaired.append(tok)
    return repaired


def _extract_order_from_llm_edits(content: str, baseline_order: list[str], allowed_keys: set[str], max_edits: int = 8) -> list[str]:
    tagged_match = re.search(r"<EDITS>(.*?)</EDITS>", content, flags=re.DOTALL)
    parse_target = tagged_match.group(1) if tagged_match else content

    order = list(baseline_order)
    applied = 0

    for raw_line in parse_target.splitlines():
        if applied >= max_edits:
            break
        line = raw_line.strip().strip("-*")
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)

        move_after = re.fullmatch(r"MOVE\s+(.+?)\s+AFTER\s+(.+)", line, flags=re.IGNORECASE)
        if move_after:
            a = _normalize_llm_key_token(move_after.group(1))
            b = _normalize_llm_key_token(move_after.group(2))
            if a in allowed_keys and b in allowed_keys and a != b and a in order and b in order:
                order.remove(a)
                order.insert(order.index(b) + 1, a)
                applied += 1
            continue

        move_before = re.fullmatch(r"MOVE\s+(.+?)\s+BEFORE\s+(.+)", line, flags=re.IGNORECASE)
        if move_before:
            a = _normalize_llm_key_token(move_before.group(1))
            b = _normalize_llm_key_token(move_before.group(2))
            if a in allowed_keys and b in allowed_keys and a != b and a in order and b in order:
                order.remove(a)
                order.insert(order.index(b), a)
                applied += 1
            continue

        swap = re.fullmatch(r"SWAP\s+(.+?)\s+(.+)", line, flags=re.IGNORECASE)
        if swap:
            a = _normalize_llm_key_token(swap.group(1))
            b = _normalize_llm_key_token(swap.group(2))
            if a in allowed_keys and b in allowed_keys and a != b and a in order and b in order:
                ia, ib = order.index(a), order.index(b)
                order[ia], order[ib] = order[ib], order[ia]
                applied += 1
            continue

    if applied > 0:
        return order
    return _extract_order_from_llm_content(content, allowed_keys)


def _build_eqn_dependencies(eqns: dict[Symbol, Expr]) -> dict[Symbol, set[Symbol]]:
    eqn_symbols = set(eqns.keys())
    return {lhs: {sym for sym in free_symbols(rhs) if sym in eqn_symbols} for lhs, rhs in eqns.items()}


def _symbol_live_spans(order: list[Symbol], eqns: dict[Symbol, Expr]) -> dict[Symbol, int]:
    order_index = {sym: i for i, sym in enumerate(order)}
    first_read: dict[Symbol, int] = {}
    last_read: dict[Symbol, int] = {}

    for lhs in sorted(eqns.keys(), key=lambda s: order_index[s]):
        idx = order_index[lhs]
        for sym in free_symbols(eqns[lhs]):
            if sym not in first_read:
                first_read[sym] = idx
            last_read[sym] = idx

    return {sym: last_read[sym] - first_read[sym] + 1 for sym in first_read}


def _order_score(order: list[Symbol],
                 eqns: dict[Symbol, Expr],
                 eqn_list: EqnList,
                 dependencies: dict[Symbol, set[Symbol]]) -> float:
    if len(order) == 0:
        return 0.0

    order_index = {sym: i for i, sym in enumerate(order)}
    remaining_uses: dict[Symbol, int] = {sym: 0 for sym in eqns}
    for rhs in eqns.values():
        for dep in free_symbols(rhs):
            if dep in remaining_uses:
                remaining_uses[dep] += 1

    live_temps: set[Symbol] = set()
    peak_live = 0
    live_area = 0
    reuse_hits = 0
    dependency_jump = 0
    complexity_frontload = 0.0

    for i, lhs in enumerate(order):
        deps = dependencies[lhs]
        reuse_hits += len(deps.intersection(live_temps))
        dependency_jump += sum(i - order_index[dep] for dep in deps)
        complexity_frontload += eqn_list.complexity.get(lhs, 0) / (i + 1)

        live_area += len(live_temps)
        live_temps.add(lhs)
        peak_live = max(peak_live, len(live_temps))

        for dep in deps:
            remaining_uses[dep] -= 1
            if remaining_uses[dep] <= 0:
                live_temps.discard(dep)

    spans = _symbol_live_spans(order, eqns)
    rhs_span_penalty = 0.0
    for lhs in order:
        rhs_syms = free_symbols(eqns[lhs])
        if len(rhs_syms) > 0:
            rhs_span_penalty += sum(spans.get(sym, 1) for sym in rhs_syms) / len(rhs_syms)

    n = max(1, len(order))
    return (
        (2.0 * reuse_hits)
        - (1.4 * peak_live)
        - (0.25 * (live_area / n))
        - (0.5 * (dependency_jump / n))
        - (0.06 * (rhs_span_penalty / n))
        + (0.02 * complexity_frontload)
    )


def _get_cthulhu_bark() -> str:
    verbs = [
        "Consulting", "Asking", "Communing with", "Conspiring with", "Inquiring upon",
        "Interrogating", "Investigating the nature of", "Gazing into", "Pondering the nature of",
        "Falling prey to", "Seeking the truth of", "Acquiring the wisdom of", "Succumbing to",
        "Investigating the mysteries of", "Seeking the arcane secrets of", "Praying to", "Meeting", "Touching",
        "Having a chat with", "Commiserating with", "Taking inspiration from"
    ]

    nouns = [
        "chaos", "the stars", "the cosmos", "the void", "Cthulhu", "Beelzebub", "Lucifer",
        "Asmodeus", "infinity", "the abyss", "longing", "the deepest recesses of the human mind",
        "the lay lines", "the lines", "the unseen", "the unknowable", "the darkness", "the unthinkable",
        "the Singularity", "Skynet"
    ]

    return f'{random.choice(verbs)} {random.choice(nouns)}...'

def ask_cthulhu(eqns: dict[Symbol, Expr],
                eqn_list: EqnList,
                gpu: str = 'Unspecified Nvidia GPU',
                model: str = 'openai/gpt-4o-mini',
                base_fn: EqnOrderingFn = prioritize_rare_symbols) -> Iterator[Symbol]:
    pprint(_get_cthulhu_bark())

    if len(eqns) == 0:
        return

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key is None or api_key.strip() == "":
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set.")

    dependencies = _build_eqn_dependencies(eqns)
    baseline_order = list(base_fn(eqns, eqn_list))
    baseline_score = _order_score(baseline_order, eqns, eqn_list, dependencies)

    fanout: dict[Symbol, int] = {lhs: 0 for lhs in eqns}
    for deps in dependencies.values():
        for dep in deps:
            fanout[dep] += 1

    reciprocal_rarity: dict[Symbol, float] = defaultdict(int)
    for rhs in eqns.values():
        for sym in free_symbols(rhs):
            reciprocal_rarity[sym] += 1
    rarity_score: dict[Symbol, float] = {}
    for lhs, rhs in eqns.items():
        rarity_score[lhs] = sum((1 / reciprocal_rarity[sym]) for sym in free_symbols(rhs) if reciprocal_rarity[sym] > 0)

    spans = _symbol_live_spans(baseline_order, eqns)
    dependency_lines: list[str] = []
    for lhs in baseline_order:
        deps = sorted(dependencies[lhs], key=str)
        complexity = eqn_list.complexity.get(lhs, -1)
        dep_text = ", ".join(map(str, deps))
        baseline_span = spans.get(lhs, 1)
        dependency_lines.append(
            f"({lhs}: deps=[{dep_text}], dep_count={len(deps)}, fanout={fanout[lhs]}, "
            f"complexity={complexity}, rarity={rarity_score[lhs]:.6f}, baseline_span={baseline_span})"
        )
    dependency_summary = "\n".join(dependency_lines)
    allowed_keys = sorted((str(lhs) for lhs in eqns.keys()), key=str)
    allowed_keys_set = set(allowed_keys)
    baseline_key_order = [str(sym) for sym in baseline_order]
    allowed_keys_text = ", ".join(allowed_keys)
    baseline_order_text = ", ".join(baseline_key_order)

    base_prompt = (
        "Task: improve the provided baseline ordering for a GPU kernel.\n\n"
        f"GPU target: {gpu}\n\n"
        "Optimization objective (higher is better): increase temporary reuse, reduce peak live temporaries, "
        "reduce dependency-distance/jumps, and keep equations which use less frequent symbols as well as high-complexity equations earlier when feasible.\n\n"
        "Allowed output keys (and ONLY these keys):\n"
        f"{allowed_keys_text}\n\n"
        "Baseline order to refine:\n"
        f"{baseline_order_text}\n\n"
        "Per-equation features:\n"
        f"{dependency_summary}\n\n"
        "Output format requirement:\n"
        "<EDITS>\n"
        "MOVE <keyA> AFTER <keyB>\n"
        "MOVE <keyA> BEFORE <keyB>\n"
        "SWAP <keyA> <keyB>\n"
        "</EDITS>\n"
        "Return at most 100 edit lines. If no beneficial edit exists, return an empty EDITS block.\n"
        "Hard constraints:\n"
        "1) Refer only to allowed output keys.\n"
        "2) Do not include explanations outside the EDITS block."
    )
    prompt_chars = len(base_prompt)

    max_hallucination_retries = 5
    input_keys = {str(lhs): lhs for lhs in eqns.keys()}
    requested_order: list[str] = []
    last_validation_error: Exception | None = None
    best_order = baseline_order
    best_score = baseline_score

    for attempt in range(1, max_hallucination_retries + 2):
        prompt = base_prompt
        if last_validation_error is not None:
            prompt += (
                "\n\nYour previous response was invalid.\n"
                f"Validation error: {last_validation_error}\n"
                "Retry now and strictly follow the EDITS block format."
            )

        req_body = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert GPU kernel scheduling assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }

        request = Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(req_body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:  # nosec B310
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_validation_error = RuntimeError(
                f"OpenRouter request failed with HTTP {exc.code} ({exc.reason}). "
                f"eqn_count={len(eqns)}, prompt_chars={prompt_chars}. Response body: {body}"
            )
            break
        except URLError as exc:
            last_validation_error = RuntimeError(f"OpenRouter request failed due to network error: {exc.reason}")
            break

        try:
            content = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            last_validation_error = RuntimeError(f"OpenRouter response missing expected message content: {response_data}")
            break

        if not isinstance(content, str):
            last_validation_error = RuntimeError(f"OpenRouter returned non-string message content: {content!r}")
            break

        requested_order = _extract_order_from_llm_edits(content, baseline_key_order, allowed_keys_set, max_edits=100)
        requested_order = _repair_order(requested_order, baseline_key_order, allowed_keys_set)
        output_keys = set(requested_order)

        if len(requested_order) == 0:
            last_validation_error = ValueError(f"LLM returned an empty ordering. Raw output: {content!r}")
        elif output_keys != set(input_keys.keys()):
            missing = sorted(set(input_keys.keys()) - output_keys)
            extra = sorted(output_keys - set(input_keys.keys()))
            last_validation_error = ValueError(
                f"LLM keyset mismatch. Missing: {missing}. Unexpected: {extra}. Raw output: {content!r}"
            )
        else:
            candidate = [input_keys[key] for key in requested_order]
            candidate_score = _order_score(candidate, eqns, eqn_list, dependencies)
            if candidate_score <= best_score:
                last_validation_error = ValueError(
                    f"LLM refinement did not improve score (candidate={candidate_score:.4f}, "
                    f"baseline={best_score:.4f})."
                )
            else:
                best_order = candidate
                best_score = candidate_score
                last_validation_error = None
                break

        if attempt <= max_hallucination_retries:
            time.sleep(0.2 * attempt)

    if last_validation_error is not None and best_order == baseline_order:
        wprint(f"ask_cthulhu fallback to base_fn ({base_fn}): {last_validation_error}")

    yield from iter(best_order)
