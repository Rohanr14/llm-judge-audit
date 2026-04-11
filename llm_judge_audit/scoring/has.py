from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import cpu_count
from typing import Iterable

from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.judge import BaseJudge


@dataclass(frozen=True)
class HASResult:
    overall: float
    by_domain: dict[str, float]


def compute_human_alignment_score(judge: BaseJudge, items: Iterable[AnchorDatasetItem]) -> HASResult:
    totals: dict[str, int] = defaultdict(int)
    matches: dict[str, int] = defaultdict(int)

    all_items = list(items)
    if not all_items:
        return HASResult(overall=0.0, by_domain={})

    workers = min(32, (cpu_count() or 1) * 5)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        preferences = list(
            executor.map(
                lambda item: judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b),
                all_items,
            )
        )

    for item, pref in zip(all_items, preferences):
        totals["overall"] += 1
        totals[item.domain] += 1
        if pref == item.majority_preference:
            matches["overall"] += 1
            matches[item.domain] += 1

    by_domain = {
        domain: (matches[domain] / count if count else 0.0)
        for domain, count in totals.items()
        if domain != "overall"
    }
    overall = matches["overall"] / totals["overall"] if totals["overall"] else 0.0
    return HASResult(overall=overall, by_domain=by_domain)
