"""Human Alignment Score computation.

HAS is "fraction of anchor items where the judge's preference matches the
human majority vote". The two subtleties are:

1. Items without human annotations (e.g. a Prolific run that hasn't finished
   yet) cannot contribute to HAS -- there is no ground truth to match
   against. The previous implementation compared the judge to the default
   ``majority_preference="Tie"`` which quietly counted unannotated items as
   agreement-on-tie. That inflates the score on any partially-annotated
   dataset.

2. Items where the human majority was "Tie" are also excluded by default.
   A judge that picks A or B when humans couldn't agree isn't *wrong* in
   any principled sense; including ties just pulls the score toward the
   judge's intrinsic tie rate, which is noise. Callers who want the old
   behaviour can opt back in via ``include_ties=True``.
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable

from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.logger import logger
from llm_judge_audit.runtime import SETTINGS


@dataclass(frozen=True)
class HASResult:
    overall: float
    by_domain: dict[str, float]
    # Number of items that actually contributed to the score (i.e. items
    # with >= 1 human annotation and, by default, a non-tie majority).
    scored_items: int = 0
    # Number of items in the input that we had to skip because of missing
    # annotations or tie-majority filtering.
    skipped_items: int = 0


def _is_scoreable(item: AnchorDatasetItem, include_ties: bool) -> bool:
    """Return True iff the item has a ground-truth majority we can score against."""
    if not item.human_annotations:
        return False
    if not include_ties and item.majority_preference == "Tie":
        return False
    return True


def compute_human_alignment_score(
    judge: BaseJudge,
    items: Iterable[AnchorDatasetItem],
    *,
    include_ties: bool = False,
    max_workers: int | None = None,
) -> HASResult:
    all_items = list(items)
    if not all_items:
        return HASResult(overall=0.0, by_domain={}, scored_items=0, skipped_items=0)

    scoreable = [i for i in all_items if _is_scoreable(i, include_ties=include_ties)]
    skipped = len(all_items) - len(scoreable)
    if skipped:
        logger.info(
            "HAS: skipping %d/%d items (no human annotations or tie-majority).",
            skipped,
            len(all_items),
        )
    if not scoreable:
        logger.warning(
            "HAS: no items are scoreable; returning 0.0. "
            "Check that anchor.json has populated human_annotations."
        )
        return HASResult(
            overall=0.0, by_domain={}, scored_items=0, skipped_items=skipped
        )

    workers = max_workers if max_workers is not None else SETTINGS.max_concurrency
    workers = max(1, min(workers, len(scoreable)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        preferences = list(
            executor.map(
                lambda item: judge.evaluate_pairwise(
                    item.prompt, item.response_a, item.response_b
                ),
                scoreable,
            )
        )

    totals: dict[str, int] = defaultdict(int)
    matches: dict[str, int] = defaultdict(int)
    for item, pref in zip(scoreable, preferences):
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
    return HASResult(
        overall=overall,
        by_domain=by_domain,
        scored_items=totals["overall"],
        skipped_items=skipped,
    )
