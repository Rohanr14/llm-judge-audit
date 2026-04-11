"""Domain transfer bias test.

Measures whether the judge's alignment with human labels varies substantially
between domains (code vs. factual vs. creative). We compute per-domain HAS-
style accuracy and report the max-min spread.

Items without a usable ground truth (no human annotations, or a tie-majority)
are excluded: there is no way to count them correct or incorrect, and
including them as "wrong by default" makes a judge look worse in proportion
to how unannotated the dataset is -- which is purely an artefact of the
Prolific pipeline, not of the judge.
"""

from collections import defaultdict
from typing import List

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.logger import logger
from llm_judge_audit.scoring.has import _is_scoreable


class DomainTransferBiasTest(BaseBiasTest):
    """Measures whether judge accuracy drops on a particular domain."""

    @property
    def name(self) -> str:
        return "Domain transfer bias"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info(
            "Running Domain Transfer Bias test with %s on %d items.",
            judge.model_name,
            len(dataset),
        )

        scoreable = [item for item in dataset if _is_scoreable(item, include_ties=False)]
        skipped = len(dataset) - len(scoreable)
        if skipped:
            logger.info(
                "Domain transfer: skipping %d/%d items with no ground truth "
                "(missing annotations or tie-majority).",
                skipped,
                len(dataset),
            )

        domain_correct: dict[str, int] = defaultdict(int)
        domain_total: dict[str, int] = defaultdict(int)

        for item in scoreable:
            pref = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)
            domain_total[item.domain] += 1
            if pref == item.majority_preference:
                domain_correct[item.domain] += 1

        if not domain_total:
            logger.warning("No scoreable items to compute domain transfer bias.")
            return BiasTestResult(
                bias_name=self.name,
                score=0.0,
                details={
                    "error": "No scoreable items in dataset",
                    "valid_items": 0,
                    "skipped_items": skipped,
                    "total_items": len(dataset),
                    "not_applicable": True,
                },
            )

        domain_accuracies = {
            domain: domain_correct[domain] / total
            for domain, total in domain_total.items()
        }

        if len(domain_accuracies) < 2:
            logger.warning(
                "Need at least 2 domains with scoreable items; got %d.",
                len(domain_accuracies),
            )
            score = 0.0
            not_applicable = True
        else:
            score = max(domain_accuracies.values()) - min(domain_accuracies.values())
            not_applicable = False

        logger.info(
            "Domain Transfer Bias score: %.2f (accuracies: %s)",
            score,
            domain_accuracies,
        )

        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "domain_accuracies": domain_accuracies,
                "valid_items": sum(domain_total.values()),
                "skipped_items": skipped,
                "total_items": len(dataset),
                "not_applicable": not_applicable,
            },
        )
