from typing import List
from collections import Counter

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.logger import logger

class ConfidenceGapTest(BaseBiasTest):
    """
    Measures confidence-consistency gap: Is the judge's expressed confidence calibrated to its actual stability?
    Test method: Ask for both a preference and a confidence score. Re-run multiple times. 
    Measure if high-confidence ratings are actually stable across runs.
    """

    def __init__(self, n_runs: int = 3):
        if n_runs < 2:
            raise ValueError("n_runs must be at least 2 to measure consistency.")
        self.n_runs = n_runs

    @property
    def name(self) -> str:
        return "Confidence-consistency gap"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info(
            f"Running Confidence-Consistency Gap test with {judge.model_name} "
            f"on {len(dataset)} items ({self.n_runs} runs each)."
        )

        item_gaps: list[float] = []
        items_with_confidence = 0
        items_without_confidence = 0

        for item in dataset:
            preferences: list[str] = []
            confidences: list[float] = []

            for _ in range(self.n_runs):
                pref, confidence = judge.evaluate_pairwise_with_confidence(
                    item.prompt, item.response_a, item.response_b
                )
                preferences.append(pref)
                if confidence is not None and 0.0 <= confidence <= 1.0:
                    confidences.append(confidence)

            non_tie_prefs = [p for p in preferences if p != "Tie"]
            if not non_tie_prefs:
                continue

            consensus_count = Counter(non_tie_prefs).most_common(1)[0][1]
            stability = consensus_count / len(non_tie_prefs)

            if confidences:
                reported_confidence = sum(confidences) / len(confidences)
                item_gaps.append(abs(reported_confidence - stability))
                items_with_confidence += 1
            else:
                items_without_confidence += 1

        if item_gaps:
            score = sum(item_gaps) / len(item_gaps)
        else:
            score = 0.0

        if items_with_confidence == 0:
            logger.warning(
                "Judge did not provide confidence scores; returning 0.0 "
                "for confidence-consistency gap."
            )

        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "items_with_confidence": items_with_confidence,
                "items_without_confidence": items_without_confidence,
                "n_runs": self.n_runs,
                "total_items": len(dataset),
            }
        )
