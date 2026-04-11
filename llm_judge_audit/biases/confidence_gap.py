"""Confidence-consistency gap bias test.

Runs each item ``n_runs`` times at a non-zero temperature so we get genuine
sampling variability, asking for both a preference and a confidence score.
The "gap" is the absolute difference between the judge's *reported*
confidence and its *observed* cross-run stability on that item.

Two important design choices:

* We sample at ``temperature > 0`` for the same reason as the cross-run test:
  at T=0 the stability is always 1.0 and the gap collapses to just "how
  confident does the model claim to be", which is uninformative.
* If the judge doesn't expose a confidence hook at all, the score is
  explicitly marked unavailable rather than silently reported as 0.0 (which
  the JRI used to read as "perfectly calibrated").
"""

from collections import Counter
from typing import List

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.biases.cross_run import SAMPLING_TEMPERATURE
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.logger import logger


class ConfidenceGapTest(BaseBiasTest):
    """Measures whether the judge's stated confidence tracks its stability."""

    def __init__(self, n_runs: int = 3, temperature: float = SAMPLING_TEMPERATURE):
        if n_runs < 2:
            raise ValueError("n_runs must be at least 2 to measure consistency.")
        if temperature <= 0.0:
            raise ValueError(
                "temperature must be > 0 for confidence-gap -- at T=0 the judge "
                "is deterministic and stability is always 1.0."
            )
        self.n_runs = n_runs
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "Confidence-consistency gap"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info(
            "Running Confidence-Consistency Gap test with %s on %d items "
            "(%d runs each at T=%.2f).",
            judge.model_name,
            len(dataset),
            self.n_runs,
            self.temperature,
        )

        item_gaps: list[float] = []
        items_with_confidence = 0
        items_without_confidence = 0

        for item in dataset:
            preferences: list[str] = []
            confidences: list[float] = []

            for _ in range(self.n_runs):
                pref, confidence = judge.evaluate_pairwise_with_confidence(
                    item.prompt,
                    item.response_a,
                    item.response_b,
                    temperature=self.temperature,
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

        # Report a dedicated "not applicable" sentinel when the judge didn't
        # expose any confidence. The JRI scoring layer is responsible for
        # handling this without trivially reading it as 0.0 (= perfect).
        if item_gaps:
            score: float | None = sum(item_gaps) / len(item_gaps)
        else:
            score = None
            logger.warning(
                "Judge did not provide any confidence scores; "
                "confidence-consistency gap is not applicable."
            )

        return BiasTestResult(
            bias_name=self.name,
            score=score if score is not None else 0.0,
            details={
                "items_with_confidence": items_with_confidence,
                "items_without_confidence": items_without_confidence,
                "n_runs": self.n_runs,
                "temperature": self.temperature,
                "total_items": len(dataset),
                "not_applicable": score is None,
            },
        )
