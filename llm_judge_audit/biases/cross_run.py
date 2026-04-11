"""Cross-run consistency bias test.

Re-submits each item ``n_runs`` times at a non-zero sampling temperature and
measures the fraction of items whose preference flipped at least once across
the runs. A non-zero temperature is *required* -- at temperature 0 every call
is deterministic and this test would always report zero inconsistency no
matter how unstable the judge actually is.
"""

from typing import List

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.logger import logger

# Moderate sampling temperature. High enough to expose genuine instability
# but not so high that the judge stops following JSON instructions.
SAMPLING_TEMPERATURE = 0.7


class CrossRunConsistencyTest(BaseBiasTest):
    """Measures whether identical inputs yield the same preference on replay."""

    def __init__(self, n_runs: int = 3, temperature: float = SAMPLING_TEMPERATURE):
        if n_runs < 2:
            raise ValueError("n_runs must be at least 2 to measure consistency.")
        if temperature <= 0.0:
            raise ValueError(
                "temperature must be > 0 for cross-run consistency -- at T=0 the "
                "judge is deterministic and the test is meaningless."
            )
        self.n_runs = n_runs
        self.temperature = temperature

    @property
    def name(self) -> str:
        return "Cross-run consistency"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info(
            "Running Cross-Run Consistency test with %s on %d items (%d runs each at T=%.2f).",
            judge.model_name,
            len(dataset),
            self.n_runs,
            self.temperature,
        )

        inconsistent_items = 0

        for item in dataset:
            preferences = []
            for _ in range(self.n_runs):
                pref = judge.evaluate_pairwise(
                    item.prompt,
                    item.response_a,
                    item.response_b,
                    temperature=self.temperature,
                )
                preferences.append(pref)

            # Count unique non-tie preferences; an item is inconsistent if
            # the judge gave both "A" and "B" across the runs.
            unique_prefs = {p for p in preferences if p != "Tie"}
            if len(unique_prefs) > 1:
                inconsistent_items += 1

        score = (inconsistent_items / len(dataset)) if dataset else 0.0

        logger.info(
            "Cross-Run Consistency score: %.2f (%d/%d inconsistent items)",
            score,
            inconsistent_items,
            len(dataset),
        )

        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "inconsistent_items": inconsistent_items,
                "total_items": len(dataset),
                "n_runs": self.n_runs,
                "temperature": self.temperature,
            },
        )
