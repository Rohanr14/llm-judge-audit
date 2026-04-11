from typing import List

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.logger import logger


class PositionBiasTest(BaseBiasTest):
    """
    Measures position bias: Does the judge prefer whichever response appears first (or second)?
    Test method: Swap response order in pairwise comparisons; measure reversal rate.
    """

    @property
    def name(self) -> str:
        return "Position bias"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info("Running Position Bias test with %s on %s items.", judge.model_name, len(dataset))

        def evaluate_item(item: AnchorDatasetItem) -> tuple[str, str]:
            pref_1 = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)
            pref_2 = judge.evaluate_pairwise(item.prompt, item.response_b, item.response_a)
            return pref_1, pref_2

        outcomes = self._parallel_map(evaluate_item, dataset)

        reversals = 0
        valid_items = 0
        tie_items = 0

        for pref_1, pref_2 in outcomes:
            if pref_1 == "Tie" or pref_2 == "Tie":
                tie_items += 1
                continue

            valid_items += 1
            if (pref_1 == "A" and pref_2 == "A") or (pref_1 == "B" and pref_2 == "B"):
                reversals += 1

        score = reversals / valid_items if valid_items else 0.0
        tie_rate = tie_items / len(dataset) if dataset else 0.0
        logger.info("Position Bias score: %.2f (%s/%s reversals).", score, reversals, valid_items)

        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "reversals": reversals,
                "valid_items": valid_items,
                "tie_items": tie_items,
                "tie_rate": tie_rate,
                "total_items": len(dataset),
            },
        )
