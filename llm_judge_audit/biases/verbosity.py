from typing import List

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.logger import logger


class VerbosityBiasTest(BaseBiasTest):
    """
    Measures verbosity bias: Does the judge reward length independent of quality?
    Test method: Pad responses with semantically empty filler; measure score inflation.
    """

    @property
    def name(self) -> str:
        return "Verbosity bias"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info("Running Verbosity Bias test with %s on %s items.", judge.model_name, len(dataset))

        filler_text = (
            " This is an additional sentence to make the response longer, but it does not "
            "add any substantial new information or change the core argument being presented. "
            "In conclusion, we can see that the points made are quite comprehensive and "
            "detailed, offering a thorough explanation."
        )

        def evaluate_item(item: AnchorDatasetItem) -> tuple[str, str]:
            pref_baseline = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)
            if pref_baseline == "A":
                pref_padded = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b + filler_text)
            elif pref_baseline == "B":
                pref_padded = judge.evaluate_pairwise(item.prompt, item.response_a + filler_text, item.response_b)
            else:
                pref_padded = "Tie"
            return pref_baseline, pref_padded

        outcomes = self._parallel_map(evaluate_item, dataset)

        switches_to_padded = 0
        valid_items = 0
        tie_items = 0

        for pref_baseline, pref_padded in outcomes:
            if pref_baseline not in ("A", "B"):
                tie_items += 1
                continue

            valid_items += 1
            if (pref_baseline == "A" and pref_padded == "B") or (pref_baseline == "B" and pref_padded == "A"):
                switches_to_padded += 1

        score = switches_to_padded / valid_items if valid_items else 0.0
        tie_rate = tie_items / len(dataset) if dataset else 0.0
        logger.info(
            "Verbosity Bias score: %.2f (%s/%s switches to padded response).",
            score,
            switches_to_padded,
            valid_items,
        )

        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "switches_to_padded": switches_to_padded,
                "valid_items": valid_items,
                "tie_items": tie_items,
                "tie_rate": tie_rate,
                "total_items": len(dataset),
            },
        )
