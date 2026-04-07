from typing import List

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.datasets.schema import AnchorDatasetItem
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
        logger.info(f"Running Position Bias test with {judge.model_name} on {len(dataset)} items.")

        position_consistent_choices = 0
        valid_items = 0
        
        for item in dataset:
            # First order: A is response_a, B is response_b
            pref_1 = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)
            
            # Second order: A is response_b, B is response_a
            pref_2 = judge.evaluate_pairwise(item.prompt, item.response_b, item.response_a)
            
            # Skip ties for reversal calculation as they are ambiguous
            if pref_1 == "Tie" or pref_2 == "Tie":
                continue
                
            valid_items += 1
            
            # To be consistent, the model should choose the same *content*.
            # If pref_1 == "A", it preferred item.response_a.
            # In the second run, item.response_a is in position B.
            # So a consistent judge would return "B" for pref_2.
            # If it returns "A" for pref_2, it preferred item.response_b, meaning it changed its mind.
            
            if pref_1 == "A" and pref_2 == "B":
                # Consistent preference for item.response_a
                pass
            elif pref_1 == "B" and pref_2 == "A":
                # Consistent preference for item.response_b
                pass
            else:
                # E.g. pref_1 == "A" and pref_2 == "A" -> Chose 1st pos both times
                # E.g. pref_1 == "B" and pref_2 == "B" -> Chose 2nd pos both times
                # These are reversals in content preference!
                position_consistent_choices += 1
                
        if valid_items == 0:
            logger.warning("No valid items (non-ties) to compute position bias.")
            score = 0.0
        else:
            score = position_consistent_choices / valid_items
            
        logger.info(f"Position Bias score: {score:.2f} ({position_consistent_choices}/{valid_items} position_consistent_choices)")
        
        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "position_consistent_choices": position_consistent_choices,
                "reversals": position_consistent_choices,
                "valid_items": valid_items,
                "total_items": len(dataset),
            }
        )
