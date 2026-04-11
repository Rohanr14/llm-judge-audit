from typing import List

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.logger import logger

class AnchoringBiasTest(BaseBiasTest):
    """
    Measures anchoring bias: Do prior scores in the session skew subsequent scores?
    Test method: Precede the target evaluation with previous evaluations all skewed toward one outcome; measure pull effect.
    """

    @property
    def name(self) -> str:
        return "Anchoring bias"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info(f"Running Anchoring Bias test with {judge.model_name} on {len(dataset)} items.")
        
        switches_to_anchor = 0
        valid_items = 0
        
        example_a_1 = {"role": "assistant", "content": "Prior evaluation preference: A"}
        example_a_2 = {"role": "assistant", "content": "Prior evaluation preference: A"}

        example_b_1 = {"role": "assistant", "content": "Prior evaluation preference: B"}
        example_b_2 = {"role": "assistant", "content": "Prior evaluation preference: B"}
        
        for item in dataset:
            pref_baseline = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)
            
            if pref_baseline not in ("A", "B"):
                continue
                
            valid_items += 1
            
            if pref_baseline == "A":
                # Try to anchor to B
                history = [example_b_1, example_b_2]
                anchor_target = "B"
            else:
                # Try to anchor to A
                history = [example_a_1, example_a_2]
                anchor_target = "A"
                
            pref_anchored = judge.evaluate_pairwise_with_history(
                item.prompt,
                item.response_a,
                item.response_b,
                history=history,
            )
            
            if pref_anchored == anchor_target:
                switches_to_anchor += 1
                
        if valid_items == 0:
            logger.warning("No valid items to compute anchoring bias.")
            score = 0.0
        else:
            score = switches_to_anchor / valid_items
            
        logger.info(f"Anchoring Bias score: {score:.2f} ({switches_to_anchor}/{valid_items} switches to anchor)")
        
        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "switches_to_anchor": switches_to_anchor,
                "valid_items": valid_items,
                "total_items": len(dataset),
            }
        )
