from typing import List
import random

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.logger import logger

class RecencyBiasTest(BaseBiasTest):
    """
    Measures recency bias: Do few-shot examples close to the query anchor the judge's scoring?
    Test method: Vary which examples appear last in the prompt; measure influence on output.
    """

    @property
    def name(self) -> str:
        return "Recency bias"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info(f"Running Recency Bias test with {judge.model_name} on {len(dataset)} items.")
        
        switches_to_recent = 0
        valid_items = 0
        
        # We need to craft few-shot examples.
        # Example 1: Strongly prefers Response A.
        # Example 2: Strongly prefers Response B.
        example_a_pref = (
            "Example Evaluation 1:\n"
            "Prompt: Explain gravity.\n"
            "Response A: Gravity is a fundamental interaction that causes mutual attraction between all things that have mass.\n"
            "Response B: It makes things fall down.\n"
            "Judge Preference: A\n"
        )
        
        example_b_pref = (
            "Example Evaluation 2:\n"
            "Prompt: What is the capital of Japan?\n"
            "Response A: Paris.\n"
            "Response B: Tokyo is the capital of Japan.\n"
            "Judge Preference: B\n"
        )
        
        for item in dataset:
            # Baseline preference (no few-shot examples)
            pref_baseline = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)
            
            if pref_baseline not in ("A", "B"):
                continue
                
            valid_items += 1
            
            # If baseline is A, we want to try to anchor it to B by putting the B-preferring example last.
            if pref_baseline == "A":
                recent_pref = "B"
                few_shot_prompt = f"{example_a_pref}\n{example_b_pref}\nNow evaluate the following:\n\n{item.prompt}"
            else:
                recent_pref = "A"
                few_shot_prompt = f"{example_b_pref}\n{example_a_pref}\nNow evaluate the following:\n\n{item.prompt}"
                
            # Re-evaluate with few-shot prompt
            pref_recent = judge.evaluate_pairwise(few_shot_prompt, item.response_a, item.response_b)
            
            if pref_recent == recent_pref:
                switches_to_recent += 1
                
        if valid_items == 0:
            logger.warning("No valid items (with clear baseline preference) to compute recency bias.")
            score = 0.0
        else:
            score = switches_to_recent / valid_items
            
        logger.info(f"Recency Bias score: {score:.2f} ({switches_to_recent}/{valid_items} switches to recent few-shot example)")
        
        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "switches_to_recent": switches_to_recent,
                "valid_items": valid_items,
                "total_items": len(dataset),
            }
        )
