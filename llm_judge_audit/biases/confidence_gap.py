from typing import List

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

    @property
    def name(self) -> str:
        return "Confidence-consistency gap"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info(f"Running Confidence-Consistency Gap test with {judge.model_name} on {len(dataset)} items.")
        
        # We need a custom evaluation function for this test to extract confidence.
        # Since BaseJudge only returns A/B/Tie, we'll have to use a slightly modified prompt 
        # and hope the existing judge parses it okay, or we'd ideally modify the judge interface.
        # For the sake of this audit framework, we will simulate it by checking if the judge 
        # returns consistent answers, and comparing that to a "confidence" prompt.
        # 
        # Since we can't easily extract confidence from the standard evaluate_pairwise, 
        # we will do a simpler version: 
        # We ask the judge to evaluate with "high confidence" requirement.
        
        # NOTE: A true implementation of this would require modifying BaseJudge to return 
        # (preference, confidence_score). For this suite, we will implement a surrogate:
        # We will run the evaluation 3 times. We will also ask the judge to output a confidence level.
        # If we can't get confidence, we'll just return 0.0 for now, but log that it requires 
        # a judge interface update.
        
        # Let's do a basic implementation where we just measure cross-run consistency 
        # as a proxy for stability, but we don't have a way to extract the model's self-reported confidence 
        # without changing the `BaseJudge.evaluate_pairwise` signature.
        
        logger.warning("Confidence-consistency gap test requires judge interface to return confidence. Returning 0.0 placeholder.")
        
        return BiasTestResult(
            bias_name=self.name,
            score=0.0,
            details={
                "note": "Not fully implemented without changing BaseJudge signature to return confidence.",
                "valid_items": 0,
                "total_items": len(dataset),
            }
        )
