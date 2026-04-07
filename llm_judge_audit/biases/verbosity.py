from typing import List

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.datasets.schema import AnchorDatasetItem
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
        logger.info(f"Running Verbosity Bias test with {judge.model_name} on {len(dataset)} items.")
        
        filler_text = (
            " This is an additional sentence to make the response longer, but it does not "
            "add any substantial new information or change the core argument being presented. "
            "In conclusion, we can see that the points made are quite comprehensive and "
            "detailed, offering a thorough explanation."
        )
        
        switches_to_padded = 0
        valid_items = 0
        
        for item in dataset:
            # Baseline preference
            pref_baseline = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)
            
            # If baseline is Tie, it's hard to measure a pure switch, but we can see if padding
            # breaks the tie. For simplicity, we only consider clear initial preferences.
            if pref_baseline not in ("A", "B"):
                continue
                
            valid_items += 1
            
            # Pad the *losing* response
            padded_a = item.response_a
            padded_b = item.response_b
            
            if pref_baseline == "A":
                # A won, so pad B
                padded_b += filler_text
            else:
                # B won, so pad A
                padded_a += filler_text
                
            # Re-evaluate
            pref_padded = judge.evaluate_pairwise(item.prompt, padded_a, padded_b)
            
            # Did the judge switch its preference to the now-padded loser?
            if pref_baseline == "A" and pref_padded == "B":
                switches_to_padded += 1
            elif pref_baseline == "B" and pref_padded == "A":
                switches_to_padded += 1
                
        if valid_items == 0:
            logger.warning("No valid items (with clear baseline preference) to compute verbosity bias.")
            score = 0.0
        else:
            score = switches_to_padded / valid_items
            
        logger.info(f"Verbosity Bias score: {score:.2f} ({switches_to_padded}/{valid_items} switches to padded response)")
        
        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "switches_to_padded": switches_to_padded,
                "valid_items": valid_items,
                "total_items": len(dataset),
            }
        )
