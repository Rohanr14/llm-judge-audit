from typing import List
from collections import Counter

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.logger import logger

class CrossRunConsistencyTest(BaseBiasTest):
    """
    Measures cross-run consistency: Does the judge give the same score on the same input twice?
    Test method: Re-submit identical inputs; measure score variance across N runs.
    """

    def __init__(self, n_runs: int = 3):
        if n_runs < 2:
            raise ValueError("n_runs must be at least 2 to measure consistency.")
        self.n_runs = n_runs

    @property
    def name(self) -> str:
        return "Cross-run consistency"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info(f"Running Cross-Run Consistency test with {judge.model_name} on {len(dataset)} items ({self.n_runs} runs each).")
        
        inconsistent_items = 0
        
        for item in dataset:
            preferences = []
            for _ in range(self.n_runs):
                pref = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)
                preferences.append(pref)
            
            # Count unique non-tie preferences
            unique_prefs = set(p for p in preferences if p != "Tie")
            
            if len(unique_prefs) > 1:
                inconsistent_items += 1
                
        if len(dataset) == 0:
            score = 0.0
        else:
            # The score is the fraction of items that showed inconsistency.
            score = inconsistent_items / len(dataset)
            
        logger.info(f"Cross-Run Consistency score: {score:.2f} ({inconsistent_items}/{len(dataset)} inconsistent items)")
        
        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "inconsistent_items": inconsistent_items,
                "total_items": len(dataset),
                "n_runs": self.n_runs,
            }
        )
