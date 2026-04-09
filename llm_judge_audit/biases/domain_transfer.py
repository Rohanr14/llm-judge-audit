from typing import List
from collections import defaultdict

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.logger import logger

class DomainTransferBiasTest(BaseBiasTest):
    """
    Measures domain transfer bias: Does the judge's reliability hold across domains?
    Test method: Evaluate code, factual, and creative domains; measure score distribution shift.
    """

    @property
    def name(self) -> str:
        return "Domain transfer bias"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info(f"Running Domain Transfer Bias test with {judge.model_name} on {len(dataset)} items.")
        
        domain_correct = defaultdict(int)
        domain_total = defaultdict(int)
        
        for item in dataset:
            pref = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)
            domain_total[item.domain] += 1
            if pref == item.majority_preference:
                domain_correct[item.domain] += 1
                
        if not domain_total:
            logger.warning("No items to compute domain transfer bias.")
            return BiasTestResult(
                bias_name=self.name,
                score=0.0,
                details={
                    "error": "No items in dataset",
                    "valid_items": 0,
                    "total_items": len(dataset),
                }
            )
            
        domain_accuracies = {}
        for domain, total in domain_total.items():
            domain_accuracies[domain] = domain_correct[domain] / total
            
        if len(domain_accuracies) < 2:
            logger.warning("Need at least 2 domains to measure domain transfer bias.")
            score = 0.0
        else:
            # Score is the maximum difference in accuracy between any two domains
            max_acc = max(domain_accuracies.values())
            min_acc = min(domain_accuracies.values())
            score = max_acc - min_acc
            
        logger.info(f"Domain Transfer Bias score: {score:.2f} (accuracies: {domain_accuracies})")
        
        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "domain_accuracies": domain_accuracies,
                "valid_items": sum(domain_total.values()),
                "total_items": len(dataset),
            }
        )
