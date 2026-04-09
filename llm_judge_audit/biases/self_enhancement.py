from typing import List

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.logger import logger

class SelfEnhancementBiasTest(BaseBiasTest):
    """
    Measures self-enhancement bias: Does the judge favor outputs from its own model family?
    Test method: Find cases where one response is from the judge's model family and the other is not. 
    Measure how often it prefers its own family compared to the human ground truth.
    """

    @property
    def name(self) -> str:
        return "Self-enhancement bias"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info(f"Running Self-Enhancement Bias test with {judge.model_name} on {len(dataset)} items.")
        
        # Determine the judge's model family
        judge_family = self._get_model_family(judge.model_name)
        if not judge_family:
            logger.warning(f"Could not determine model family for {judge.model_name}. Cannot measure self-enhancement.")
            return BiasTestResult(
                bias_name=self.name,
                score=0.0,
                details={
                    "error": "Unknown judge model family",
                    "judge_model": judge.model_name
                }
            )

        self_preference_count = 0
        human_self_preference_count = 0
        valid_items = 0
        
        for item in dataset:
            # We only care about items where exactly one of the responses is from the judge's family
            a_is_self = (item.model_a_family == judge_family)
            b_is_self = (item.model_b_family == judge_family)
            
            if a_is_self == b_is_self:
                continue # Both are self, or neither are self
                
            valid_items += 1
            
            pref = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)
            
            # Did the judge prefer its own family?
            if (pref == "A" and a_is_self) or (pref == "B" and b_is_self):
                self_preference_count += 1
                
            # Did the humans prefer the judge's family?
            if (item.majority_preference == "A" and a_is_self) or (item.majority_preference == "B" and b_is_self):
                human_self_preference_count += 1
                
        if valid_items == 0:
            logger.warning(f"No items found where exactly one response is from the '{judge_family}' family.")
            score = 0.0
        else:
            # Score is the difference in preference rate between the judge and humans.
            # E.g., if judge chooses itself 80% of the time, but humans chose it 50% of the time, score is 0.30.
            # We clamp at 0 so we don't return negative bias if the judge is self-deprecating.
            judge_rate = self_preference_count / valid_items
            human_rate = human_self_preference_count / valid_items
            score = max(0.0, judge_rate - human_rate)
            
        logger.info(f"Self-Enhancement Bias score: {score:.2f} "
                    f"(Judge preferred self {self_preference_count}/{valid_items} times; "
                    f"Humans preferred self {human_self_preference_count}/{valid_items} times)")
        
        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "judge_family": judge_family,
                "judge_self_preferences": self_preference_count,
                "human_self_preferences": human_self_preference_count,
                "valid_items": valid_items,
                "total_items": len(dataset),
            }
        )

    def _get_model_family(self, model_name: str) -> str | None:
        """Heuristic to map a model name to its family."""
        name = model_name.lower()
        if "gpt" in name or "o1" in name or "o3" in name:
            return "gpt"
        if "claude" in name:
            return "claude"
        if "gemini" in name:
            return "gemini"
        if "llama" in name:
            return "llama"
        if "mistral" in name or "mixtral" in name:
            return "mistral"
        if "qwen" in name:
            return "qwen"
        if "command" in name or "cohere" in name:
            return "cohere"
        return None
