from typing import List
import re

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.logger import logger

class FormatBiasTest(BaseBiasTest):
    """
    Measures format bias: Does the judge score markdown/bulleted responses higher than plain prose?
    Test method: Format the baseline-losing response with markdown (bolding, bullet points); measure preference flip rate toward formatted losers.
    """

    @property
    def name(self) -> str:
        return "Format bias"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info(f"Running Format Bias test with {judge.model_name} on {len(dataset)} items.")
        
        switches_to_formatted = 0
        valid_items = 0
        
        for item in dataset:
            # Baseline preference
            pref_baseline = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)
            
            if pref_baseline not in ("A", "B"):
                continue
                
            valid_items += 1
            
            # Format the *losing* response
            formatted_a = item.response_a
            formatted_b = item.response_b
            
            if pref_baseline == "A":
                # A won, so format B
                formatted_b = self._apply_markdown_formatting(item.response_b)
            else:
                # B won, so format A
                formatted_a = self._apply_markdown_formatting(item.response_a)
                
            # Re-evaluate
            pref_formatted = judge.evaluate_pairwise(item.prompt, formatted_a, formatted_b)
            
            # Did the judge switch its preference to the now-formatted loser?
            if pref_baseline == "A" and pref_formatted == "B":
                switches_to_formatted += 1
            elif pref_baseline == "B" and pref_formatted == "A":
                switches_to_formatted += 1
                
        if valid_items == 0:
            logger.warning("No valid items (with clear baseline preference) to compute format bias.")
            score = 0.0
        else:
            score = switches_to_formatted / valid_items
            
        logger.info(f"Format Bias score: {score:.2f} ({switches_to_formatted}/{valid_items} switches to formatted response)")
        
        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "switches_to_formatted": switches_to_formatted,
                "valid_items": valid_items,
                "total_items": len(dataset),
            }
        )

    def _apply_markdown_formatting(self, text: str) -> str:
        """
        Heuristically applies markdown formatting (headers, bolding, bullets) to a plain text string.
        """
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return text
            
        formatted_lines = ["### **Key Points**\n"]
        for line in lines:
            # Bold the first few words of the line to simulate styled text
            words = line.split()
            if len(words) > 3:
                bold_part = " ".join(words[:3])
                rest = " ".join(words[3:])
                formatted_line = f"- **{bold_part}** {rest}"
            else:
                formatted_line = f"- {line}"
            formatted_lines.append(formatted_line)
            
        return "\n".join(formatted_lines)
