from typing import List

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.logger import logger


class FormatBiasTest(BaseBiasTest):
    """
    Measures format bias: Does the judge score markdown/bulleted responses higher than plain prose?
    Test method: Format the baseline-losing response with markdown (bolding, bullet points);
    measure preference flip rate toward formatted losers.
    """

    @property
    def name(self) -> str:
        return "Format bias"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info(f"Running Format Bias test with {judge.model_name} on {len(dataset)} items.")

        switches_to_formatted_loser = 0
        switches_to_formatted_winner = 0
        valid_items = 0

        for item in dataset:
            # Baseline preference
            pref_baseline = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)

            if pref_baseline not in ("A", "B"):
                continue

            valid_items += 1

            # Treatment 1: format the *losing* response.
            loser_formatted_a = item.response_a
            loser_formatted_b = item.response_b
            if pref_baseline == "A":
                loser_formatted_b = self._apply_markdown_formatting(item.response_b)
            else:
                loser_formatted_a = self._apply_markdown_formatting(item.response_a)

            pref_loser_formatted = judge.evaluate_pairwise(
                item.prompt,
                loser_formatted_a,
                loser_formatted_b,
            )

            if (pref_baseline == "A" and pref_loser_formatted == "B") or (
                pref_baseline == "B" and pref_loser_formatted == "A"
            ):
                switches_to_formatted_loser += 1

            # Treatment 2: format the *winning* response to control for general formatting preference.
            winner_formatted_a = item.response_a
            winner_formatted_b = item.response_b
            if pref_baseline == "A":
                winner_formatted_a = self._apply_markdown_formatting(item.response_a)
            else:
                winner_formatted_b = self._apply_markdown_formatting(item.response_b)

            pref_winner_formatted = judge.evaluate_pairwise(
                item.prompt,
                winner_formatted_a,
                winner_formatted_b,
            )
            if (pref_baseline == "A" and pref_winner_formatted == "B") or (
                pref_baseline == "B" and pref_winner_formatted == "A"
            ):
                switches_to_formatted_winner += 1

        if valid_items == 0:
            logger.warning("No valid items (with clear baseline preference) to compute format bias.")
            score = 0.0
        else:
            raw_loser_rate = switches_to_formatted_loser / valid_items
            raw_winner_rate = switches_to_formatted_winner / valid_items
            score = max(0.0, raw_loser_rate - raw_winner_rate)

        logger.info(
            "Format Bias score: %.2f (%s loser-switches, %s winner-switches, %s valid).",
            score,
            switches_to_formatted_loser,
            switches_to_formatted_winner,
            valid_items,
        )

        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "switches_to_formatted_loser": switches_to_formatted_loser,
                "switches_to_formatted_winner": switches_to_formatted_winner,
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
