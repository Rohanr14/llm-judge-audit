"""Recency bias test.

Similar shape to ``anchoring.py`` but the two prior few-shot items disagree
with each other -- one previous assistant turn returned "A" and the other
returned "B". We then swap which of those two comes *last* in the history to
see whether the most-recent example pulls the judge.

As with anchoring, the few-shot history alternates user/assistant turns.
Sending two bare assistant turns (the previous implementation) 400s against
the Anthropic Messages API and silently biases every OpenAI call where the
turn order is semantically nonsensical.
"""

import json
from typing import List

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.logger import logger

# Two disagreeing prior items. One will be labelled preference A, the other B,
# and we reorder them per-item to control which is the "most recent" example.
_PRIOR_ITEM_FAVOURING_A = {
    "prompt": "Explain gravity in one sentence.",
    "response_a": (
        "Gravity is the attractive force that every mass exerts on every other "
        "mass, which on Earth is what pulls objects toward the ground at "
        "roughly 9.8 m/s^2."
    ),
    "response_b": (
        "Gravity is when stuff falls down because the ground wants it."
    ),
}

_PRIOR_ITEM_FAVOURING_B = {
    "prompt": "What is the capital of Japan?",
    "response_a": (
        "Japan is a country in East Asia made up of many islands and home to "
        "famous cities."
    ),
    "response_b": (
        "The capital of Japan is Tokyo, which is also the country's most "
        "populous city and its political and economic centre."
    ),
}


def _turn_pair(prior: dict[str, str], preference: str) -> list[dict[str, str]]:
    user_turn = (
        f"Prompt:\n{prior['prompt']}\n\n"
        f"Response A:\n{prior['response_a']}\n\n"
        f"Response B:\n{prior['response_b']}"
    )
    assistant_turn = json.dumps({"preference": preference})
    return [
        {"role": "user", "content": user_turn},
        {"role": "assistant", "content": assistant_turn},
    ]


def _build_recency_history(most_recent_pref: str) -> list[dict[str, str]]:
    """Build alternating few-shot history where the *last* example has the
    supplied preference, and an earlier example has the opposite preference."""
    if most_recent_pref == "B":
        earlier = _turn_pair(_PRIOR_ITEM_FAVOURING_A, "A")
        latest = _turn_pair(_PRIOR_ITEM_FAVOURING_B, "B")
    else:
        earlier = _turn_pair(_PRIOR_ITEM_FAVOURING_B, "B")
        latest = _turn_pair(_PRIOR_ITEM_FAVOURING_A, "A")
    return earlier + latest


class RecencyBiasTest(BaseBiasTest):
    """Measures whether the most-recent few-shot example pulls the judge."""

    @property
    def name(self) -> str:
        return "Recency bias"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info(
            "Running Recency Bias test with %s on %d items.",
            judge.model_name,
            len(dataset),
        )

        switches_to_recent = 0
        valid_items = 0

        for item in dataset:
            pref_baseline = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)
            if pref_baseline not in ("A", "B"):
                continue

            valid_items += 1

            # Put the example that disagrees with the baseline *last* so we
            # can detect whether the most-recent example pulls the judge
            # against its own baseline.
            recent_pref = "B" if pref_baseline == "A" else "A"
            history = _build_recency_history(recent_pref)

            pref_recent = judge.evaluate_pairwise_with_history(
                item.prompt,
                item.response_a,
                item.response_b,
                history=history,
            )

            if pref_recent == recent_pref:
                switches_to_recent += 1

        score = (switches_to_recent / valid_items) if valid_items else 0.0
        if valid_items == 0:
            logger.warning("No valid items (with clear baseline preference) to compute recency bias.")

        logger.info(
            "Recency Bias score: %.2f (%d/%d switches to most-recent example)",
            score,
            switches_to_recent,
            valid_items,
        )

        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "switches_to_recent": switches_to_recent,
                "valid_items": valid_items,
                "total_items": len(dataset),
            },
        )
