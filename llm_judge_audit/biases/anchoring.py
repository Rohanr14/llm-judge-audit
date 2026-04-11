"""Anchoring bias test.

We probe whether a streak of prior judgements inside the same session can drag
the judge toward a particular answer on an unrelated new item. To do that we
construct a short few-shot conversation that alternates ``user``/``assistant``
turns -- two fake "prior items" where the assistant always returned the same
preference -- and then ask the judge to evaluate the real item.

The history *must* alternate user/assistant turns because the Anthropic
Messages API rejects consecutive assistant messages with a 400 error; the
previous version of this test sent two bare assistant messages and 400'd on
every Anthropic call. The assistant replies are also valid preference JSON so
they match what the judge itself produces, which is what makes the few-shot
"anchor" plausible to the model.
"""

import json
from typing import List

from llm_judge_audit.biases.base import BaseBiasTest, BiasTestResult
from llm_judge_audit.datasets.schema import AnchorDatasetItem
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.logger import logger

# Two synthetic prior items. Kept short so they don't dominate the context but
# long enough that each response is a plausible "evaluation".
_PRIOR_ITEMS = [
    {
        "prompt": "Explain in one sentence why the sky appears blue during the day.",
        "response_a": (
            "Sunlight contains all colours of the visible spectrum, and when it "
            "enters Earth's atmosphere the shorter blue wavelengths are scattered "
            "in all directions by air molecules more strongly than the longer "
            "red wavelengths, which is why we see a blue sky."
        ),
        "response_b": (
            "The sky is blue because the air is blue and reflects that colour "
            "back at us."
        ),
    },
    {
        "prompt": "Write a short greeting for a customer-support email.",
        "response_a": (
            "Hello! Thanks so much for reaching out to our support team. I hope "
            "your day is going well -- I'd love to help you sort this out as "
            "quickly as possible."
        ),
        "response_b": (
            "Hi there, thanks for contacting support. I'll take a look at your "
            "issue and get back to you shortly with next steps."
        ),
    },
]


def _build_anchoring_history(anchor_pref: str) -> list[dict[str, str]]:
    """Build an alternating user/assistant few-shot history.

    Every assistant turn returns the same ``anchor_pref`` -- that's the
    "anchor" we're testing whether the judge will drift towards.
    """
    history: list[dict[str, str]] = []
    for prior in _PRIOR_ITEMS:
        user_turn = (
            f"Prompt:\n{prior['prompt']}\n\n"
            f"Response A:\n{prior['response_a']}\n\n"
            f"Response B:\n{prior['response_b']}"
        )
        history.append({"role": "user", "content": user_turn})
        history.append(
            {"role": "assistant", "content": json.dumps({"preference": anchor_pref})}
        )
    return history


class AnchoringBiasTest(BaseBiasTest):
    """Measures whether prior answers in the session pull new judgements.

    For each item we first take a clean baseline preference, then replay the
    item with an anchoring history of two prior "decisions" that all went
    *against* the baseline. The score is the fraction of items where the judge
    switched to match the anchor -- higher = more anchoring bias.
    """

    @property
    def name(self) -> str:
        return "Anchoring bias"

    def run(self, judge: BaseJudge, dataset: List[AnchorDatasetItem]) -> BiasTestResult:
        logger.info(
            "Running Anchoring Bias test with %s on %d items.",
            judge.model_name,
            len(dataset),
        )

        switches_to_anchor = 0
        valid_items = 0

        for item in dataset:
            pref_baseline = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)
            if pref_baseline not in ("A", "B"):
                continue

            valid_items += 1
            anchor_target = "B" if pref_baseline == "A" else "A"
            history = _build_anchoring_history(anchor_target)

            pref_anchored = judge.evaluate_pairwise_with_history(
                item.prompt,
                item.response_a,
                item.response_b,
                history=history,
            )

            if pref_anchored == anchor_target:
                switches_to_anchor += 1

        score = (switches_to_anchor / valid_items) if valid_items else 0.0
        if valid_items == 0:
            logger.warning("No valid items to compute anchoring bias.")

        logger.info(
            "Anchoring Bias score: %.2f (%d/%d switches to anchor)",
            score,
            switches_to_anchor,
            valid_items,
        )

        return BiasTestResult(
            bias_name=self.name,
            score=score,
            details={
                "switches_to_anchor": switches_to_anchor,
                "valid_items": valid_items,
                "total_items": len(dataset),
            },
        )
