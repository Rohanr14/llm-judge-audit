"""Pydantic schema for the anchor dataset.

The schema is deliberately permissive about annotation completeness so that
``anchor.json`` can be loaded at any point in the Prolific pipeline:

* Freshly built (``download.py``) -- zero human annotations yet.
* Mid-study -- some items fully rated, others still in-flight.
* Final -- every item has >= 3 annotations.

The ``min_length=3`` invariant from the previous version blew up on every
one of those intermediate states, which was the wrong place to enforce it:
rater coverage is a *scoring* concern, not a schema concern. HAS now filters
out items with fewer than 1 annotation at compute time (``has.py``).

The schema also carries dataset-provenance fields that ``download.py`` now
emits (``source_winner``, ``chosen_slot``) so downstream code can recover the
dataset-preferred answer even after the per-item position randomisation.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class HumanAnnotation(BaseModel):
    """Represents a single human rater's evaluation."""

    rater_id: str = Field(..., description="Unique identifier for the human rater (e.g., Prolific ID)")
    preference: Literal["A", "B", "Tie"] = Field(..., description="The rater's chosen preference")
    confidence: Optional[int] = Field(None, ge=1, le=5, description="Rater's confidence score (1-5)")
    rationale: Optional[str] = Field(
        None, description="Optional explanation for why this preference was chosen"
    )


class AnchorDatasetItem(BaseModel):
    """A single pairwise-comparison item.

    Ground truth for HAS comes from ``majority_preference``, which is
    derived from ``human_annotations`` once enough raters have submitted.
    """

    item_id: str = Field(..., description="Unique identifier for the evaluation item")
    domain: Literal["code", "factual", "creative"] = Field(
        ..., description="Domain for stratified evaluation"
    )
    difficulty: Literal["easy", "medium", "hard"] = Field(
        "medium", description="Estimated difficulty of the prompt"
    )

    prompt: str = Field(..., description="The user instruction or query")
    response_a: str = Field(..., description="The first response to evaluate")
    response_b: str = Field(..., description="The second response to evaluate")

    # Internal metadata; hide from the judge to prevent self-enhancement bias.
    model_a_family: Optional[str] = Field(
        None, description="Model family that generated response A (e.g. 'gpt', 'claude')"
    )
    model_b_family: Optional[str] = Field(
        None, description="Model family that generated response B (e.g. 'gemini', 'llama')"
    )

    # Optional dataset-provenance ground truth. lmsys/mt_bench provide a
    # 'winner' column; hh_rlhf labels one response as 'chosen'. After slot
    # randomisation ``chosen_slot`` still points at whichever slot holds the
    # dataset-preferred answer, so downstream code can recover that signal.
    source_winner: Optional[Literal["A", "B", "Tie"]] = Field(
        None, description="Dataset-provided winner label after slot randomisation."
    )
    chosen_slot: Optional[Literal["A", "B"]] = Field(
        None, description="Which slot holds the dataset-preferred answer."
    )

    # Ground-truth data. ``human_annotations`` may be empty for items that
    # haven't been through the Prolific pipeline yet; HAS scoring is
    # responsible for filtering those out rather than having schema load fail.
    human_annotations: List[HumanAnnotation] = Field(
        default_factory=list,
        description="Individual human ratings. HAS requires >= 1 to score an item; "
        "the study targets >= 3 raters per item for a stable majority.",
    )
    majority_preference: Literal["A", "B", "Tie"] = Field(
        "Tie",
        description="Aggregated ground truth from human_annotations. Defaults to 'Tie' "
        "on unannotated items; HAS explicitly excludes tie-majority items by default.",
    )
    is_gold_standard: bool = Field(
        False,
        description="Whether this item has 100% human consensus (e.g. 3/3 agree). "
        "Used for stricter HAS weighting.",
    )

    @model_validator(mode="after")
    def _check_consensus(self) -> "AnchorDatasetItem":
        # Consistency check: ``is_gold_standard`` should only be True when
        # every rater agrees *and* the rater count is non-zero.
        if self.is_gold_standard:
            prefs = {a.preference for a in self.human_annotations}
            if len(self.human_annotations) == 0 or len(prefs) != 1:
                # Don't raise -- just silently demote to non-gold. This makes
                # the schema tolerant of partial/mid-study exports.
                object.__setattr__(self, "is_gold_standard", False)
        return self


class AnchorDataset(BaseModel):
    """The complete human-annotated anchor dataset."""

    version: str = Field("1.0", description="Version of the dataset")
    items: List[AnchorDatasetItem] = Field(
        ...,
        description="The collection of items. Expected to be ~100 items (balanced across domains).",
    )

    def get_items_by_domain(
        self, domain: Literal["code", "factual", "creative"]
    ) -> List[AnchorDatasetItem]:
        """Helper to fetch items stratified by domain."""
        return [item for item in self.items if item.domain == domain]
