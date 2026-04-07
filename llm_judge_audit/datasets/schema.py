from typing import Literal, List, Optional
from pydantic import BaseModel, Field

class HumanAnnotation(BaseModel):
    """Represents a single human rater's evaluation."""
    rater_id: str = Field(..., description="Unique identifier for the human rater (e.g., Prolific ID)")
    preference: Literal["A", "B", "Tie"] = Field(..., description="The rater's chosen preference")
    confidence: Optional[int] = Field(None, ge=1, le=5, description="Rater's confidence score (1-5)")
    rationale: Optional[str] = Field(None, description="Optional explanation for why this preference was chosen")

class AnchorDatasetItem(BaseModel):
    """
    Represents a single pairwise comparison item in the anchor dataset.
    This serves as the ground-truth for computing the Human Alignment Score (HAS).
    """
    item_id: str = Field(..., description="Unique identifier for the evaluation item (e.g., 'code-001')")
    domain: Literal["code", "factual", "creative"] = Field(..., description="The specific domain to enable stratified evaluation")
    difficulty: Literal["easy", "medium", "hard"] = Field("medium", description="Estimated difficulty of the prompt")
    
    prompt: str = Field(..., description="The user instruction or query")
    response_a: str = Field(..., description="The first response to evaluate")
    response_b: str = Field(..., description="The second response to evaluate")
    
    # Metadata for internal tracking (should typically be hidden from the judge to prevent self-enhancement bias in baseline testing)
    model_a_family: Optional[str] = Field(None, description="Model family that generated response A (e.g. 'gpt', 'claude')")
    model_b_family: Optional[str] = Field(None, description="Model family that generated response B (e.g. 'gemini', 'llama')")
    
    # Ground truth data
    human_annotations: List[HumanAnnotation] = Field(
        ..., 
        min_length=3, 
        description="Individual human ratings, minimum 3 per item to ensure a stable majority"
    )
    majority_preference: Literal["A", "B", "Tie"] = Field(
        ..., 
        description="The aggregated ground truth preference based on the majority vote of human_annotations"
    )
    is_gold_standard: bool = Field(False, description="Whether this item had 100% human consensus (e.g. 3/3 agree). Can be used for stricter HAS weighting.")

class AnchorDataset(BaseModel):
    """The complete human-annotated anchor dataset."""
    version: str = Field("1.0", description="Version of the dataset")
    items: List[AnchorDatasetItem] = Field(
        ..., 
        description="The collection of items. Expected to be ~100 items (balanced across domains)."
    )

    def get_items_by_domain(self, domain: Literal["code", "factual", "creative"]) -> List[AnchorDatasetItem]:
        """Helper to fetch items stratified by domain."""
        return [item for item in self.items if item.domain == domain]
