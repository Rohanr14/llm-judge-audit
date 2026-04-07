from pydantic import ValidationError
import pytest
from llm_judge_audit.datasets.schema import HumanAnnotation, AnchorDatasetItem, AnchorDataset

def test_human_annotation_valid():
    annotation = HumanAnnotation(
        rater_id="rater_1",
        preference="A",
        confidence=5,
        rationale="Much more concise and accurate."
    )
    assert annotation.preference == "A"
    assert annotation.confidence == 5

def test_human_annotation_invalid_confidence():
    with pytest.raises(ValidationError):
        HumanAnnotation(
            rater_id="rater_1",
            preference="A",
            confidence=6  # Invalid, max is 5
        )

def test_anchor_dataset_item_valid():
    item = AnchorDatasetItem(
        item_id="code-001",
        domain="code",
        difficulty="medium",
        prompt="Write a Python script to add two numbers.",
        response_a="def add(a, b): return a + b",
        response_b="print('hello')",
        model_a_family="gpt",
        model_b_family="unknown",
        human_annotations=[
            HumanAnnotation(rater_id="r1", preference="A", confidence=5),
            HumanAnnotation(rater_id="r2", preference="A", confidence=4),
            HumanAnnotation(rater_id="r3", preference="A", confidence=5),
        ],
        majority_preference="A",
        is_gold_standard=True
    )
    assert item.item_id == "code-001"
    assert len(item.human_annotations) == 3

def test_anchor_dataset_item_min_annotations():
    with pytest.raises(ValidationError):
        AnchorDatasetItem(
            item_id="code-002",
            domain="factual",
            difficulty="easy",
            prompt="What is the capital of France?",
            response_a="Paris",
            response_b="London",
            model_a_family="gpt",
            model_b_family="claude",
            human_annotations=[
                HumanAnnotation(rater_id="r1", preference="A", confidence=5),
                # Only 1 annotation, minimum is 3
            ],
            majority_preference="A"
        )

def test_anchor_dataset_domain_filter():
    dataset = AnchorDataset(
        version="1.0",
        items=[
            AnchorDatasetItem(
                item_id="code-001", 
                domain="code", 
                difficulty="easy", 
                prompt="A", 
                response_a="A", 
                response_b="B",
                model_a_family="gpt",
                model_b_family="gpt",
                human_annotations=[HumanAnnotation(rater_id=f"r{i}", preference="A") for i in range(3)], 
                majority_preference="A"
            ),
            AnchorDatasetItem(
                item_id="creative-001", 
                domain="creative", 
                difficulty="hard", 
                prompt="B", 
                response_a="C", 
                response_b="D",
                model_a_family="gpt",
                model_b_family="claude",
                human_annotations=[HumanAnnotation(rater_id=f"r{i}", preference="B") for i in range(3)], 
                majority_preference="B"
            )
        ]
    )
    
    code_items = dataset.get_items_by_domain("code")
    assert len(code_items) == 1
    assert code_items[0].item_id == "code-001"

    creative_items = dataset.get_items_by_domain("creative")
    assert len(creative_items) == 1
    assert creative_items[0].item_id == "creative-001"
