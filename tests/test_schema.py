import pytest
from pydantic import ValidationError

from llm_judge_audit.datasets.schema import AnchorDataset, AnchorDatasetItem, HumanAnnotation


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

def test_anchor_dataset_item_accepts_partial_annotations():
    # Schema is permissive about annotation counts so mid-study Prolific
    # exports load cleanly. HAS enforces rater coverage at score time, not
    # at schema time.
    item = AnchorDatasetItem(
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
        ],
        majority_preference="A",
    )
    assert len(item.human_annotations) == 1


def test_anchor_dataset_item_accepts_empty_annotations():
    # Freshly built anchor.json has zero annotations and must still parse.
    item = AnchorDatasetItem(
        item_id="code-003",
        domain="creative",
        difficulty="easy",
        prompt="Write a haiku about autumn.",
        response_a="Leaves crunch under foot / the morning air has bite now / I miss the summer",
        response_b="Fall is here and it feels cool and the leaves are falling to the ground fast.",
    )
    assert item.human_annotations == []
    assert item.majority_preference == "Tie"
    assert item.is_gold_standard is False


def test_anchor_dataset_item_carries_source_winner_and_chosen_slot():
    item = AnchorDatasetItem(
        item_id="hh-001",
        domain="factual",
        difficulty="medium",
        prompt="Why is the sky blue?",
        response_a="Rayleigh scattering: shorter wavelengths scatter more.",
        response_b="Because it is.",
        source_winner="A",
        chosen_slot="A",
    )
    assert item.source_winner == "A"
    assert item.chosen_slot == "A"

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
