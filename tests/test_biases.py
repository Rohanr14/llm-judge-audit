import pytest

from llm_judge_audit.biases.position import PositionBiasTest
from llm_judge_audit.biases.verbosity import VerbosityBiasTest
from llm_judge_audit.datasets.schema import AnchorDatasetItem, HumanAnnotation
from llm_judge_audit.judge import BaseJudge

@pytest.fixture
def dummy_dataset():
    return [
        AnchorDatasetItem(
            item_id="1",
            domain="code",
            prompt="Write a loop",
            response_a="Response A",
            response_b="Response B",
            human_annotations=[
                HumanAnnotation(rater_id="r1", preference="A"),
                HumanAnnotation(rater_id="r2", preference="A"),
                HumanAnnotation(rater_id="r3", preference="A"),
            ],
            majority_preference="A"
        ),
        AnchorDatasetItem(
            item_id="2",
            domain="factual",
            prompt="Capital of France?",
            response_a="Paris",
            response_b="London",
            human_annotations=[
                HumanAnnotation(rater_id="r1", preference="A"),
                HumanAnnotation(rater_id="r2", preference="A"),
                HumanAnnotation(rater_id="r3", preference="A"),
            ],
            majority_preference="A"
        )
    ]

class MockJudge(BaseJudge):
    def __init__(self, evaluate_func):
        super().__init__(model_name="mock-judge")
        self.evaluate_func = evaluate_func

    def evaluate_pairwise(self, prompt, a, b):
        return self.evaluate_func(prompt, a, b)


def test_position_bias(dummy_dataset):
    # Mock judge always prefers whichever is passed in as "A" (the first parameter after prompt)
    def always_first(prompt, a, b):
        return "A"
    
    judge = MockJudge(always_first)
    test = PositionBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # pref_1 will be "A", pref_2 will be "A". This is a reversal of *content* preference
    # to stick with the first position. So score should be 1.0 (100% position bias).
    assert result.score == 1.0
    assert result.details["reversals"] == 2


def test_position_bias_no_bias(dummy_dataset):
    # Mock judge always prefers the actual 'Response A' or 'Paris' content
    def prefers_content_a(prompt, a, b):
        if a in ("Response A", "Paris"):
            return "A"
        elif b in ("Response A", "Paris"):
            return "B"
        return "Tie"
            
    judge = MockJudge(prefers_content_a)
    test = PositionBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # Consistent preference based on content, not position
    assert result.score == 0.0


def test_verbosity_bias(dummy_dataset):
    # Mock judge prefers the longer response
    def prefers_longer(prompt, a, b):
        if len(a) > len(b):
            return "A"
        elif len(b) > len(a):
            return "B"
        return "A" # baseline tie breaker
        
    judge = MockJudge(prefers_longer)
    test = VerbosityBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # Since we pad the loser, and it prefers longer, it will always switch to the padded loser.
    # Score should be 1.0.
    assert result.score == 1.0


def test_verbosity_bias_no_bias(dummy_dataset):
    # Mock judge always prefers actual 'Response A' content, regardless of length
    def prefers_content_a(prompt, a, b):
        # We use a simple substring check because the string might be padded
        if "Response A" in a or "Paris" in a:
            return "A"
        if "Response A" in b or "Paris" in b:
            return "B"
        return "Tie"
        
    judge = MockJudge(prefers_content_a)
    test = VerbosityBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # No switches
    assert result.score == 0.0
