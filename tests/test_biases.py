import pytest

from llm_judge_audit.biases.position import PositionBiasTest
from llm_judge_audit.biases.verbosity import VerbosityBiasTest
from llm_judge_audit.biases.cross_run import CrossRunConsistencyTest
from llm_judge_audit.biases.sycophancy import SycophancyBiasTest
from llm_judge_audit.biases.self_enhancement import SelfEnhancementBiasTest
from llm_judge_audit.biases.recency import RecencyBiasTest
from llm_judge_audit.biases.format_bias import FormatBiasTest
from llm_judge_audit.biases.anchoring import AnchoringBiasTest
from llm_judge_audit.biases.confidence_gap import ConfidenceGapTest
from llm_judge_audit.biases.domain_transfer import DomainTransferBiasTest
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
            majority_preference="A",
            model_a_family="gpt",
            model_b_family="claude"
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
            majority_preference="A",
            model_a_family="gemini",
            model_b_family="gpt"
        )
    ]

class MockJudge(BaseJudge):
    def __init__(self, evaluate_func, model_name="mock-judge"):
        super().__init__(model_name=model_name)
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
    assert result.details["position_locked_choices"] == 2


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


def test_cross_run_consistency_bias(dummy_dataset):
    # Stateful mock judge that alternates between A and B
    class AlternatingMockJudge(BaseJudge):
        def __init__(self):
            super().__init__(model_name="mock-judge")
            self.call_count = 0
            
        def evaluate_pairwise(self, prompt, a, b):
            self.call_count += 1
            if self.call_count % 2 == 1:
                return "A"
            else:
                return "B"
                
    judge = AlternatingMockJudge()
    test = CrossRunConsistencyTest(n_runs=3)
    result = test.run(judge, dummy_dataset)
    
    # It alternates A, B, A for item 1 -> unique_prefs = {"A", "B"} -> inconsistent
    # It alternates B, A, B for item 2 -> unique_prefs = {"A", "B"} -> inconsistent
    # Score should be 1.0.
    assert result.score == 1.0
    assert result.details["inconsistent_items"] == 2


def test_cross_run_consistency_no_bias(dummy_dataset):
    # Mock judge that is perfectly consistent
    def always_a(prompt, a, b):
        return "A"
        
    judge = MockJudge(always_a)
    test = CrossRunConsistencyTest(n_runs=3)
    result = test.run(judge, dummy_dataset)
    
    # Never changes its mind
    assert result.score == 0.0
    assert result.details["inconsistent_items"] == 0


def test_cross_run_invalid_runs():
    with pytest.raises(ValueError, match="n_runs must be at least 2"):
        CrossRunConsistencyTest(n_runs=1)


def test_sycophancy_bias(dummy_dataset):
    # Mock judge that capitulates to whatever is suggested in the prompt
    def sycophantic_judge(prompt, a, b):
        if "strongly preferred Response A" in prompt:
            return "A"
        elif "strongly preferred Response B" in prompt:
            return "B"
        return "A" # Default baseline preference

    judge = MockJudge(sycophantic_judge)
    test = SycophancyBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # Baseline is A. It will suggest B. The judge will switch to B. Score should be 1.0.
    assert result.score == 1.0
    assert result.details["switches_to_anchor"] == 2


def test_sycophancy_bias_no_bias(dummy_dataset):
    # Mock judge that ignores the suggestion and sticks to its baseline
    def resolute_judge(prompt, a, b):
        return "A"

    judge = MockJudge(resolute_judge)
    test = SycophancyBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # Baseline is A. It will suggest B. The judge sticks to A. Score should be 0.0.
    assert result.score == 0.0
    assert result.details["switches_to_anchor"] == 0


def test_self_enhancement_bias(dummy_dataset):
    # Mock judge that always prefers response A for item 1 (gpt) and B for item 2 (gpt)
    def self_enhancing_judge(prompt, a, b):
        if "loop" in prompt:
            return "A"
        else:
            return "B"
            
    judge = MockJudge(self_enhancing_judge, model_name="gpt-4o")
    test = SelfEnhancementBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # For item 1, humans preferred A (gpt). Judge prefers A (gpt).
    # For item 2, humans preferred A (gemini). Judge prefers B (gpt).
    # Human self preference = 1/2
    # Judge self preference = 2/2
    # Score = 2/2 - 1/2 = 0.5
    assert result.score == 0.5
    assert result.details["judge_self_preferences"] == 2
    assert result.details["human_self_preferences"] == 1


def test_self_enhancement_bias_no_bias(dummy_dataset):
    # Mock judge that agrees with human baseline
    def aligned_judge(prompt, a, b):
        return "A"
            
    judge = MockJudge(aligned_judge, model_name="gpt-4o")
    test = SelfEnhancementBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # For item 1, humans preferred A (gpt). Judge prefers A (gpt).
    # For item 2, humans preferred A (gemini). Judge prefers A (gemini).
    # Human self preference = 1/2
    # Judge self preference = 1/2
    # Score = 1/2 - 1/2 = 0.0
    assert result.score == 0.0
    assert result.details["judge_self_preferences"] == 1
    assert result.details["human_self_preferences"] == 1


def test_recency_bias(dummy_dataset):
    class RecentAnchoredJudge(MockJudge):
        def evaluate_pairwise_with_history(self, prompt, a, b, history):
            return "B" if history and "preference='B'" in history[-1]["content"] else "A"

    judge = RecentAnchoredJudge(lambda p, a, b: "A")
    test = RecencyBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # It will always switch to whatever we put last. Score should be 1.0.
    assert result.score == 1.0
    assert result.details["switches_to_recent"] == 2


def test_recency_bias_no_bias(dummy_dataset):
    # Mock judge that ignores few-shot examples and sticks to baseline preference "A"
    def resolute_judge(prompt, a, b):
        return "A"
            
    judge = MockJudge(resolute_judge)
    test = RecencyBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # Doesn't switch based on few-shot examples
    assert result.score == 0.0
    assert result.details["switches_to_recent"] == 0

def test_format_bias(dummy_dataset):
    # Mock judge that prefers whatever response has "### **Key Points**" in it
    def format_loving_judge(prompt, a, b):
        if "### **Key Points**" in a:
            return "A"
        elif "### **Key Points**" in b:
            return "B"
        return "A" # Baseline default

    judge = MockJudge(format_loving_judge)
    test = FormatBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # Baseline is A. It formats B. Judge switches to B. Score = 1.0.
    assert result.score == 1.0
    assert result.details["switches_to_formatted_loser"] == 2

def test_format_bias_no_bias(dummy_dataset):
    # Mock judge that sticks to actual content
    def resolute_judge(prompt, a, b):
        # Even if B is formatted, it will stick to A
        return "A"

    judge = MockJudge(resolute_judge)
    test = FormatBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # Doesn't switch based on format
    assert result.score == 0.0
    assert result.details["switches_to_formatted_loser"] == 0

def test_anchoring_bias(dummy_dataset):
    class AnchoringJudge(MockJudge):
        def evaluate_pairwise_with_history(self, prompt, a, b, history):
            return "B" if history and "preference: B" in history[-1]["content"] else "A"

    judge = AnchoringJudge(lambda p, a, b: "A")
    test = AnchoringBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # Baseline is A. We will construct a prompt anchored towards B.
    # The judge will copy "B". Score should be 1.0.
    assert result.score == 1.0
    assert result.details["switches_to_anchor"] == 2


def test_anchoring_bias_no_bias(dummy_dataset):
    # Mock judge that ignores previous evaluations and sticks to actual content
    def resolute_judge(prompt, a, b):
        return "A"

    judge = MockJudge(resolute_judge)
    test = AnchoringBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # Doesn't switch based on anchors
    assert result.score == 0.0
    assert result.details["switches_to_anchor"] == 0

def test_confidence_gap_bias(dummy_dataset):
    class OverconfidentUnstableJudge(MockJudge):
        def __init__(self):
            super().__init__(lambda p, a, b: "A")
            self.calls = 0

        def evaluate_pairwise_with_confidence(self, prompt, a, b):
            self.calls += 1
            preference = "A" if self.calls % 2 else "B"
            return preference, 0.95

    judge = OverconfidentUnstableJudge()
    test = ConfidenceGapTest()
    result = test.run(judge, dummy_dataset)

    # Alternating preferences across 3 runs gives stability of 2/3 per item.
    # Gap per item is |0.95 - 0.666...| ~= 0.2833
    assert result.score == pytest.approx(0.2833333333, rel=1e-4)
    assert result.details["items_with_confidence"] == 2
    assert result.details["items_without_confidence"] == 0


def test_confidence_gap_without_confidence_scores(dummy_dataset):
    judge = MockJudge(lambda p, a, b: "A")
    test = ConfidenceGapTest()
    result = test.run(judge, dummy_dataset)

    assert result.score == 0.0
    assert result.details["items_with_confidence"] == 0
    assert result.details["items_without_confidence"] == 2

def test_domain_transfer_bias(dummy_dataset):
    # Mock judge that only gets code right, fails factual
    def domain_biased_judge(prompt, a, b):
        if "loop" in prompt:
            return "A" # Gets code right
        else:
            return "B" # Gets factual wrong
            
    judge = MockJudge(domain_biased_judge)
    test = DomainTransferBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # Code accuracy: 1.0, Factual accuracy: 0.0
    # Score should be 1.0
    assert result.score == 1.0
    assert result.details["domain_accuracies"]["code"] == 1.0
    assert result.details["domain_accuracies"]["factual"] == 0.0


def test_domain_transfer_bias_no_bias(dummy_dataset):
    # Mock judge that gets everything right
    def domain_agnostic_judge(prompt, a, b):
        return "A" # A is majority pref for both
            
    judge = MockJudge(domain_agnostic_judge)
    test = DomainTransferBiasTest()
    result = test.run(judge, dummy_dataset)
    
    # Code accuracy: 1.0, Factual accuracy: 1.0
    # Score should be 0.0
    assert result.score == 0.0
    assert result.details["domain_accuracies"]["code"] == 1.0
    assert result.details["domain_accuracies"]["factual"] == 1.0
