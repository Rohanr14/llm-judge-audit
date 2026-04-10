import pytest

from llm_judge_audit.scoring.jri import compute_jri
from llm_judge_audit.scoring.thresholds import get_threshold_rule, normalize_bias_key


def test_normalize_bias_key_aliases():
    assert normalize_bias_key("Cross-run consistency") == "cross_run"
    assert normalize_bias_key("Confidence-consistency gap") == "confidence_gap"
    assert normalize_bias_key("Format bias") == "format"


def test_get_threshold_rule():
    rule = get_threshold_rule("Cross-run consistency")
    assert rule is not None
    assert rule["severity"] == "minor"
    assert rule["threshold"] == 0.35


def test_compute_jri_no_biases():
    assert compute_jri(has_score=0.8, bias_scores=[]) == 80.0


def test_compute_jri_weighted_biases():
    score = compute_jri(
        has_score=0.8,
        bias_scores=[(0.2, "critical"), (0.1, "minor")],
    )
    assert score == pytest.approx(81.0)
