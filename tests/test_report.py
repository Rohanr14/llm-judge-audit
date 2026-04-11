from unittest.mock import patch

from click.testing import CliRunner

from llm_judge_audit.biases.confidence_gap import ConfidenceGapTest
from llm_judge_audit.biases.cross_run import CrossRunConsistencyTest
from llm_judge_audit.biases.position import PositionBiasTest
from llm_judge_audit.cli import _build_test_instance, _resolve_tests, main
from llm_judge_audit.datasets.schema import AnchorDatasetItem, HumanAnnotation
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.report import (
    compute_human_alignment_score,
    summarize_bias_results,
)


class AlwaysAJudge(BaseJudge):
    def __init__(self):
        super().__init__("always-a")

    def _evaluate_pairwise_impl(self, prompt, response_a, response_b, temperature=0.0):
        return "A"


def _dataset_items():
    return [
        AnchorDatasetItem(
            item_id="1",
            domain="code",
            prompt="P1",
            response_a="Good",
            response_b="Bad",
            human_annotations=[HumanAnnotation(rater_id=f"r{i}", preference="A") for i in range(3)],
            majority_preference="A",
        ),
        AnchorDatasetItem(
            item_id="2",
            domain="factual",
            prompt="P2",
            response_a="Wrong",
            response_b="Right",
            human_annotations=[HumanAnnotation(rater_id=f"x{i}", preference="B") for i in range(3)],
            majority_preference="B",
        ),
    ]


def test_resolve_tests_all_and_subset():
    assert _resolve_tests("all") == ["position", "verbosity", "cross_run", "sycophancy", "self_enhancement", "recency", "format_bias", "anchoring", "confidence_gap", "domain_transfer"]
    assert _resolve_tests("position") == ["position"]


def test_build_test_instance_with_custom_runs():
    cross_run_test = _build_test_instance("cross_run", cross_run_runs=5, confidence_runs=3)
    confidence_test = _build_test_instance("confidence_gap", cross_run_runs=3, confidence_runs=6)

    assert isinstance(cross_run_test, CrossRunConsistencyTest)
    assert cross_run_test.n_runs == 5
    assert isinstance(confidence_test, ConfidenceGapTest)
    assert confidence_test.n_runs == 6


def test_report_helpers():
    judge = AlwaysAJudge()
    items = _dataset_items()
    has = compute_human_alignment_score(judge, items)
    assert has.overall == 0.5

    bias_result = PositionBiasTest().run(judge, items)
    summaries = summarize_bias_results([bias_result])
    assert summaries[0].bias_key == "position"


def test_bias_key_aliases_apply_thresholds():
    judge = AlwaysAJudge()
    items = _dataset_items()

    cross_run_summary = summarize_bias_results([CrossRunConsistencyTest(n_runs=3).run(judge, items)])[0]
    confidence_summary = summarize_bias_results([ConfidenceGapTest(n_runs=3).run(judge, items)])[0]

    assert cross_run_summary.bias_key == "cross_run"
    assert cross_run_summary.threshold == 0.35
    assert confidence_summary.bias_key == "confidence_gap"
    assert confidence_summary.threshold == 0.35


def test_cli_runs(tmp_path):
    runner = CliRunner()
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        '{"version":"1.0","items":[{"item_id":"1","domain":"code","difficulty":"easy","prompt":"P","response_a":"A","response_b":"B","human_annotations":[{"rater_id":"r1","preference":"A"},{"rater_id":"r2","preference":"A"},{"rater_id":"r3","preference":"A"}],"majority_preference":"A"}]}'
    )
    out = tmp_path / "report.json"

    with patch("llm_judge_audit.cli.get_judge", return_value=AlwaysAJudge()):
        result = runner.invoke(
            main,
            [
                "--model",
                "gpt-4o",
                "--api-key",
                "test-key",
                "--tests",
                "position",
                "--dataset",
                str(dataset),
                "--output",
                str(out),
                "--no-pretty",
            ],
        )
    assert result.exit_code == 0
    assert out.exists()
