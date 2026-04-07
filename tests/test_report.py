from click.testing import CliRunner

from llm_judge_audit.biases.position import PositionBiasTest
from llm_judge_audit.cli import _resolve_tests, main
from llm_judge_audit.datasets.schema import AnchorDatasetItem, HumanAnnotation
from llm_judge_audit.judge import BaseJudge
from llm_judge_audit.report import (
    compute_human_alignment_score,
    summarize_bias_results,
)


class AlwaysAJudge(BaseJudge):
    def __init__(self):
        super().__init__("always-a")

    def evaluate_pairwise(self, prompt, response_a, response_b):
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
    assert _resolve_tests("all") == ["position", "verbosity"]
    assert _resolve_tests("position") == ["position"]


def test_report_helpers():
    judge = AlwaysAJudge()
    items = _dataset_items()
    has = compute_human_alignment_score(judge, items)
    assert has.overall == 0.5

    bias_result = PositionBiasTest().run(judge, items)
    summaries = summarize_bias_results([bias_result])
    assert summaries[0].bias_key == "position"


def test_cli_runs(tmp_path):
    runner = CliRunner()
    dataset = tmp_path / "dataset.json"
    dataset.write_text(
        '{"version":"1.0","items":[{"item_id":"1","domain":"code","difficulty":"easy","prompt":"P","response_a":"A","response_b":"B","human_annotations":[{"rater_id":"r1","preference":"A"},{"rater_id":"r2","preference":"A"},{"rater_id":"r3","preference":"A"}],"majority_preference":"A"}]}'
    )
    out = tmp_path / "report.json"

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