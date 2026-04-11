from __future__ import annotations

from pathlib import Path

import click

from llm_judge_audit.biases.position import PositionBiasTest
from llm_judge_audit.biases.verbosity import VerbosityBiasTest
from llm_judge_audit.biases.cross_run import CrossRunConsistencyTest
from llm_judge_audit.biases.sycophancy import SycophancyBiasTest
from llm_judge_audit.biases.self_enhancement import SelfEnhancementBiasTest
from llm_judge_audit.biases.recency import RecencyBiasTest
from llm_judge_audit.biases.format_bias import FormatBiasTest
from llm_judge_audit.biases.anchoring import AnchoringBiasTest
from llm_judge_audit.biases.confidence_gap import ConfidenceGapTest
from llm_judge_audit.biases import DomainTransferBiasTest
from llm_judge_audit.judge import JudgeAPIError, get_judge
from llm_judge_audit.logger import logger
from llm_judge_audit.report import (
    build_audit_report,
    compute_human_alignment_score,
    load_anchor_dataset,
    print_terminal_report,
    write_html_report,
    write_json_report,
)

BIAS_TEST_REGISTRY = {
    "position": PositionBiasTest,
    "verbosity": VerbosityBiasTest,
    "cross_run": CrossRunConsistencyTest,
    "sycophancy": SycophancyBiasTest,
    "self_enhancement": SelfEnhancementBiasTest,
    "recency": RecencyBiasTest,
    "format_bias": FormatBiasTest,
    "anchoring": AnchoringBiasTest,
    "confidence_gap": ConfidenceGapTest,
    "domain_transfer": DomainTransferBiasTest,
}


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--model", "model_name", required=True, help="Judge model name (e.g., gpt-4o, claude-3-5-sonnet).")
@click.option("--api-key", default=None, help="Optional API key override. If omitted, env vars are used.")
@click.option(
    "--tests",
    default="all",
    show_default=True,
    help="Comma-separated bias tests to run. Supported: position,verbosity,cross_run,sycophancy,self_enhancement,recency,format_bias,anchoring,confidence_gap or 'all'.",
)
@click.option(
    "--dataset",
    "dataset_path",
    default=str(Path(__file__).parent / "datasets" / "anchor.json"),
    show_default=True,
    help="Path to anchor dataset JSON.",
)
@click.option("--output", "json_output", default="audit_report.json", show_default=True, help="JSON output file path.")
@click.option("--html-output", default=None, help="Optional HTML output file path.")
@click.option("--pretty/--no-pretty", default=True, show_default=True, help="Print terminal summary.")
@click.option(
    "--cross-run-runs",
    default=3,
    show_default=True,
    type=click.IntRange(min=2),
    help="Number of repeated runs for cross-run consistency test.",
)
@click.option(
    "--confidence-runs",
    default=3,
    show_default=True,
    type=click.IntRange(min=2),
    help="Number of repeated runs for confidence-consistency gap test.",
)
def main(
    model_name: str,
    api_key: str | None,
    tests: str,
    dataset_path: str,
    json_output: str,
    html_output: str | None,
    pretty: bool,
    cross_run_runs: int,
    confidence_runs: int,
) -> None:
    """Run LLM judge bias audit and emit machine-readable and human-readable reports."""
    selected = _resolve_tests(tests)
    logger.info("Selected tests: %s", ", ".join(selected))

    dataset = load_anchor_dataset(dataset_path)
    annotated_items = sum(1 for item in dataset.items if len(item.human_annotations) >= 3 and item.majority_preference in {"A", "B"})
    if annotated_items == 0:
        raise click.ClickException(
            "Audit incomplete: dataset has no fully annotated items with A/B majority preference yet. "
            "Run once Prolific annotations are merged."
        )
    judge = get_judge(model_name, api_key=api_key)

    try:
        bias_results = []
        for test_key in selected:
            test = _build_test_instance(
                test_key=test_key,
                cross_run_runs=cross_run_runs,
                confidence_runs=confidence_runs,
            )
            bias_results.append(test.run(judge, dataset.items))

        has_result = compute_human_alignment_score(judge, dataset.items)
    except JudgeAPIError as exc:
        raise click.ClickException(f"Audit incomplete due to judge API failure: {exc}") from exc

    report = build_audit_report(
        model_name=model_name,
        dataset=dataset,
        selected_tests=selected,
        has_result=has_result,
        bias_results=bias_results,
    )

    json_path = write_json_report(report, json_output)
    logger.info("Wrote JSON report: %s", json_path)

    if html_output:
        html_path = write_html_report(report, html_output)
        logger.info("Wrote HTML report: %s", html_path)

    if pretty:
        print_terminal_report(report)


def _resolve_tests(raw_tests: str) -> list[str]:
    if raw_tests.strip().lower() == "all":
        return list(BIAS_TEST_REGISTRY.keys())

    selected = [token.strip().lower() for token in raw_tests.split(",") if token.strip()]
    unknown = [token for token in selected if token not in BIAS_TEST_REGISTRY]
    if unknown:
        supported = ", ".join(BIAS_TEST_REGISTRY.keys())
        raise click.BadParameter(
            f"Unknown test(s): {', '.join(unknown)}. Supported tests: {supported} or 'all'.",
            param_hint="--tests",
        )
    if not selected:
        raise click.BadParameter("No tests selected.", param_hint="--tests")
    return selected


def _build_test_instance(test_key: str, cross_run_runs: int, confidence_runs: int):
    if test_key == "cross_run":
        return CrossRunConsistencyTest(n_runs=cross_run_runs)
    if test_key == "confidence_gap":
        return ConfidenceGapTest(n_runs=confidence_runs)
    return BIAS_TEST_REGISTRY[test_key]()


if __name__ == "__main__":
    main()
