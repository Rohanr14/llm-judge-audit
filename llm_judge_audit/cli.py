from __future__ import annotations

from pathlib import Path

import click

from llm_judge_audit.biases import DomainTransferBiasTest
from llm_judge_audit.biases.anchoring import AnchoringBiasTest
from llm_judge_audit.biases.confidence_gap import ConfidenceGapTest
from llm_judge_audit.biases.cross_run import CrossRunConsistencyTest
from llm_judge_audit.biases.format_bias import FormatBiasTest
from llm_judge_audit.biases.position import PositionBiasTest
from llm_judge_audit.biases.recency import RecencyBiasTest
from llm_judge_audit.biases.self_enhancement import SelfEnhancementBiasTest
from llm_judge_audit.biases.sycophancy import SycophancyBiasTest
from llm_judge_audit.biases.verbosity import VerbosityBiasTest
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
from llm_judge_audit.runtime import SETTINGS

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
    help="Comma-separated bias tests to run. Supported: position,verbosity,cross_run,sycophancy,"
    "self_enhancement,recency,format_bias,anchoring,confidence_gap,domain_transfer or 'all'.",
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
@click.option(
    "--max-concurrency",
    default=4,
    show_default=True,
    type=click.IntRange(min=1),
    help="Process-wide cap on concurrent judge API calls. Lower this on rate-limited providers.",
)
@click.option(
    "--cache-path",
    "cache_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Enable disk-backed judge checkpoint cache at this JSONL path. "
    "Lets a crashed audit resume without re-spending API credits.",
)
@click.option(
    "--has-weight",
    default=0.6,
    show_default=True,
    type=click.FloatRange(min=0.0, max=1.0),
    help="Weight of Human Alignment Score in the JRI composite (0-1). "
    "The remaining 1 - has_weight is assigned to bias scores.",
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
    max_concurrency: int,
    cache_path: Path | None,
    has_weight: float,
) -> None:
    """Run LLM judge bias audit and emit machine-readable and human-readable reports."""
    # Apply runtime settings up-front so every downstream caller sees them.
    SETTINGS.configure(max_concurrency=max_concurrency, cache_path=cache_path)
    logger.info(
        "Runtime: max_concurrency=%d, cache_path=%s",
        SETTINGS.max_concurrency,
        SETTINGS.cache_path or "disabled",
    )

    selected = _resolve_tests(tests)
    logger.info("Selected tests: %s", ", ".join(selected))

    dataset = load_anchor_dataset(dataset_path)
    # HAS-scoreable: has at least one human annotation with an A/B majority.
    # Fewer than 3 raters per item isn't fatal any more -- we warn instead
    # and let HAS do its own filtering downstream.
    scoreable_items = sum(
        1
        for item in dataset.items
        if item.human_annotations and item.majority_preference in {"A", "B"}
    )
    fully_annotated = sum(
        1
        for item in dataset.items
        if len(item.human_annotations) >= 3 and item.majority_preference in {"A", "B"}
    )
    if scoreable_items == 0:
        raise click.ClickException(
            "Audit incomplete: dataset has no items with any human annotation "
            "and an A/B majority. Run once Prolific annotations are merged."
        )
    if fully_annotated < scoreable_items:
        logger.warning(
            "HAS will use %d items with any annotation (%d have the target >=3 raters).",
            scoreable_items,
            fully_annotated,
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
        has_weight=has_weight,
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
