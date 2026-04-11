from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from llm_judge_audit.biases.base import BiasTestResult
from llm_judge_audit.datasets.schema import AnchorDataset
from llm_judge_audit.mitigations import get_mitigation
from llm_judge_audit.scoring.has import HASResult, compute_human_alignment_score
from llm_judge_audit.scoring.jri import compute_jri as compute_jri_score
from llm_judge_audit.scoring.thresholds import get_threshold_rule, normalize_bias_key

__all__ = [
    "AuditReport",
    "BiasSummary",
    "build_audit_report",
    "compute_human_alignment_score",
    "compute_jri",
    "load_anchor_dataset",
    "print_terminal_report",
    "summarize_bias_results",
    "write_html_report",
    "write_json_report",
]


class BiasSummary(BaseModel):
    bias_name: str
    bias_key: str
    score: float
    severity: str = "unknown"
    threshold: float | None = None
    flagged: bool = False
    mitigation: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AuditReport(BaseModel):
    model_name: str
    generated_at_utc: str
    dataset_version: str
    total_items: int
    selected_tests: list[str]
    human_alignment_score: float
    has_by_domain: dict[str, float]
    jri: float
    flagged_biases: list[str]
    bias_results: list[BiasSummary]


def load_anchor_dataset(path: str | Path) -> AnchorDataset:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return AnchorDataset.model_validate(payload)


def summarize_bias_results(results: Iterable[BiasTestResult]) -> list[BiasSummary]:
    summaries: list[BiasSummary] = []
    for result in results:
        bias_key = normalize_bias_key(result.bias_name)
        rule = get_threshold_rule(result.bias_name)
        threshold = rule["threshold"] if rule else None
        severity = rule["severity"] if rule else "unknown"
        flagged = threshold is not None and result.score > threshold
        summaries.append(
            BiasSummary(
                bias_name=result.bias_name,
                bias_key=bias_key,
                score=float(result.score),
                severity=severity,
                threshold=threshold,
                flagged=flagged,
                mitigation=get_mitigation(bias_key),
                details=result.details,
            )
        )
    return summaries


def compute_jri(
    has_score: float,
    bias_summaries: Iterable[BiasSummary],
    *,
    has_weight: float = 0.6,
) -> float:
    return compute_jri_score(
        has_score=has_score,
        bias_scores=((bias.score, bias.severity) for bias in bias_summaries),
        has_weight=has_weight,
    )


def build_audit_report(
    *,
    model_name: str,
    dataset: AnchorDataset,
    selected_tests: list[str],
    has_result: HASResult,
    bias_results: list[BiasTestResult],
    has_weight: float = 0.6,
) -> AuditReport:
    summaries = summarize_bias_results(bias_results)
    jri = compute_jri(has_result.overall, summaries, has_weight=has_weight)
    flagged = [bias.bias_key for bias in summaries if bias.flagged]

    return AuditReport(
        model_name=model_name,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        dataset_version=dataset.version,
        total_items=len(dataset.items),
        selected_tests=selected_tests,
        human_alignment_score=round(has_result.overall, 4),
        has_by_domain={k: round(v, 4) for k, v in sorted(has_result.by_domain.items())},
        jri=jri,
        flagged_biases=flagged,
        bias_results=summaries,
    )


def write_json_report(report: AuditReport, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return out


def print_terminal_report(report: AuditReport) -> None:
    print("\n=== LLM Judge Audit Report ===")
    print(f"Model: {report.model_name}")
    print(f"Generated: {report.generated_at_utc}")
    print(f"Dataset version: {report.dataset_version} ({report.total_items} items)")
    print(f"HAS: {report.human_alignment_score:.2%}")
    print(f"JRI: {report.jri:.2f}/100")

    if report.has_by_domain:
        print("\nHAS by domain:")
        for domain, score in report.has_by_domain.items():
            print(f"  - {domain}: {score:.2%}")

    print("\nBias results:")
    for bias in report.bias_results:
        marker = "⚠" if bias.flagged else "✓"
        threshold_txt = f" (threshold {bias.threshold:.2f})" if bias.threshold is not None else ""
        print(f"  {marker} {bias.bias_name}: {bias.score:.3f}{threshold_txt}")

    if report.flagged_biases:
        print("\nMitigations:")
        for bias in report.bias_results:
            if bias.flagged and bias.mitigation:
                print(f"  - {bias.bias_name}: {bias.mitigation}")


def write_html_report(report: AuditReport, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = "\n".join(
        (
            f"<tr><td>{b.bias_name}</td><td>{b.score:.3f}</td>"
            f"<td>{b.severity}</td><td>{'yes' if b.flagged else 'no'}</td>"
            f"<td>{b.mitigation or ''}</td></tr>"
        )
        for b in report.bias_results
    )

    html = f"""
            <!doctype html>
            <html lang=\"en\">
            <head>
              <meta charset=\"utf-8\" />
              <title>LLM Judge Audit Report</title>
              <style>
                body {{ font-family: Arial, sans-serif; margin: 2rem; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #f6f6f6; }}
              </style>
            </head>
            <body>
              <h1>LLM Judge Audit Report</h1>
              <p><strong>Model:</strong> {report.model_name}</p>
              <p><strong>Generated (UTC):</strong> {report.generated_at_utc}</p>
              <p><strong>Dataset:</strong> v{report.dataset_version} ({report.total_items} items)</p>
              <p><strong>HAS:</strong> {report.human_alignment_score:.2%}</p>
              <p><strong>JRI:</strong> {report.jri:.2f}/100</p>
              <h2>Bias Results</h2>
              <table>
                <thead>
                  <tr><th>Bias</th><th>Score</th><th>Severity</th><th>Flagged</th><th>Mitigation</th></tr>
                </thead>
                <tbody>
                  {rows}
                </tbody>
              </table>
            </body>
            </html>
            """
    out.write_text(html, encoding="utf-8")
    return out
