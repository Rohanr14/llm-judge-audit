from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, Field

from llm_judge_audit.biases.base import BiasTestResult
from llm_judge_audit.datasets.schema import AnchorDataset, AnchorDatasetItem
from llm_judge_audit.judge import BaseJudge

SEVERITY_THRESHOLDS: dict[str, dict[str, Any]] = {
    "position": {"severity": "critical", "threshold": 0.15},
    "sycophancy": {"severity": "critical", "threshold": 0.15},
    "self_enhancement": {"severity": "critical", "threshold": 0.15},
    "verbosity": {"severity": "moderate", "threshold": 0.25},
    "anchoring": {"severity": "moderate", "threshold": 0.25},
    "domain_transfer": {"severity": "moderate", "threshold": 0.25},
    "recency": {"severity": "minor", "threshold": 0.35},
    "format": {"severity": "minor", "threshold": 0.35},
    "confidence_gap": {"severity": "minor", "threshold": 0.35},
    "cross_run": {"severity": "minor", "threshold": 0.35},
}

SEVERITY_WEIGHTS = {"critical": 3.0, "moderate": 2.0, "minor": 1.0}

MITIGATION_RECOMMENDATIONS = {
    "position": "Run pairwise comparisons in both A/B and B/A order, then average outcomes.",
    "verbosity": "Use explicit length-independence instructions and penalize filler-only differences.",
    "self_enhancement": "Use a judge from a different model family than the model under evaluation.",
    "sycophancy": "Remove suggested ratings and sentiment anchors from judge-visible context.",
    "anchoring": "Randomize item order and reset judge context between evaluations.",
    "cross_run": "Evaluate each item multiple times and aggregate via majority vote.",
    "recency": "Randomize few-shot order to reduce recency anchoring.",
    "format": "Normalize markdown/plaintext formatting before sending responses to the judge.",
    "confidence_gap": "Ignore confidence language and rely on choice stability metrics.",
    "domain_transfer": "Report domain-specific scores and prefer domain-specialized judges when needed.",
}


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


@dataclass(frozen=True)
class HASResult:
    overall: float
    by_domain: dict[str, float]


def _normalize_bias_key(name: str) -> str:
    normalized = name.lower().replace(" bias", "")
    return normalized.replace("-", "_").replace(" ", "_")


def load_anchor_dataset(path: str | Path) -> AnchorDataset:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return AnchorDataset.model_validate(payload)


def compute_human_alignment_score(judge: BaseJudge, items: Iterable[AnchorDatasetItem]) -> HASResult:
    totals: dict[str, int] = defaultdict(int)
    matches: dict[str, int] = defaultdict(int)

    all_items = list(items)
    if not all_items:
        return HASResult(overall=0.0, by_domain={})

    for item in all_items:
        pref = judge.evaluate_pairwise(item.prompt, item.response_a, item.response_b)
        totals["overall"] += 1
        totals[item.domain] += 1
        if pref == item.majority_preference:
            matches["overall"] += 1
            matches[item.domain] += 1

    by_domain = {
        domain: (matches[domain] / count if count else 0.0)
        for domain, count in totals.items()
        if domain != "overall"
    }
    overall = matches["overall"] / totals["overall"] if totals["overall"] else 0.0
    return HASResult(overall=overall, by_domain=by_domain)


def summarize_bias_results(results: Iterable[BiasTestResult]) -> list[BiasSummary]:
    summaries: list[BiasSummary] = []
    for result in results:
        bias_key = _normalize_bias_key(result.bias_name)
        rule = SEVERITY_THRESHOLDS.get(bias_key)
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
                mitigation=MITIGATION_RECOMMENDATIONS.get(bias_key),
                details=result.details,
            )
        )
    return summaries


def compute_jri(has_score: float, bias_summaries: Iterable[BiasSummary]) -> float:
    biases = list(bias_summaries)
    if not biases:
        return round(max(0.0, min(100.0, has_score * 100.0)), 2)

    total_weight = 0.0
    weighted_bias = 0.0
    for bias in biases:
        weight = SEVERITY_WEIGHTS.get(bias.severity, 1.0)
        total_weight += weight
        weighted_bias += bias.score * weight

    avg_weighted_bias = weighted_bias / total_weight if total_weight else 0.0

    # 60% weight on human alignment, 40% on inverted weighted bias burden.
    jri = ((has_score * 100.0) * 0.6) + ((1.0 - avg_weighted_bias) * 100.0 * 0.4)
    return round(max(0.0, min(100.0, jri)), 2)


def build_audit_report(
    *,
    model_name: str,
    dataset: AnchorDataset,
    selected_tests: list[str],
    has_result: HASResult,
    bias_results: list[BiasTestResult],
) -> AuditReport:
    summaries = summarize_bias_results(bias_results)
    jri = compute_jri(has_result.overall, summaries)
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
        f"<tr><td>{b.bias_name}</td><td>{b.score:.3f}</td><td>{b.severity}</td><td>{'yes' if b.flagged else 'no'}</td><td>{b.mitigation or ''}</td></tr>"
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