from __future__ import annotations

from typing import Iterable

from llm_judge_audit.scoring.thresholds import SEVERITY_WEIGHTS


def compute_jri(
    *,
    has_score: float,
    bias_scores: Iterable[tuple[float, str]],
    has_weight: float = 0.6,
) -> float:
    """
    Compute Judge Reliability Index on a 0-100 scale.

    `bias_scores` is an iterable of `(score, severity)` tuples where `score` is in [0, 1].
    """
    bias_list = list(bias_scores)
    if not bias_list:
        return round(max(0.0, min(100.0, has_score * 100.0)), 2)

    total_weight = 0.0
    weighted_bias = 0.0
    for score, severity in bias_list:
        weight = SEVERITY_WEIGHTS.get(severity, 1.0)
        total_weight += weight
        weighted_bias += score * weight

    avg_weighted_bias = weighted_bias / total_weight if total_weight else 0.0
    bias_weight = 1.0 - has_weight
    jri = ((has_score * 100.0) * has_weight) + ((1.0 - avg_weighted_bias) * 100.0 * bias_weight)
    return round(max(0.0, min(100.0, jri)), 2)
