from __future__ import annotations

from typing import Any

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

BIAS_KEY_ALIASES = {
    "cross_run_consistency": "cross_run",
    "confidence_consistency_gap": "confidence_gap",
    "format_bias": "format",
}


def normalize_bias_key(name: str) -> str:
    normalized = name.lower().replace(" bias", "")
    normalized = normalized.replace("-", "_").replace(" ", "_")
    return BIAS_KEY_ALIASES.get(normalized, normalized)


def get_threshold_rule(bias_name: str) -> dict[str, Any] | None:
    return SEVERITY_THRESHOLDS.get(normalize_bias_key(bias_name))
