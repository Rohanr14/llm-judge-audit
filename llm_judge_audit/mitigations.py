from __future__ import annotations

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


def get_mitigation(bias_key: str) -> str | None:
    return MITIGATION_RECOMMENDATIONS.get(bias_key)
