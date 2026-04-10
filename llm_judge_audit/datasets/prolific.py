from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from llm_judge_audit.datasets.schema import AnchorDataset, HumanAnnotation


def _normalize_preference(value: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().upper()
    if normalized in {"A", "B", "TIE"}:
        return "Tie" if normalized == "TIE" else normalized
    return None


def _majority_preference(annotations: list[dict[str, Any]]) -> str:
    if not annotations:
        return "Tie"
    counts = Counter(a["preference"] for a in annotations)
    top_count = max(counts.values())
    winners = [pref for pref, count in counts.items() if count == top_count]
    return winners[0] if len(winners) == 1 else "Tie"


def apply_prolific_annotations(
    *,
    anchor_path: str | Path,
    annotations_csv_path: str | Path,
    output_path: str | Path,
    item_id_col: str = "item_id",
    rater_id_col: str = "participant_id",
    preference_col: str = "preference",
    confidence_col: str = "confidence",
    rationale_col: str = "rationale",
    require_min_annotations: bool = True,
) -> Path:
    """
    Merge Prolific CSV annotations into anchor dataset and recompute majority labels.
    """
    payload = json.loads(Path(anchor_path).read_text(encoding="utf-8"))
    items = payload.get("items", [])
    items_by_id = {item["item_id"]: item for item in items}

    per_item_per_rater: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    with Path(annotations_csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = (row.get(item_id_col) or "").strip()
            rater_id = (row.get(rater_id_col) or "").strip()
            preference = _normalize_preference(row.get(preference_col) or "")
            if not item_id or not rater_id or not preference:
                continue
            if item_id not in items_by_id:
                continue

            ann: dict[str, Any] = {
                "rater_id": rater_id,
                "preference": preference,
            }

            confidence_raw = (row.get(confidence_col) or "").strip()
            if confidence_raw:
                try:
                    confidence = int(confidence_raw)
                    if 1 <= confidence <= 5:
                        ann["confidence"] = confidence
                except ValueError:
                    pass

            rationale = (row.get(rationale_col) or "").strip()
            if rationale:
                ann["rationale"] = rationale

            # Keep one response per (item, rater); latest row wins.
            per_item_per_rater[item_id][rater_id] = ann

    for item_id, item in items_by_id.items():
        annotations = list(per_item_per_rater.get(item_id, {}).values())
        item["human_annotations"] = annotations
        item["majority_preference"] = _majority_preference(annotations)
        item["is_gold_standard"] = bool(
            annotations and len({a["preference"] for a in annotations}) == 1
        )

    # Validate before writing when a complete annotation set is expected.
    if require_min_annotations:
        AnchorDataset.model_validate(payload)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def build_annotation_progress(
    *,
    anchor_path: str | Path,
    annotations_csv_path: str | Path,
    item_id_col: str = "item_id",
    rater_id_col: str = "participant_id",
    preference_col: str = "preference",
) -> dict[str, Any]:
    payload = json.loads(Path(anchor_path).read_text(encoding="utf-8"))
    items = payload.get("items", [])

    item_to_domain = {item["item_id"]: item.get("domain", "unknown") for item in items}
    per_item_raters: dict[str, set[str]] = defaultdict(set)

    with Path(annotations_csv_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = (row.get(item_id_col) or "").strip()
            rater_id = (row.get(rater_id_col) or "").strip()
            preference = _normalize_preference(row.get(preference_col) or "")
            if not item_id or not rater_id or not preference:
                continue
            if item_id not in item_to_domain:
                continue
            per_item_raters[item_id].add(rater_id)

    domain_totals: dict[str, int] = defaultdict(int)
    domain_started: dict[str, int] = defaultdict(int)
    domain_complete: dict[str, int] = defaultdict(int)

    for item in items:
        item_id = item["item_id"]
        domain = item.get("domain", "unknown")
        num_raters = len(per_item_raters.get(item_id, set()))
        domain_totals[domain] += 1
        if num_raters > 0:
            domain_started[domain] += 1
        if num_raters >= 3:
            domain_complete[domain] += 1

    total_items = len(items)
    started_items = sum(1 for item in items if len(per_item_raters.get(item["item_id"], set())) > 0)
    complete_items = sum(1 for item in items if len(per_item_raters.get(item["item_id"], set())) >= 3)

    return {
        "total_items": total_items,
        "items_with_at_least_1_annotation": started_items,
        "items_with_at_least_3_annotations": complete_items,
        "completion_rate_3plus": (complete_items / total_items) if total_items else 0.0,
        "domains": {
            domain: {
                "total_items": domain_totals[domain],
                "items_with_at_least_1_annotation": domain_started[domain],
                "items_with_at_least_3_annotations": domain_complete[domain],
            }
            for domain in sorted(domain_totals.keys())
        },
    }


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge Prolific CSV annotations into anchor.json and recompute majority labels."
    )
    parser.add_argument("--anchor", required=True, help="Path to source anchor.json")
    parser.add_argument("--annotations", required=True, help="Path to Prolific CSV export")
    parser.add_argument("--output", help="Path to write updated anchor.json")
    parser.add_argument("--item-id-col", default="item_id")
    parser.add_argument("--rater-id-col", default="participant_id")
    parser.add_argument("--preference-col", default="preference")
    parser.add_argument("--confidence-col", default="confidence")
    parser.add_argument("--rationale-col", default="rationale")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow writing partially annotated anchor data (<3 ratings per item).",
    )
    parser.add_argument(
        "--progress-only",
        action="store_true",
        help="Only print annotation coverage progress; do not write merged anchor output.",
    )
    args = parser.parse_args()

    if args.progress_only:
        progress = build_annotation_progress(
            anchor_path=args.anchor,
            annotations_csv_path=args.annotations,
            item_id_col=args.item_id_col,
            rater_id_col=args.rater_id_col,
            preference_col=args.preference_col,
        )
        print(json.dumps(progress, indent=2))
        return

    if not args.output:
        raise SystemExit("--output is required unless --progress-only is set.")

    out = apply_prolific_annotations(
        anchor_path=args.anchor,
        annotations_csv_path=args.annotations,
        output_path=args.output,
        item_id_col=args.item_id_col,
        rater_id_col=args.rater_id_col,
        preference_col=args.preference_col,
        confidence_col=args.confidence_col,
        rationale_col=args.rationale_col,
        require_min_annotations=not args.allow_partial,
    )
    print(f"Wrote merged dataset: {out}")


if __name__ == "__main__":
    _main()
