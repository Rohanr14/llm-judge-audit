"""In-place, Prolific-safe anchor dataset migration.

Use this when the schema or the set of metadata fields evolves *while a
Prolific study is in flight*. It only touches fields that are hidden from
raters or cheap to compute from the existing prompt/response text:

* ``difficulty``     -- derived from ``classify_difficulty(prompt)``.
* ``chosen_slot``    -- left as-is if already present, otherwise ``None``.
* ``source_winner``  -- left as-is if already present, otherwise ``None``.
* ``model_a_family`` -- left as-is if already present, otherwise ``None``.
* ``model_b_family`` -- left as-is if already present, otherwise ``None``.
* ``is_gold_standard`` -- left as-is if already present, otherwise ``False``.
* ``human_annotations`` -- defaulted to ``[]`` if missing.
* ``majority_preference`` -- defaulted to ``"Tie"`` if missing.

What it deliberately does NOT do:

* Rewrite ``prompt`` / ``response_a`` / ``response_b`` text.
* Swap slot A and slot B (would invalidate Prolific annotations in flight).
* Drop items that fail leak-marker checks (same reason).
* Recompute ``item_id``.

A full clean regeneration (including slot randomisation and leak filtering)
is what ``download.py`` is for; that's appropriate between studies, not
during one. The migration here is the mid-study emergency tool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from llm_judge_audit.datasets.download import classify_difficulty
from llm_judge_audit.datasets.schema import AnchorDataset

DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_PATH = DATASET_DIR / "anchor.json"

# Fields this migration is allowed to populate if missing. Anything not in
# this list is an explicit out-of-scope change.
_MIGRATION_FIELDS = {
    "difficulty",
    "chosen_slot",
    "source_winner",
    "model_a_family",
    "model_b_family",
    "is_gold_standard",
    "human_annotations",
    "majority_preference",
}


def _migrate_item(item: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return (updated_item, list_of_fields_that_were_added)."""
    updated: dict[str, Any] = dict(item)
    added: list[str] = []

    if "difficulty" not in updated:
        updated["difficulty"] = classify_difficulty(updated.get("prompt", "") or "")
        added.append("difficulty")

    for field, default in (
        ("chosen_slot", None),
        ("source_winner", None),
        ("model_a_family", None),
        ("model_b_family", None),
        ("is_gold_standard", False),
    ):
        if field not in updated:
            updated[field] = default
            added.append(field)

    if "human_annotations" not in updated:
        updated["human_annotations"] = []
        added.append("human_annotations")

    if "majority_preference" not in updated:
        updated["majority_preference"] = "Tie"
        added.append("majority_preference")

    return updated, added


def migrate_anchor(
    path: str | Path = DEFAULT_PATH,
    output: str | Path | None = None,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Load ``path``, migrate its items, validate the result, and write it back.

    Returns a small summary dict so callers (CLI, tests) can report what
    actually changed without having to diff JSON.
    """
    src = Path(path)
    dst = Path(output) if output is not None else src

    with src.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    items = payload.get("items", [])
    field_counts: dict[str, int] = {name: 0 for name in _MIGRATION_FIELDS}
    migrated: list[dict[str, Any]] = []
    for item in items:
        new_item, added = _migrate_item(item)
        for field_name in added:
            field_counts[field_name] = field_counts.get(field_name, 0) + 1
        migrated.append(new_item)

    payload["items"] = migrated
    payload.setdefault("version", "1.0")

    if validate:
        # Will raise pydantic ValidationError if anything looks wrong; the
        # whole point of the relaxed schema is that partially-annotated
        # datasets should pass this check cleanly.
        AnchorDataset.model_validate(payload)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "input_path": str(src),
        "output_path": str(dst),
        "items": len(migrated),
        "fields_added": {k: v for k, v in field_counts.items() if v},
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        default=str(DEFAULT_PATH),
        help="Anchor JSON to migrate (default: llm_judge_audit/datasets/anchor.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the migrated dataset here. Defaults to overwriting --path in place.",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip pydantic validation after migration (for debugging only).",
    )
    args = parser.parse_args()

    summary = migrate_anchor(
        path=args.path,
        output=args.output,
        validate=not args.no_validate,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
