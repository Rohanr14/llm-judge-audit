import json
import time
from collections import Counter
from pathlib import Path
import pandas as pd
from datasets import load_dataset
from llm_judge_audit.config import config

MIN_FIELD_LEN = 50
DATASET_DIR = Path("llm_judge_audit/datasets")
ANCHOR_PATH = DATASET_DIR / "anchor.json"
PROLIFIC_TASKS_PATH = DATASET_DIR / "prolific_tasks.csv"


def get_items_from_source(dataset_name: str, split: str, config_name: str = None, retries: int = 3):
    print(f"Loading {dataset_name} (split: {split}, config: {config_name})...")
    for attempt in range(retries):
        try:
            ds = load_dataset(
                dataset_name,
                config_name,
                split=split,
                token=config.HF_TOKEN,
                revision="refs/convert/parquet",
                download_mode="reuse_dataset_if_exists",
            )
            return ds.to_pandas()
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Attempt {attempt + 1} failed: {e}. Retrying in 5s...")
                time.sleep(5)
                continue
            else:
                print(f"FAILED to load {dataset_name}: {e}")
                return pd.DataFrame()
    return pd.DataFrame()


def classify_domain(prompt: str) -> str:
    p = prompt.lower()
    if any(k in p for k in [
        "python", " code", "sql", "debug", "function", "algorithm",
        "implement", "program", "html", "css", "javascript", "script",
        "website", "webpage", "programming", "compiler", "syntax",
    ]):
        return "code"
    if any(k in p for k in [
        "story", "poem", "joke", "write a", "write me", "write an",
        "compose", "creative", "fiction", "essay", "song", "speech",
        "blog post", "novel", "haiku", "limerick", "screenplay",
    ]):
        return "creative"
    return "factual"


def safe_extract(row, source_name: str):
    """Maps each dataset's schema to (prompt, response_a, response_b)."""
    try:
        if source_name in ("lmsys", "mt_bench"):
            # Both datasets share the same conversation_a / conversation_b schema
            return (
                row["conversation_a"][0]["content"],
                row["conversation_a"][1]["content"],
                row["conversation_b"][1]["content"],
            )
        elif source_name == "hh_rlhf":
            chosen = row.get("chosen", "")
            rejected = row.get("rejected", "")
            if not chosen or not rejected:
                return None, None, None
            prompt_parts = chosen.split("\n\nAssistant:")
            prompt = prompt_parts[0].replace("\n\nHuman:", "").strip()
            response_a = prompt_parts[1].strip() if len(prompt_parts) > 1 else ""
            response_b = rejected.split("\n\nAssistant:")[-1].strip()
            return prompt, response_a, response_b
        return None, None, None
    except (KeyError, IndexError, TypeError):
        return None, None, None


def is_valid(p, ra, rb) -> bool:
    """Rejects items that are empty, too short, or have identical responses."""
    if not p or not ra or not rb:
        return False
    if len(p.strip()) < MIN_FIELD_LEN:
        return False
    if len(ra.strip()) < MIN_FIELD_LEN:
        return False
    if len(rb.strip()) < MIN_FIELD_LEN:
        return False
    if ra.strip() == rb.strip():
        return False
    return True


def fetch_and_transform_all_sources():
    lmsys_df = get_items_from_source("lmsys/chatbot_arena_conversations", "train")
    mt_bench_df = get_items_from_source("lmsys/mt_bench_human_judgments", "train")
    hh_rlhf_df = get_items_from_source("Anthropic/hh-rlhf", "train", None)

    if lmsys_df.empty or mt_bench_df.empty or hh_rlhf_df.empty:
        failed = [
            n for n, df in [
                ("lmsys", lmsys_df),
                ("mt_bench", mt_bench_df),
                ("hh_rlhf", hh_rlhf_df),
            ]
            if df.empty
        ]
        print(f"\nCRITICAL: Failed to load: {', '.join(failed)}. Check HF token and dataset access.")
        return

    target_counts = {"code": 33, "factual": 33, "creative": 34}
    # Each source contributes at most ~1/3 of each domain bucket in the first pass
    per_source_cap = {domain: max(count // 3, 5) for domain, count in target_counts.items()}

    sources = [
        (lmsys_df, "lmsys"),
        (mt_bench_df, "mt_bench"),
        (hh_rlhf_df, "hh_rlhf"),
    ]

    stratified_items: dict[str, list] = {"code": [], "factual": [], "creative": []}
    source_domain_counts: dict[str, dict[str, int]] = {
        source_name: {"code": 0, "factual": 0, "creative": 0}
        for _, source_name in sources
    }

    def try_add(p, ra, rb, source_name, cap=None):
        """Attempts to add an item; returns True if added."""
        if not is_valid(p, ra, rb):
            return False
        domain = classify_domain(p)
        if len(stratified_items[domain]) >= target_counts[domain]:
            return False
        if cap is not None and source_domain_counts[source_name][domain] >= cap[domain]:
            return False
        if any(x["prompt"] == p for x in stratified_items[domain]):
            return False
        item = {
            "item_id": f"{source_name}-{domain}-{len(stratified_items[domain]):03d}",
            "domain": domain,
            "prompt": p,
            "response_a": ra,
            "response_b": rb,
            "human_annotations": [],
            "majority_preference": "Tie",
        }
        stratified_items[domain].append(item)
        source_domain_counts[source_name][domain] += 1
        return True

    # Pass 1: fill with per-source cap to ensure diversity
    for df, source_name in sources:
        for _, row in df.sample(frac=1, random_state=42).iterrows():
            if all(len(stratified_items[d]) >= target_counts[d] for d in target_counts):
                break
            p, ra, rb = safe_extract(row, source_name)
            try_add(p, ra, rb, source_name, cap=per_source_cap)

    # Pass 2: fill remaining slots from any source without cap
    for df, source_name in sources:
        for _, row in df.sample(frac=1, random_state=99).iterrows():
            if all(len(stratified_items[d]) >= target_counts[d] for d in target_counts):
                break
            p, ra, rb = safe_extract(row, source_name)
            try_add(p, ra, rb, source_name, cap=None)

    all_items = (
        stratified_items["code"]
        + stratified_items["factual"]
        + stratified_items["creative"]
    )

    if len(all_items) < 100:
        print(f"\nWARNING: Only sourced {len(all_items)}/100 items. Check source data quality.")

    source_breakdown = dict(Counter(i["item_id"].split("-")[0] for i in all_items))
    domain_breakdown = dict(Counter(i["domain"] for i in all_items))
    print(f"\nSource breakdown: {source_breakdown}")
    print(f"Domain breakdown: {domain_breakdown}")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    with ANCHOR_PATH.open("w", encoding="utf-8") as f:
        json.dump({"version": "1.0", "items": all_items}, f, indent=2)

    pd.DataFrame(all_items)[
        ["item_id", "domain", "prompt", "response_a", "response_b"]
    ].to_csv(PROLIFIC_TASKS_PATH, index=False)

    print(f"VALIDATED: {len(all_items)} items sourced.")
    print(f"Output: {ANCHOR_PATH}, {PROLIFIC_TASKS_PATH}")


if __name__ == "__main__":
    fetch_and_transform_all_sources()
