"""Fetch source pairwise-comparison datasets and build a fresh anchor.json.

This builder is the canonical way to produce ``anchor.json``. It:

* Extracts single-turn prompts/responses from lmsys chatbot arena, mt_bench, and
  hh_rlhf (multi-turn conversations are skipped rather than spliced so that
  conversation-boundary markers can never leak into response text).
* Carries the ``model_a`` / ``model_b`` metadata through as ``model_a_family``
  and ``model_b_family``.
* Carries the original lmsys/mt_bench ``winner`` column as ``source_winner``
  so the downstream scoring code has a second ground-truth signal.
* Randomises the A/B position of every hh_rlhf item so the "chosen" response
  is not always slot A (position bias would otherwise be baked into the set).

Call this before starting a Prolific run. Re-running it mid-study will
invalidate any in-flight annotations because item ordering / slot assignment
may change.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from llm_judge_audit.config import config

MIN_FIELD_LEN = 50
DATASET_DIR = Path(__file__).resolve().parent
ANCHOR_PATH = DATASET_DIR / "anchor.json"
PROLIFIC_TASKS_PATH = DATASET_DIR / "prolific_tasks.csv"

# Marker strings that indicate a multi-turn conversation got spliced into a
# response; items matching any of these are discarded wholesale.
LEAK_MARKERS = ("\n\nHuman:", "\n\nAssistant:", "Human: ", "Assistant: ")

# Fixed seed for reproducible per-item position randomisation.
RANDOMISATION_SEED = 20250411


def get_items_from_source(dataset_name: str, split: str, config_name: str | None = None, retries: int = 3):
    print(f"Loading {dataset_name} (split: {split}, config: {config_name})...")
    for attempt in range(retries):
        try:
            try:
                ds = load_dataset(
                    dataset_name,
                    config_name,
                    split=split,
                    token=config.HF_TOKEN,
                    revision="refs/convert/parquet",
                    download_mode="reuse_dataset_if_exists",
                )
            except Exception:
                ds = load_dataset(
                    dataset_name,
                    config_name,
                    split=split,
                    token=config.HF_TOKEN,
                    download_mode="reuse_dataset_if_exists",
                )
            return ds.to_pandas()
        except Exception as e:
            if attempt < retries - 1:
                print(f"  Attempt {attempt + 1} failed: {e}. Retrying in 5s...")
                time.sleep(5)
                continue
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


def classify_difficulty(prompt: str) -> str:
    tokens = len(prompt.split())
    if tokens < 18:
        return "easy"
    if tokens <= 45:
        return "medium"
    return "hard"


def infer_model_family(model_name: str | None) -> str | None:
    if not model_name:
        return None
    name = model_name.lower()
    if "gpt" in name or "o1" in name or "o3" in name:
        return "gpt"
    if "claude" in name:
        return "claude"
    if "gemini" in name or "bard" in name or "palm" in name:
        return "gemini"
    if "llama" in name:
        return "llama"
    if "mistral" in name or "mixtral" in name:
        return "mistral"
    if "qwen" in name:
        return "qwen"
    if "command" in name or "cohere" in name:
        return "cohere"
    if "vicuna" in name or "koala" in name or "alpaca" in name:
        return "llama"  # vicuna/koala/alpaca are llama derivatives
    return None


def _contains_leak(*texts: str) -> bool:
    return any(marker in (text or "") for text in texts for marker in LEAK_MARKERS)


def _hh_rlhf_single_turn(chosen: str, rejected: str) -> tuple[str | None, str | None, str | None]:
    """Extract a single-turn prompt + chosen/rejected response.

    Rejects any conversation with more than one assistant turn on either side
    so that follow-up human turns cannot leak into the response.
    """
    if not chosen or not rejected:
        return None, None, None

    def split_one_turn(text: str) -> tuple[str, str] | None:
        # hh_rlhf conversations look like "\n\nHuman: X\n\nAssistant: Y"
        # for single turn, and include more "Human:" / "Assistant:" turns
        # when multi-turn. Reject anything that isn't exactly one turn.
        human_count = text.count("Human:")
        assistant_count = text.count("Assistant:")
        if human_count != 1 or assistant_count != 1:
            return None
        # The first "Human:" is the prompt, the first "Assistant:" is the
        # response. Use partition so we don't accidentally split mid-response.
        _, _, after_human = text.partition("Human:")
        prompt_part, _, after_assistant = after_human.partition("Assistant:")
        return prompt_part.strip(), after_assistant.strip()

    chosen_split = split_one_turn(chosen)
    rejected_split = split_one_turn(rejected)
    if not chosen_split or not rejected_split:
        return None, None, None

    chosen_prompt, chosen_response = chosen_split
    rejected_prompt, rejected_response = rejected_split
    # Sanity check: both "chosen" and "rejected" should share the same prompt.
    if chosen_prompt != rejected_prompt:
        return None, None, None
    return chosen_prompt, chosen_response, rejected_response


def safe_extract(row: dict, source_name: str) -> dict | None:
    """Return a partial item dict or ``None`` when the row is unusable.

    The returned dict exposes:
      - prompt, response_a, response_b
      - model_a_family, model_b_family
      - source_winner: one of "A", "B", "Tie", None (from dataset metadata)
      - chosen_slot: "A" if response_a is the dataset-preferred answer, "B"
                     if response_b is, or None when the source has no winner.
    """
    try:
        if source_name in ("lmsys", "mt_bench"):
            conv_a = row.get("conversation_a")
            conv_b = row.get("conversation_b")
            if conv_a is None or conv_b is None:
                return None
            if len(conv_a) < 2 or len(conv_b) < 2:
                return None

            prompt = conv_a[0].get("content", "") if isinstance(conv_a[0], dict) else ""
            response_a = conv_a[1].get("content", "") if isinstance(conv_a[1], dict) else ""
            response_b = conv_b[1].get("content", "") if isinstance(conv_b[1], dict) else ""
            # Multi-turn safety check: only keep conversations that are a single
            # user -> assistant exchange.
            if len(conv_a) > 2 or len(conv_b) > 2:
                return None

            model_a = row.get("model_a")
            model_b = row.get("model_b")

            winner_col = row.get("winner")
            source_winner: str | None = None
            if isinstance(winner_col, str):
                lowered = winner_col.lower()
                if lowered in {"model_a", "a"}:
                    source_winner = "A"
                elif lowered in {"model_b", "b"}:
                    source_winner = "B"
                elif "tie" in lowered:
                    source_winner = "Tie"
            return {
                "prompt": prompt,
                "response_a": response_a,
                "response_b": response_b,
                "model_a_family": infer_model_family(model_a),
                "model_b_family": infer_model_family(model_b),
                "source_winner": source_winner,
                "chosen_slot": source_winner if source_winner in {"A", "B"} else None,
            }

        if source_name == "hh_rlhf":
            chosen = row.get("chosen", "")
            rejected = row.get("rejected", "")
            prompt, chosen_resp, rejected_resp = _hh_rlhf_single_turn(chosen, rejected)
            if prompt is None:
                return None
            # By construction ``chosen`` is the preferred answer; we put it in
            # slot A here and randomise later so that slot A is not always
            # the dataset-preferred answer.
            return {
                "prompt": prompt,
                "response_a": chosen_resp,
                "response_b": rejected_resp,
                "model_a_family": None,
                "model_b_family": None,
                "source_winner": "A",  # chosen is A before randomisation
                "chosen_slot": "A",
            }

        return None
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def is_valid(p: str, ra: str, rb: str) -> bool:
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
    if _contains_leak(p, ra, rb):
        return False
    return True


def _randomise_slot(item: dict, rng: random.Random) -> dict:
    """Flip response_a/response_b for half the items, preserving ground truth.

    The ``source_winner`` and ``chosen_slot`` fields are updated to point at
    the *new* slot, so downstream scoring can still recover who was the
    dataset-preferred answer regardless of which slot it ended up in.
    """
    if rng.random() < 0.5:
        item["response_a"], item["response_b"] = item["response_b"], item["response_a"]
        item["model_a_family"], item["model_b_family"] = item["model_b_family"], item["model_a_family"]
        if item.get("source_winner") in {"A", "B"}:
            item["source_winner"] = "B" if item["source_winner"] == "A" else "A"
        if item.get("chosen_slot") in {"A", "B"}:
            item["chosen_slot"] = "B" if item["chosen_slot"] == "A" else "A"
    return item


def fetch_and_transform_all_sources() -> None:
    lmsys_df = get_items_from_source("lmsys/chatbot_arena_conversations", "train")
    mt_bench_df = get_items_from_source("lmsys/mt_bench_human_judgments", "human")
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
    per_source_cap = {domain: max(count // 3, 5) for domain, count in target_counts.items()}

    sources = [
        (lmsys_df, "lmsys"),
        (mt_bench_df, "mt_bench"),
        (hh_rlhf_df, "hh_rlhf"),
    ]

    stratified_items: dict[str, list[dict]] = {"code": [], "factual": [], "creative": []}
    source_domain_counts: dict[str, dict[str, int]] = {
        source_name: {"code": 0, "factual": 0, "creative": 0}
        for _, source_name in sources
    }

    rng = random.Random(RANDOMISATION_SEED)

    def try_add(extracted: dict | None, source_name: str, cap: dict | None) -> bool:
        if extracted is None:
            return False
        if not is_valid(extracted["prompt"], extracted["response_a"], extracted["response_b"]):
            return False
        domain = classify_domain(extracted["prompt"])
        if len(stratified_items[domain]) >= target_counts[domain]:
            return False
        if cap is not None and source_domain_counts[source_name][domain] >= cap[domain]:
            return False
        if any(x["prompt"] == extracted["prompt"] for x in stratified_items[domain]):
            return False

        # Randomise position for ANY item that carries a source_winner; this
        # balances slot assignment across the whole dataset, not just hh_rlhf.
        _randomise_slot(extracted, rng)

        item = {
            "item_id": f"{source_name}-{domain}-{len(stratified_items[domain]):03d}",
            "domain": domain,
            "difficulty": classify_difficulty(extracted["prompt"]),
            "prompt": extracted["prompt"],
            "response_a": extracted["response_a"],
            "response_b": extracted["response_b"],
            "model_a_family": extracted.get("model_a_family"),
            "model_b_family": extracted.get("model_b_family"),
            "source_winner": extracted.get("source_winner"),
            "chosen_slot": extracted.get("chosen_slot"),
            "human_annotations": [],
            "majority_preference": "Tie",
            "is_gold_standard": False,
        }
        stratified_items[domain].append(item)
        source_domain_counts[source_name][domain] += 1
        return True

    for df, source_name in sources:
        for _, row in df.sample(frac=1, random_state=42).iterrows():
            if all(len(stratified_items[d]) >= target_counts[d] for d in target_counts):
                break
            extracted = safe_extract(row.to_dict() if hasattr(row, "to_dict") else dict(row), source_name)
            try_add(extracted, source_name, cap=per_source_cap)

    for df, source_name in sources:
        for _, row in df.sample(frac=1, random_state=99).iterrows():
            if all(len(stratified_items[d]) >= target_counts[d] for d in target_counts):
                break
            extracted = safe_extract(row.to_dict() if hasattr(row, "to_dict") else dict(row), source_name)
            try_add(extracted, source_name, cap=None)

    all_items = (
        stratified_items["code"]
        + stratified_items["factual"]
        + stratified_items["creative"]
    )

    if len(all_items) < 100:
        print(f"\nWARNING: Only sourced {len(all_items)}/100 items. Check source data quality.")

    source_breakdown = dict(Counter(i["item_id"].split("-")[0] for i in all_items))
    domain_breakdown = dict(Counter(i["domain"] for i in all_items))
    slot_balance = dict(Counter(i.get("chosen_slot") for i in all_items if i.get("chosen_slot")))
    print(f"\nSource breakdown: {source_breakdown}")
    print(f"Domain breakdown: {domain_breakdown}")
    print(f"chosen_slot balance after randomisation: {slot_balance}")

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
