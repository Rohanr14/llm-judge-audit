# LLM Judge Calibration Suite — Project Overview & Blueprint

---

## Project Summary

**Name:** LLM Judge Calibration Suite (`llm-judge-audit`)
**Type:** Open-source Python CLI tool + HuggingFace leaderboard + arXiv preprint
**Hardware required:** MacBook Air M2 (or any machine with API access)
**Compute required:** Zero local inference — pure API calls
**Timeline:** 4–6 weeks solo

---

## The Problem

LLM-as-judge has become the default evaluation method across industry and research. Almost every team that ships AI products, runs model comparisons, or publishes evaluation results relies on a language model to score other language models. The assumption baked into every one of these pipelines is that the judge is reliable — that it scores consistently, resists manipulation, and aligns reasonably well with human judgment.

That assumption has never been systematically tested in a reusable way.

Individual biases are documented across a dozen papers in isolation:
- CALM identified 12 bias categories
- SWAY showed sycophancy-detecting judges are themselves sycophantic
- The Sage framework showed even frontier models reverse preferences in ~25% of hard cases
- Position bias, verbosity bias, self-enhancement bias, and anchoring have independent replications

But every paper tests one or two biases with custom, non-reusable code. No unified tool exists to audit a judge before a team commits to it. The result is that most evaluation pipelines are built on an untested foundation.

**The true need:** Teams need trust calibration — a definitive, standardized answer to "is this judge reliable enough to use as a proxy for human judgment on my specific task?" before they ship results or make model selection decisions based on automated evals.

---

## The Solution

A pip-installable CLI (`llm-judge-audit`) that:
1. Accepts any judge model via API key + model name
2. Runs a 10-dimension bias test suite
3. Compares results against a human-annotated ground-truth dataset
4. Produces a severity-weighted report card with a single composite score (the **Judge Reliability Index**)
5. Outputs concrete, actionable mitigation recommendations for every flagged bias
6. Feeds results into a live HuggingFace Spaces leaderboard

---

## The 10 Bias Dimensions

| # | Bias | What it measures | Test method |
|---|------|-----------------|-------------|
| 1 | **Position bias** | Does the judge prefer whichever response appears first (or second)? | Swap response order in pairwise comparisons; measure reversal rate |
| 2 | **Verbosity bias** | Does the judge reward length independent of quality? | Pad the baseline-losing response with semantically empty filler; measure preference flip rate toward padded losers |
| 3 | **Self-enhancement bias** | Does the judge favor outputs from its own model family? | Present known GPT-4/Claude/Gemini outputs; test each with the matching judge |
| 4 | **Sycophancy bias** | Does the judge capitulate when given a "suggested rating"? | Provide judge with prior scores or user opinions; measure drift toward those anchors |
| 5 | **Anchoring bias** | Do prior scores in the session skew subsequent scores? | Vary what scores appear before the target item; measure pull effect |
| 6 | **Cross-run consistency** | Does the judge give the same score on the same input twice? | Re-submit identical inputs; measure score variance across N runs |
| 7 | **Recency bias** | Do few-shot examples close to the query anchor the judge's scoring? | Vary which examples appear last in the prompt; measure influence on output |
| 8 | **Format bias** | Does the judge score markdown/bulleted responses higher than plain prose? | Present identical content in different formats; measure score delta |
| 9 | **Confidence-consistency gap** | Is the judge's expressed confidence calibrated to its actual stability? | Correlate language like "clearly better" against re-run reversal rates |
| 10 | **Domain transfer bias** | Does the judge's reliability hold across domains? | Evaluate code, factual, and creative domains; measure score distribution shift |

---

## Scoring System

### Human Alignment Score (HAS)
A fixed, 100-item human-annotated dataset (pairwise preferences labeled by 3+ human raters per item, sourced via Prolific, ~$150) serves as ground truth. The HAS is the percentage of cases where the judge's preference matches the majority human preference. This answers not just "is the judge consistent?" but "is the judge correct?"

### Severity-weighted thresholds

| Severity | Biases | Flag threshold |
|----------|--------|---------------|
| **Critical** | Position | Score > 0.25 |
| **Critical** | Sycophancy, self-enhancement | Score > 0.20 |
| **Moderate** | Verbosity, anchoring | Score > 0.30 |
| **Moderate** | Domain transfer | Score > 0.25 |
| **Minor** | Recency, format, confidence gap, cross-run | Score > 0.35 |

Thresholds are calibrated against published frontier-judge numbers:
Zheng et al. (MT-Bench / LLM-as-a-Judge, 2023) report position-flip rates
around 0.20–0.30 on strong GPT-4 judges, and the CALM benchmark (Ye et al.,
2024) gives comparable ranges. The previous 0.15 cutoff was stricter than
any published frontier model and would have flagged essentially every
judge; the current numbers represent "should worry" lines rather than
"basically anyone passes" lines.

### Judge Reliability Index (JRI)
A single composite score from 0–100, combining all weighted bias scores and the HAS:

```
JRI = 100 * (has_weight * HAS + (1 - has_weight) * (1 - mean_weighted_bias))
```

where `mean_weighted_bias` is the severity-weighted mean of the ten bias
dimensions (weights: critical=3, moderate=2, minor=1).

**Default `has_weight = 0.6` (60% HAS / 40% bias).** This weighting says
"correctness against humans matters more than any individual bias, but
bias is still a meaningful chunk of the composite". The split is
configurable via the `--has-weight` CLI flag (range 0.0–1.0) if you want
to weight a specific judge against a specific use case differently; the
default is documented in every report for reproducibility.

| JRI range | Interpretation |
|-----------|---------------|
| > 75 | Reliable for most use cases |
| 50–75 | Use with caution; mitigate flagged biases first |
| < 50 | Do not use without significant mitigation |

The JRI gives teams a single number they can report in methods sections and internal evaluation docs — a shared unit of measurement for judge reliability.

### Concurrency and checkpointing

Running the full 10-bias suite against a rate-limited judge can easily hit
5000+ API calls. The CLI exposes two knobs for that:

* `--max-concurrency N` (default `4`) caps process-wide concurrent judge
  requests. Lower this if you're getting 429s; raise it if your provider's
  rate limit is generous.
* `--cache-path PATH` enables a disk-backed JSONL cache keyed on
  `(model, prompt, response_a, response_b)`. A crashed audit can resume
  against the same path without re-spending API credits, and re-running a
  subset of biases against the same model becomes effectively free. Only
  deterministic (`temperature=0`) calls without few-shot history are
  cached -- cross-run, confidence-gap, anchoring, and recency always go
  fresh so their variability is real.

---

## Mitigation Recommendations (auto-generated per flagged bias)

| Bias | Recommended mitigation |
|------|----------------------|
| Position bias | Run all pairwise evaluations in both orders; average the scores |
| Verbosity bias | Add explicit length-independence instruction to judge system prompt |
| Self-enhancement | Use a judge from a different model family than the one being evaluated |
| Sycophancy | Strip suggested ratings or user sentiment from judge context |
| Anchoring | Randomize evaluation order; reset context between items |
| Cross-run consistency | Run N=3 evaluations per item and take the majority vote |
| Recency bias | Randomize few-shot example order |
| Format bias | Normalize response format before presenting to judge |
| Confidence gap | Ignore confidence language; use only the final score |
| Domain transfer | Use a domain-specific judge or fine-tune prompt engineering per domain |

---

## Deliverables

| Deliverable | Description |
|-------------|-------------|
| `pip install llm-judge-audit` | CLI tool accepting any judge via API key + model name |
| GitHub repo | Full test suite, human-annotated anchor dataset (CC-BY), scoring code, reproduction instructions |
| HuggingFace dataset | 100-item human-annotated ground-truth dataset + bias test scenarios |
| HuggingFace Spaces leaderboard | JRI scores across 8+ judge models, updated quarterly, open to community PRs |
| Blog post | "We audited 8 popular LLM judges — here's what we found" — domain-stratified heatmaps, severity breakdowns, mitigation guide |
| arXiv preprint | Introduces the JRI as a standardized judge reliability metric; positions this as the first systematic, unified audit framework |

---

## Prolific Annotation Workflow (practical)

1. Export tasks to Prolific from `llm_judge_audit/datasets/prolific_tasks.csv`.
2. Collect a CSV export with at least:
   - `item_id`
   - participant/rater id (default expected column: `participant_id`)
   - `preference` (`A`, `B`, or `Tie`)
   - optional `confidence` (1–5) and `rationale`
3. Merge annotations back into the anchor dataset:

```bash
python -m llm_judge_audit.datasets.prolific \
  --anchor llm_judge_audit/datasets/anchor.json \
  --annotations prolific_export.csv \
  --output llm_judge_audit/datasets/anchor_annotated.json
```

This recomputes `majority_preference`, updates `human_annotations`, and marks unanimous items as `is_gold_standard=true`.

While responses are still coming in, you can monitor progress without writing merged output:

```bash
python -m llm_judge_audit.datasets.prolific \
  --anchor llm_judge_audit/datasets/anchor.json \
  --annotations prolific_export.csv \
  --progress-only
```

If you want to snapshot partial results before every item has 3 annotations, add `--allow-partial` to the merge command.

---

## Roadmap

### Week 1 — Foundation

**Goal:** Scaffolding, dataset design, and the first two bias tests running end-to-end.

Tasks:
- [x] Set up Python package structure (`llm-judge-audit/`)
- [x] Implement judge interface: accepts model name + API key, standardizes pairwise comparison calls
- [x] Design the 100-item human-annotated anchor dataset schema (domain, prompt, response A, response B, human preference)
- [x] Source human annotations via Prolific (launch study, ~$150 budget, 3 raters per item)
- [x] Implement bias test #1: **position bias** (swap response order, measure reversal rate)
- [x] Implement bias test #2: **verbosity bias** (pad the losing response, measure flip rate toward the padded loser)
- [x] Write unit tests for the judge interface and first two bias modules

**End-of-week checkpoint:** CLI runs `llm-judge-audit --model gpt-4o --tests position,verbosity` and produces a structured JSON report.

---

### Week 2 — Core Bias Suite

**Goal:** All 10 bias dimensions implemented and tested against GPT-4o.

Tasks:
- [ ] Implement bias tests #3–6: self-enhancement, sycophancy, anchoring, cross-run consistency
- [ ] Implement bias tests #7–10: recency, format, confidence gap, domain transfer
- [ ] Build the domain stratification layer (code / factual / creative splits per test)
- [ ] Collect Prolific annotations (should be complete by mid-week)
- [ ] Implement Human Alignment Score (HAS) computation against anchor dataset
- [ ] Run full suite against GPT-4o as calibration/sanity check
- [ ] Refine test cases for any biases that don't clearly manifest on first run

**End-of-week checkpoint:** All 10 bias modules run. HAS computation works. GPT-4o produces a complete JRI report card.

---

### Week 3 — Scoring, CLI, and Leaderboard

**Goal:** JRI scoring system finalized, CLI polished, leaderboard live.

Tasks:
- [ ] Implement severity-weighted JRI formula; calibrate thresholds against GPT-4o results
- [ ] Implement per-bias mitigation recommendation engine
- [ ] Build the report card output: JSON (machine-readable) + terminal pretty-print + optional HTML export
- [ ] Run full suite against 7 additional judge models: Claude 3.5 Sonnet, Gemini 1.5 Pro, Mistral Large, LLaMA-3.1-70B, Qwen2.5-72B, Command R+, GPT-4o-mini
- [ ] Stand up HuggingFace Spaces leaderboard (static table initially, with update workflow documented)
- [ ] Write CONTRIBUTING.md so community can submit new model results via PR

**End-of-week checkpoint:** 8 models benchmarked. JRI leaderboard live on HuggingFace Spaces. `pip install` works from TestPyPI.

---

### Week 4 — Validation and Writeup

**Goal:** Results validated, blog post drafted, arXiv preprint submitted.

Tasks:
- [ ] Stress-test CLI against edge cases (malformed API responses, rate limits, model refusals)
- [ ] Validate JRI scores: do high-JRI models actually perform better in downstream eval tasks? Run a simple sanity check using a held-out task set
- [ ] Create visualizations: per-model heatmaps (bias × domain), JRI distribution bar chart, HAS vs. JRI scatter plot
- [ ] Write blog post (~2,500 words): framing, methodology, results, bias-by-bias breakdown, mitigation guide, leaderboard
- [ ] Write arXiv preprint: introduce JRI formally, review related work (CALM, SWAY, Sage, position bias papers), describe methodology, report results across 8 models
- [ ] Publish to PyPI (stable v1.0.0)
- [ ] Submit arXiv preprint
- [ ] Post blog post and announce on Twitter/X, HuggingFace, r/MachineLearning, r/LocalLLaMA

**End-of-week checkpoint:** Everything shipped. GitHub repo public. Blog post live. arXiv submitted.

---

### Ongoing (post-launch)
- Accept community PRs to add new model results to the leaderboard
- Update leaderboard quarterly as new frontier models release
- Respond to issues and iterate on bias test cases based on community feedback
- Track citations in papers using LLM-as-judge (target: papers reporting their judge's JRI score)

---

## Technical Architecture

```
llm-judge-audit/
├── llm_judge_audit/
│   ├── __init__.py
│   ├── cli.py                  # Click-based CLI entrypoint
│   ├── judge.py                # Judge interface: wraps any model via API
│   ├── biases/
│   │   ├── position.py
│   │   ├── verbosity.py
│   │   ├── self_enhancement.py
│   │   ├── sycophancy.py
│   │   ├── anchoring.py
│   │   ├── cross_run.py
│   │   ├── recency.py
│   │   ├── format_bias.py
│   │   ├── confidence_gap.py
│   │   └── domain_transfer.py
│   ├── scoring/
│   │   ├── jri.py              # JRI formula + severity weighting
│   │   ├── has.py              # Human Alignment Score computation
│   │   └── thresholds.py       # Configurable flag thresholds
│   ├── mitigations.py          # Per-bias recommendation engine
│   ├── report.py               # JSON + terminal + HTML report generation
│   └── datasets/
│       ├── anchor.json         # 100-item human-annotated ground truth
│       └── bias_scenarios/     # Test scenarios per bias dimension
├── tests/
├── leaderboard/                # Static data + update scripts for HF Spaces
├── notebooks/                  # Reproduction notebooks for blog/preprint
├── pyproject.toml
└── README.md
```

---

## Key Design Decisions

**Why pairwise comparisons only (not single-response scoring)?**
Pairwise comparison is the dominant LLM-as-judge pattern in practice (used by Chatbot Arena, most RLHF pipelines, and preference datasets). Absolute scoring has different bias profiles. This tool focuses on where most real evaluation actually happens.

**Why 100 human-annotated items (not 1,000)?**
Statistical power analysis: detecting a 10% HAS difference with 80% power requires ~85 items. 100 gives a small buffer and keeps Prolific costs manageable (~$150). The anchor dataset is held fixed across all model evaluations, so comparability is preserved.

**Why domain stratification across all 10 biases?**
A judge used for code review has a completely different bias profile than one used for creative writing. Aggregate scores hide domain-specific failures. Teams selecting a judge for a specific task need domain-specific reliability data, not just overall averages.

**Why a single composite JRI score?**
Coordination. Papers need something to cite, teams need something to put in their docs, and practitioners need something to compare. A composite score with transparent weighting is more useful than 10 separate numbers when the primary use case is "should I use this judge or not?"

---

## Success Metrics

| Metric | 30-day target | 90-day target |
|--------|--------------|--------------|
| GitHub stars | 200 | 800 |
| PyPI downloads | 500 | 3,000 |
| Papers citing JRI | 0 | 3–5 |
| Models on leaderboard | 8 | 15+ |
| HuggingFace dataset downloads | 200 | 1,000 |

The 90-day target that matters most: at least one AI evaluation paper reports their judge model's JRI score in its methods section. That's the signal that JRI has become a unit of measurement, not just a tool.

---

## Why This Project, Why Now

The timing is optimal for three reasons:

1. **Karpathy's "evaluation crisis" declaration (March 2025)** created explicit, named demand for better evaluation tooling. The problem has a widely-recognized label and an audience primed to care.

2. **LLM-as-judge adoption is at peak ubiquity.** Every team uses it, which means every team has a stake in knowing whether their judge is trustworthy. The audience for this tool is essentially the entire applied AI community.

3. **The research is fragmented but mature.** The individual bias papers exist. The CALM taxonomy exists. The Sage results exist. What's missing is synthesis into a reusable artifact — which is an engineering problem, not a research problem, and therefore entirely tractable in four weeks.

The JRI becomes valuable the moment it exists, because it solves a coordination problem: teams can't currently compare judge reliability across papers because everyone measures different things differently. A standard metric changes that immediately.
