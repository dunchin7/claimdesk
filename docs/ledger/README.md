# Reserve & Leakage Ledger

**The two-sided P&L for automated claims.** Every claims-AI tool prices the dollars you lose to wrong **approvals** (leakage). This prices the other half — the dollars you lose to wrong **denials** — and shows you the exact confidence threshold where automating a claims book stops paying.

> Part of [ClaimDesk](../../README.md). ClaimDesk decides warranty/product-protection claims; this is the instrument that answers the money question on top of it — *what is automating those decisions actually worth, once you count the mistakes in both directions?*

---

## Why this exists

An AI claims adjudicator makes two kinds of costly mistake:

| Error | What happens | Who prices it today |
|---|---|---|
| **Leakage** | auto-**approve** a claim you shouldn't pay | everyone — it's the industry's headline metric |
| **False denial** | auto-**reject** a valid claim | **nobody** |

A false denial is usually the *more* expensive one: you tend to pay the claim anyway on appeal, **plus** appeal/complaint handling and bad-faith exposure (UCSPA/NAIC), **plus** the churn of a wronged customer. Yet it's invisible on every leakage dashboard. This tool puts both on one ledger.

---

## What it shows (on a realistic 200-claim book)

The same claims, decided by a cheap model and a frontier model, priced through one cost model:

| metric (full auto-resolve) | cheap (gpt-4o-mini) | frontier |
|---|---|---|
| decision accuracy | 82% | 86% |
| **leakage** (wrong approvals) | **$0** | **$0** |
| **false-denial** events | **29** | 6 |
| labor saved (LAE) | $2,100 | $1,908 |
| false-denial liability | **$22,950** | $1,450 |
| **net value of automation** | **–$20,850** | **+$458** |
| per 1,000 claims | **–$104,250** | +$2,290 |

Three things fall out of this, and they're the whole point:

1. **Leakage is ~zero on both tiers.** The metric the whole market optimizes is already solved here. The action is entirely on the side nobody prices.
2. **The cheap model's automation is net-negative** — not because it pays bad claims, but because it *denies good ones*. That cost is invisible until you put it on the ledger.
3. **You can't threshold your way out of it.** The cheap model reports ~0.90 confidence on its *wrong* denials too, so raising the auto-resolve bar changes nothing until ~0.95 — where auto-resolve collapses to 10% and you've given up on automation. The frontier model's confidence actually separates good from bad, so it has a real operating point (auto-resolve 72% at conf ≥ 0.90, net +$1,440).

The flagged-claims table makes the failure legible: the cheap model systematically rejects covered *spontaneous-breakdown* claims ("speaker died, display glitching, no drops") — a real, repeatable pattern, not noise.

---

## The dashboard

`index.html` is a single self-contained file (no server, no build):

```bash
open docs/ledger/index.html
```

Live controls:
- **Claim book** — realistic (200) vs the adversarial stress set (30 deliberate traps).
- **Model tier** — cheap vs frontier; watch the false-denial curve move.
- **Cost assumptions** — every coefficient is an editable input. This is by design: the tool doesn't tell you what a wrongful denial costs *your* book — it prices *your* numbers.
- **Auto-resolve threshold** — drag it and watch the two cost curves cross.

The in-browser math mirrors the Python cost model exactly (enforced by `backend/tests/test_dashboard.py`), so the numbers on screen equal the numbers the engine computes.

---

## Reproduce it

Needs `OPENAI_API_KEY` (any OpenAI-compatible endpoint via `OPENAI_BASE_URL`); set `OPENAI_REASONER_MODEL` to point the "frontier" runs at a stronger model.

```bash
# 1. generate a realistically-distributed book with gold labels fixed by construction
uv run --project backend python scripts/generate_ce_book.py --n 200

# 2. score it on each model tier (real LLM calls; writes book_synth_<tier>.json)
uv run --project backend python scripts/score_book.py --tier cheap    --tag synth --claims data/atlas_experiment/synth_claims.jsonl
uv run --project backend python scripts/score_book.py --tier frontier --tag synth --claims data/atlas_experiment/synth_claims.jsonl

# 3. price the two-sided P&L + crossover sweep
uv run --project backend python scripts/build_ledger.py --tag synth

# 4. regenerate the dashboard
uv run --project backend python scripts/build_dashboard.py
```

---

## How it's built

```
backend/app/ledger/
├── cost_model.py   # leakage / false-denial / labor cost per decision (cited, editable coefficients)
└── ledger.py       # price a book -> two-sided P&L + sweep_threshold() crossover curve
scripts/
├── generate_ce_book.py   # realistic book, gold labels by construction
├── score_book.py         # decide each claim + capture confidence, per model tier
├── build_ledger.py       # the P&L + crossover, printed + ledger_*.json
└── build_dashboard.py    # generate the self-contained dashboard
backend/tests/
├── test_ledger.py        # cost-model + sweep unit tests
└── test_dashboard.py     # JS↔Python parity (the on-screen numbers are trustworthy)
```

The cost model is deterministic and unit-tested; the decisions are genuine output from two real models on every claim.

---

## What's real, and what's an assumption

Stated plainly, because the difference matters:

**Real:** the claim *distribution* is realistic (66% approve / 23% reject / 11% needs_info); gold labels are ground truth by construction; the decisions and confidences are genuine model output on all 200 claims, both tiers; the per-1,000 figures come from that book, not a handful of cherry-picked cases; the engine math is tested.

**Assumptions (editable inputs, not measured facts):** the cost coefficients (review labor, dispute EV, churn) are plausible cited defaults, not derived from a real book; claim values come from a keyword heuristic; confidence is the model's self-reported number, not a trained calibrator; and the claims are *synthetic* templated text, not a real insurer's loss runs.

So this is a **working instrument with a real, non-obvious finding** — not a validated measurement of any specific book. Point it at real coefficients and real claims and it does the same thing on day one; that's the design.

---

## The one-line version

Everyone shows *"our AI decides claims."* This shows *"here's what automating those decisions is actually worth — including the expensive half (wrong denials) that nobody else puts a dollar on, and the threshold where it flips from profit to loss."*
