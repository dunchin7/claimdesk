# ClaimDesk

**An AI claims co-pilot for warranty / product-protection adjudication.** It reads a customer's claim in plain English, decides it against the policy, drafts the reply — and, crucially, knows when *not* to trust itself and hand off to a human.

> Built solo as a deep exploration of the claims-adjudication problem that sits at the center of every TPA (third-party administrator). Runs on synthetic data; the interesting work was less "make a demo that works" and more "find exactly where automation stops being trustworthy."

---

## The idea in three sentences

An LLM is fluent but a poor judge of its own confidence — it will state a wrong answer as surely as a right one. So ClaimDesk is built around that flaw: a separate model scores how likely each decision is *actually* correct, and only the narrow band it can prove is safe gets auto-resolved. Everything else routes to a human, and every decision quotes the exact policy clause it relied on so it stays defensible.

That's the whole thesis: **automate the easy 80%, prove it's safe, and put a human on the hard 20% — never the other way around.**

---

## How it works

```
customer message
      │
      ▼
1. EXTRACT      LLM → structured claim facts (SKU, failure mode, severity, evidence…)
      │
      ▼
2. ADJUDICATE   LLM + policy → decision (approve / reject / needs_info) + a quoted policy clause
      │
      ▼
3. VERIFY       the quoted clause must be a verbatim substring of the real policy, or confidence is downgraded
      │
      ▼
4. DRAFT        LLM → the customer-facing email
      │
      ▼
5. SCORE        a calibrated model → P(this decision is correct)
      │
      ▼
6. ROUTE        ≥0.70 auto-resolve · 0.60–0.70 quick human check · <0.60 full human review
```

Each step is a small, focused job rather than one giant prompt — easier to make reliable, test, and reason about.

---

## The parts that actually matter

These are the design choices that separate this from a demo:

- **Calibrated trust, not vibes.** The LLM's self-reported "high confidence" was correct only ~77% of the time — useless for auto-deciding. So a separate model (gradient-boosted trees) turns 16 signals about each decision into an honest probability it's correct, and that number — not the LLM's word — gates auto-resolution.
- **Defensible decisions.** Every decision must quote a policy clause *verbatim*. The quote is checked character-for-character against the source; a paraphrase or hallucination fails the check, downgrades confidence, and routes to a human. A denial you can't cite is a denial you can't ship.
- **Eval-gated, like CI.** Correctness is measured against a locked 200-claim benchmark. A change that drops accuracy more than 2 points fails the build. Prompts are never tuned against the benchmark itself (that's overfitting) — iteration happens on a separate dev set.
- **Safe by default.** When confidence is low, or a fraud signal fires, or a cost cap trips, the claim goes to a human. The system is designed to fail toward review, not toward a wrong auto-decision.

---

## Results (on a 200-claim synthetic benchmark)

| Metric | Value |
|---|---|
| Decision accuracy | ~90% |
| Auto-resolve rate | ~80% |
| Accuracy *on the auto-resolved band* | ~99% |
| Citations passing the verbatim check | 99.5% |
| p50 / p95 latency | ~7s / ~10s |
| Cost per claim | ~$0.001 |

**Read these honestly:** they're synthetic. The harder, more interesting question is how a system like this behaves on real, messy claims where customers don't quote SKUs or dates and evidence is thin — the "cold-start" problem of going live on a new book of business. That's the frontier this is built to explore.

---

## Tech & why

A deliberately boring, single-operator stack — every choice favors fewer moving parts:

| Layer | Choice | Why |
|---|---|---|
| API / pipeline | Python · FastAPI · async | one language end to end |
| Data | Postgres + pgvector | relational + vectors + queue + state in **one** database |
| LLM access | LiteLLM + Instructor | provider-agnostic; structured (typed) outputs, not parsed prose |
| Orchestration | LangGraph | parallel specialist steps + durable, crash-resumable execution |
| Trust / fraud models | XGBoost | calibrated probabilities from tabular signals; small, fast, proven |
| LLM provider | OpenAI via LiteLLM | two tiers — a cheap model for extraction/vision, an optional frontier model for the hard reasoning (adjudication + citation verification), configurable per alias |

Guiding rules: **managed over self-hosted; provider-agnostic at the boundary; structured outputs over free text; one language, one DB; prove every choice with an eval before shipping it** (several "fancier" options — RAG, hybrid search, a planner, an agent memory — were built, measured, found not to help at this scale, and gated off behind flags with the re-enable condition documented).

---

## Run it locally

Requires [Docker](https://docs.docker.com/get-docker/), [uv](https://docs.astral.sh/uv/), and an OpenAI API key.

```bash
cp .env.example .env          # add your OpenAI key
just setup                    # docker up + deps + migrate
just synth                    # generate the synthetic claim set
just dev                      # API on :8000
just demo                     # run a sample claim through the pipeline
just eval                     # score the pipeline against the synthetic benchmark
```

---

## Repo layout

```
backend/app/
├── adjudication/   extract → adjudicate → verify-citation → draft pipeline
├── ai/             LLM abstraction, prompts (versioned), schemas, agents, tools
├── confidence/     the calibrator (turns signals into P(decision correct))
├── fraud/          XGBoost fraud scorer + features
├── retrieval/      pgvector search, chunkers, hybrid/rerank (gated)
├── vision/         photo damage classification + EXIF / AI-image checks
├── api/            FastAPI routes (claims, admin queue, webhooks)
├── hitl/           human-in-the-loop routing + operator queue
├── evals/          the benchmark runner, judges, report cards
└── db/             SQLAlchemy models + Alembic migrations
scripts/            synthetic data generation, eval, training, ingestion
```

---

## Scope & honesty

Solo build. Synthetic policy and synthetic claims throughout. It's a prototype meant to explore the problem space and the trust boundary, not a production system — and it's deliberately honest about where it would break on real data. That tension (works on clean cases, must earn trust on messy ones) is the whole point.

---

## License

[Apache-2.0](LICENSE).
