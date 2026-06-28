"""Adjudicate real CE claims against the real plan terms (Coverage Atlas demo).

Runs a small matrix of realistic device claims against each plan's actual
T&C text (from data/policies/ce_atlas_raw/), using the CE-native adjudicate
prompt. Shows how the SAME claim is decided differently under different plans
— and that the engine cites the governing clause and routes the ambiguous /
silent cases to a human instead of auto-paying them.

Needs an LLM key (OPENAI_API_KEY; optionally OPENAI_REASONER_MODEL for the
frontier tier). Run:
    uv run --project backend python scripts/adjudicate_ce_demo.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.adjudication.pipeline import process_claim  # noqa: E402

RAW_DIR = ROOT / "data" / "policies" / "ce_atlas_raw"
CE_PROMPT = "adjudicate_ce_v1"

PLANS = [
    ("AppleCare+", "applecare_plus.md"),
    ("Samsung Care+", "samsung_care_plus.md"),
    ("Allstate Protection", "allstate_protection.md"),
]

CLAIMS = [
    "Dropped my phone and the screen cracked; everything else works. I have the plan, no prior claims.",
    "My phone battery swelled up and won't hold a charge after 14 months of normal use.",
    "My phone was stolen from my gym bag yesterday. I have the standard protection plan.",
    "Spilled coffee on my device and now it won't power on at all.",
    "There's a small scratch on the back housing and I'd like it replaced.",
]


async def main() -> int:
    policy_texts = {name: (RAW_DIR / fn).read_text(encoding="utf-8") for name, fn in PLANS}

    print(f"{'plan':<20} {'claim':<46} {'outcome':<11} {'conf':<7} {'route':<12} cited clause")
    print("-" * 130)

    async def run(plan: str, claim: str) -> tuple[str, str, object]:
        r = await process_claim(
            raw_input=claim,
            policy_text_override=policy_texts[plan],
            adjudicate_prompt_override=CE_PROMPT,
            verify_support=True,
        )
        return plan, claim, r

    tasks = [run(plan, claim) for plan, _ in PLANS for claim in CLAIMS]
    # Bounded concurrency
    sem = asyncio.Semaphore(4)

    async def gated(coro):  # noqa: ANN001
        async with sem:
            return await coro

    results = await asyncio.gather(*(gated(t) for t in tasks))

    for plan, claim, r in results:
        clause = (r.decision.policy_citation or "")[:42].replace("\n", " ")
        flags = "+".join(r.route_reasons) if r.route_reasons else ""
        print(
            f"{plan:<20} {claim[:44]:<46} {r.decision.outcome:<11} "
            f"{r.decision.confidence:<7} {r.route:<12} \"{clause}\""
            + (f"  [{flags}]" if flags else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
