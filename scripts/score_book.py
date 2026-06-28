"""Score the claim book with per-claim confidence, for the ledger sweep.

Runs a single, realistic adjudication per claim (full policy text + claim ->
outcome + a calibrated confidence) on whichever model the environment points
at, and writes a book file the Reserve & Leakage Ledger can sweep.

    # cheap tier
    OPENAI_CHAT_MODEL=... OPENAI_REASONER_MODEL=... \\
      uv run --project backend python scripts/score_book.py --tier cheap

    # frontier tier (point the env at the frontier deployment)
      ... --tier frontier

Writes data/atlas_experiment/book_<tier>.json:
    [{id, plan, gold, decision, confidence, claim_value, controlling}]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from pydantic import BaseModel, Field  # noqa: E402

from app.ai.llm import chat  # noqa: E402
from app.ledger import estimate_claim_value  # noqa: E402

EXP = ROOT / "data" / "atlas_experiment"
RAW = ROOT / "data" / "policies" / "ce_atlas_raw"
PLAN_DOC = {
    "applecare": "applecare_plus.md",
    "samsung": "samsung_care_plus.md",
    "allstate": "allstate_protection.md",
}
MODEL = "reasoner"


class LedgerDecision(BaseModel):
    outcome: Literal["approve", "reject", "needs_info"]
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Your calibrated probability (0-1) that this outcome is correct.",
    )
    rationale: str = Field(default="", description="one sentence")


async def _decide(plan_doc: str, plan_name: str, claim: str) -> LedgerDecision:
    prompt = (
        f"You are a claims adjudicator for {plan_name}. Decide the claim strictly per the policy.\n\n"
        f"<policy>\n{plan_doc}\n</policy>\n\n<claim>\n{claim}\n</claim>\n\n"
        "Decide outcome (approve / reject / needs_info). Approve only if the policy clearly covers it; "
        "reject if an exclusion or unmet condition applies; needs_info if you genuinely can't tell. "
        "Then give your CALIBRATED confidence (0-1) that this outcome is correct — be honest, use the "
        "full range; lower it when the claim is ambiguous or the policy is silent."
    )
    return await chat(
        messages=[{"role": "user", "content": prompt}],
        model_alias=MODEL,
        response_model=LedgerDecision,
        temperature=0.0,
    )


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", required=True, help="label for the output file, e.g. cheap | frontier")
    ap.add_argument("--claims", default=str(EXP / "claims.jsonl"), help="path to the claim set (jsonl)")
    ap.add_argument("--tag", default="", help="dataset tag; output is book_<tag>_<tier>.json (blank -> book_<tier>.json)")
    args = ap.parse_args()

    claims = [json.loads(l) for l in Path(args.claims).read_text().splitlines() if l.strip()]
    docs = {k: (RAW / v).read_text(encoding="utf-8") for k, v in PLAN_DOC.items()}
    plan_names = json.loads((EXP / "provisions.json").read_text())
    sem = asyncio.Semaphore(6)
    done = 0

    async def one(c: dict) -> dict | None:
        nonlocal done
        async with sem:
            try:
                d = await _decide(docs[c["plan"]], plan_names[c["plan"]]["plan_name"], c["raw_input"])
                done += 1
                print(f"[score] {done}/{len(claims)} {c['id']:<4} {d.outcome:<10} conf={d.confidence:.2f}", flush=True)
                return {
                    "id": c["id"], "plan": c["plan"], "gold": c["gold_outcome"],
                    "decision": d.outcome, "confidence": d.confidence,
                    "claim_value": estimate_claim_value(c["raw_input"]),
                    "controlling": c["controlling_provision"],
                }
            except Exception as e:  # noqa: BLE001
                done += 1
                print(f"[score] {done}/{len(claims)} {c['id']:<4} FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
                return None

    rows = [r for r in await asyncio.gather(*(one(c) for c in claims)) if r is not None]
    prefix = f"{args.tag}_" if args.tag else ""
    out = EXP / f"book_{prefix}{args.tier}.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n[score] wrote {out} ({len(rows)} claims)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
