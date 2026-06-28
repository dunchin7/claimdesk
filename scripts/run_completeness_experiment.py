"""Completeness experiment: does enumerate-and-check catch omission errors
that a free LLM decision (A) and a generic faithfulness check (B) miss?

Three systems, same information, different structure:
  A  free decision  — full policy text + claim -> decide + cite. (what most claims AI does)
  B  A + generic faithfulness — A's decision, plus a verbatim-citation check and an
     LLM "is the rationale faithful to the policy" judge (the commoditized guardrail layer).
  C  enumerate-and-check — claim + the COMPLETE enumerated provision list; the model must
     judge EVERY provision applicable/not, then decide; an applicable exclusion/unmet
     condition controls; ambiguity -> needs_info.

The dangerous error is OMISSION: approving a claim a provision should have stopped.
Headline metric = wrong approvals on trap claims (leakage), per system, + trap accuracy.

Fairness: A/B get the full policy text (which CONTAINS every exclusion); C gets the same
provisions enumerated. Same information — the only difference is whether consideration is forced.

Run (needs OPENAI_API_KEY; optionally OPENAI_REASONER_MODEL):
    uv run --project backend python scripts/run_completeness_experiment.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from pydantic import BaseModel, Field  # noqa: E402

from app.adjudication.citation import verify_citation  # noqa: E402
from app.ai.llm import chat  # noqa: E402

EXP_DIR = ROOT / "data" / "atlas_experiment"
RAW_DIR = ROOT / "data" / "policies" / "ce_atlas_raw"
PLAN_DOC = {
    "applecare": "applecare_plus.md",
    "samsung": "samsung_care_plus.md",
    "allstate": "allstate_protection.md",
}
MODEL = "reasoner"  # frontier tier if OPENAI_REASONER_MODEL set, else chat model

Outcome = Literal["approve", "reject", "needs_info"]


# ---- structured outputs ----
class ExpDecision(BaseModel):
    outcome: Outcome
    policy_citation: str = Field(default="", description="verbatim clause the decision rests on")
    rationale: str = Field(description="one or two sentences")


class FaithVerdict(BaseModel):
    faithful: bool = Field(description="true if the rationale is supported by the policy text")
    reason: str = ""


class ProvVerdict(BaseModel):
    id: str
    applies: bool
    why: str = ""


class CompletenessDecision(BaseModel):
    provision_verdicts: list[ProvVerdict] = Field(description="one verdict for EVERY provision shown")
    outcome: Outcome
    controlling_clause: str = Field(default="", description="verbatim clause that controls the outcome")
    rationale: str


# ---- system A: free decision ----
async def system_a(plan_doc: str, plan_name: str, claim: str) -> ExpDecision:
    prompt = (
        f"You are a claims adjudicator for {plan_name}. Decide the claim strictly per the policy.\n\n"
        f"<policy>\n{plan_doc}\n</policy>\n\n<claim>\n{claim}\n</claim>\n\n"
        "Decide outcome (approve / reject / needs_info), cite a verbatim clause, and give a 1-2 sentence rationale. "
        "Approve only if the policy covers it; reject if an exclusion or unmet condition applies; needs_info if you can't tell."
    )
    return await chat(messages=[{"role": "user", "content": prompt}], model_alias=MODEL, response_model=ExpDecision, temperature=0.0)


# ---- system B: A + generic faithfulness guard ----
async def system_b(plan_doc: str, a: ExpDecision) -> tuple[Outcome, bool]:
    """Returns (effective_outcome, flagged). Generic guard = verbatim check + faithfulness judge.
    If the citation isn't grounded or the rationale isn't faithful, B routes to needs_info."""
    verbatim_ok = bool(a.policy_citation) and verify_citation(a.policy_citation, plan_doc).verbatim
    prompt = (
        "Is the following decision rationale FAITHFUL to (i.e. supported by) the policy text? "
        "Answer about faithfulness only — not whether the decision is ultimately correct.\n\n"
        f"<policy>\n{plan_doc}\n</policy>\n\n<rationale>\n{a.rationale}\nCited: {a.policy_citation}\n</rationale>"
    )
    faith = await chat(messages=[{"role": "user", "content": prompt}], model_alias=MODEL, response_model=FaithVerdict, temperature=0.0)
    flagged = (not verbatim_ok) or (not faith.faithful)
    # The generic guard only downgrades to review when grounding/faithfulness fails;
    # it does not otherwise change A's outcome.
    return ("needs_info" if flagged else a.outcome), flagged


# ---- system C: enumerate-and-check ----
def _render_provisions(provs: list[dict]) -> str:
    out = []
    for p in provs:
        clause = f' clause: "{p["clause"]}"' if p.get("clause") else " (no single clause — a coverage gap)"
        out.append(f'- [{p["id"]}] ({p["type"]}) applies_when: {p["applies_when"]} | effect: {p["effect"]} |{clause}')
    return "\n".join(out)


async def system_c(provs: list[dict], plan_name: str, claim: str) -> CompletenessDecision:
    prompt = (
        f"You are a claims adjudicator for {plan_name}. Below is the COMPLETE list of this plan's "
        "provisions. Work in two steps.\n\n"
        f"PROVISIONS:\n{_render_provisions(provs)}\n\n"
        f"<claim>\n{claim}\n</claim>\n\n"
        "STEP 1 — For EVERY provision above, state whether it applies to this claim (applies: true/false) and a brief why. "
        "Do not skip any.\n"
        "STEP 2 — Decide the outcome from the applicable provisions, with these rules:\n"
        "- If any EXCLUSION applies, or a required CONDITION is not met, the claim is rejected (cite that clause).\n"
        "- If a CONDITION's facts can't be determined from the claim (a measurement/count/date is missing), or a coverage GAP applies, return needs_info.\n"
        "- Otherwise, if a coverage provision applies and nothing blocks it, approve.\n"
        "Return all provision verdicts, the outcome, the controlling verbatim clause, and a 1-2 sentence rationale."
    )
    return await chat(messages=[{"role": "user", "content": prompt}], model_alias=MODEL, response_model=CompletenessDecision, temperature=0.0)


async def main() -> int:
    provisions = json.loads((EXP_DIR / "provisions.json").read_text())
    claims = [json.loads(l) for l in (EXP_DIR / "claims.jsonl").read_text().splitlines() if l.strip()]
    plan_docs = {k: (RAW_DIR / v).read_text(encoding="utf-8") for k, v in PLAN_DOC.items()}

    sem = asyncio.Semaphore(4)
    done = 0

    async def run_one(c: dict) -> dict | None:
        # One claim failing must not lose the other results, so each is isolated
        # and a failure returns None (filtered out + counted) rather than
        # aborting the whole gather.
        nonlocal done
        async with sem:
            try:
                plan = c["plan"]
                doc = plan_docs[plan]
                plan_name = provisions[plan]["plan_name"]
                provs = provisions[plan]["provisions"]
                a = await system_a(doc, plan_name, c["raw_input"])
                b_outcome, b_flagged = await system_b(doc, a)
                cc = await system_c(provs, plan_name, c["raw_input"])
                applicable = [v.id for v in cc.provision_verdicts if v.applies]
                row = {
                    "id": c["id"], "is_trap": c["is_trap"], "gold": c["gold_outcome"],
                    "controlling": c["controlling_provision"],
                    "A": a.outcome, "B": b_outcome, "B_flagged": b_flagged, "C": cc.outcome,
                    "C_applicable": applicable,
                    "C_surfaced_controlling": (c["controlling_provision"] in applicable) if c["controlling_provision"] not in ("none", "") else None,
                }
                done += 1
                print(f"[exp] {done}/{len(claims)} {c['id']:<4} A={a.outcome:<10} B={b_outcome:<10} C={cc.outcome}", flush=True)
                return row
            except Exception as e:  # noqa: BLE001
                done += 1
                print(f"[exp] {done}/{len(claims)} {c['id']:<4} FAILED: {type(e).__name__}: {str(e)[:140]}", flush=True)
                return None

    gathered = await asyncio.gather(*(run_one(c) for c in claims))
    results = [r for r in gathered if r is not None]
    n_failed = len(gathered) - len(results)
    if n_failed:
        print(f"[exp] WARNING: {n_failed}/{len(claims)} claims failed and are excluded from scoring", flush=True)

    # ---- score ----
    def acc(rows: list[dict], sys: str) -> float:
        return sum(1 for r in rows if r[sys] == r["gold"]) / len(rows) if rows else 0.0

    traps = [r for r in results if r["is_trap"]]
    ctrls = [r for r in results if not r["is_trap"]]

    def wrong_approvals(rows: list[dict], sys: str) -> int:
        # the dangerous omission error: approved when gold says reject/needs_info
        return sum(1 for r in rows if r[sys] == "approve" and r["gold"] != "approve")

    print("\n=== Completeness experiment ===")
    print(f"claims: {len(results)}  (traps: {len(traps)}, controls: {len(ctrls)})\n")
    print(f"{'metric':<38} {'A (free)':>10} {'B (+faith)':>12} {'C (enum)':>10}")
    print("-" * 74)
    print(f"{'Trap accuracy':<38} {acc(traps,'A'):>9.0%} {acc(traps,'B'):>11.0%} {acc(traps,'C'):>9.0%}")
    print(f"{'Control accuracy':<38} {acc(ctrls,'A'):>9.0%} {acc(ctrls,'B'):>11.0%} {acc(ctrls,'C'):>9.0%}")
    print(f"{'Overall accuracy':<38} {acc(results,'A'):>9.0%} {acc(results,'B'):>11.0%} {acc(results,'C'):>9.0%}")
    print(f"{'WRONG APPROVALS on traps (leakage)':<38} {wrong_approvals(traps,'A'):>10} {wrong_approvals(traps,'B'):>12} {wrong_approvals(traps,'C'):>10}")
    c_surf = [r for r in traps if r["C_surfaced_controlling"] is not None]
    if c_surf:
        n = sum(1 for r in c_surf if r["C_surfaced_controlling"])
        print(f"{'C surfaced the controlling provision':<38} {'':>10} {'':>12} {n}/{len(c_surf)}")
    print()

    # per-claim detail
    print(f"{'id':<5} {'trap':<5} {'gold':<10} {'A':<10} {'B':<10} {'C':<10} controlling")
    for r in sorted(results, key=lambda x: x["id"]):
        mark = lambda s: ("✓" if r[s] == r["gold"] else "✗")  # noqa: E731
        print(f"{r['id']:<5} {('Y' if r['is_trap'] else '-'):<5} {r['gold']:<10} "
              f"{r['A']+' '+mark('A'):<10} {r['B']+' '+mark('B'):<10} {r['C']+' '+mark('C'):<10} {r['controlling']}")

    (EXP_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n[exp] wrote {EXP_DIR / 'results.json'}")
    print("\nVerdict rule: C is real ONLY if it cuts wrong-approvals-on-traps well below A and B "
          "without wrecking control accuracy. If not, kill it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
