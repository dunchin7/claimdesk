"""Generate a realistically-distributed CE claim book with gold labels.

The 30-claim trap set is deliberately adversarial (67% traps) — great for
stress-testing, useless for a credible per-1,000 figure. This builds a book
that looks like a real protection-claims mix instead: mostly clean covered
claims, with reject / needs_info / trap minorities. Every claim's gold label is
known **by construction** (the scenario template defines the correct outcome),
so model decisions can be scored against ground truth without an LLM oracle.

    uv run --project backend python scripts/generate_ce_book.py [--n 200] [--seed 7]

Writes data/atlas_experiment/synth_claims.jsonl (same schema as claims.jsonl),
consumable by score_book.py --claims ... .
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.ledger import estimate_claim_value  # noqa: E402

EXP = ROOT / "data" / "atlas_experiment"

DEVICES = {
    "applecare": ["iPhone"],
    "samsung": ["Samsung Galaxy phone", "Galaxy S23", "Samsung phone"],
    "allstate": ["laptop", "iPad", "tablet", "pair of headphones", "Bluetooth speaker", "phone"],
}

# Each scenario: gold outcome, controlling provision, which plans it applies to,
# a relative frequency weight (realistic book), and several phrasings.
# Slots: {dev} device, {m} a month count.
SCENARIOS = [
    # ---- covered / approve (the bulk of a real book) ----
    {"key": "adh_screen", "gold": "approve", "weight": 22,
     "prov": {"applecare": "AC-COV-adh", "samsung": "SS-COV-adh", "allstate": "AL-COV-adh"},
     "plans": ["applecare", "samsung", "allstate"], "m": (1, 18),
     "tpl": [
        "I dropped my {dev} and the screen cracked. Everything else works. It's my first claim, bought the plan with it {m} months ago.",
        "Cracked the screen on my {dev} when it slipped out of my pocket onto the pavement. {m} months old, first claim.",
        "My {dev} screen shattered after a drop last week — otherwise fine. Had the plan since purchase {m} months back.",
     ]},
    {"key": "breakdown", "gold": "approve", "weight": 14,
     "prov": {"applecare": "AC-COV-defect", "samsung": "SS-COV-mech", "allstate": "AL-COV-breakdown"},
     "plans": ["applecare", "samsung", "allstate"], "m": (3, 20),
     "tpl": [
        "My {dev} just stopped working on its own — no drops, no spills. It's {m} months old and I have the plan.",
        "The {dev} won't power on anymore, totally spontaneous, no damage. {m} months old, covered.",
        "My {dev}'s speaker died and the display started glitching by itself. No accidents. {m} months in, I have coverage.",
     ]},
    {"key": "liquid_covered", "gold": "approve", "weight": 12,
     "prov": {"applecare": "AC-COV-adh", "samsung": "SS-COV-adh", "allstate": "AL-COV-adh"},
     "plans": ["applecare", "samsung", "allstate"], "m": (1, 16),
     "tpl": [
        "I spilled water on my {dev} and now it won't turn on. {m} months old, first claim, I have the accident coverage.",
        "Knocked a glass of juice onto my {dev} — it's dead now. Bought the accident plan with it {m} months ago.",
        "My {dev} got splashed and stopped working. {m} months old, I have the plan with accidental coverage.",
     ]},
    {"key": "battery_measured", "gold": "approve", "weight": 5,
     "prov": {"applecare": "AC-COV-battery"}, "plans": ["applecare"], "m": (10, 22),
     "tpl": [
        "My iPhone's battery health shows 73% and it dies by mid-afternoon. {m} months old, AppleCare+.",
        "Battery health is down to 68% on my iPhone after {m} months. AppleCare+, can you replace it?",
        "iPhone battery is at 76% capacity now and barely lasts the day — {m} months old, I have AppleCare+.",
     ]},
    {"key": "adh_other", "gold": "approve", "weight": 5,
     "prov": {"applecare": "AC-COV-adh", "samsung": "SS-COV-adh", "allstate": "AL-COV-adh"},
     "plans": ["applecare", "allstate"], "m": (2, 18),
     "tpl": [
        "I dropped my {dev} and the back is damaged plus a button stopped working. {m} months old, first claim, accident plan.",
        "My {dev} took a fall and now the camera glass is broken and it won't focus. {m} months old, covered for accidents.",
     ]},
    # ---- clear rejects ----
    {"key": "cosmetic", "gold": "reject", "weight": 5,
     "prov": {"applecare": "AC-EXC-cosmetic", "samsung": "SS-EXC-cosmetic", "allstate": "AL-EXC-cosmetic"},
     "plans": ["applecare", "samsung", "allstate"], "m": (2, 18),
     "tpl": [
        "There's a scratch and a small dent on my {dev}. It works completely fine, I just want it looking new. I have the plan.",
        "My {dev} has some scuffs on the casing — functions perfectly, I'd just like it cleaned up cosmetically. Covered?",
     ]},
    {"key": "theft_base", "gold": "reject", "weight": 4,
     "prov": {"applecare": "AC-EXC-theft", "samsung": "SS-EXC-theft", "allstate": "AL-EXC-theft"},
     "plans": ["applecare", "samsung", "allstate"], "m": (1, 20),
     "tpl": [
        "My {dev} was stolen out of my bag. I have the regular plan, not the theft add-on. How do I get a replacement?",
        "Someone took my {dev} at a cafe. I just have the standard plan. What now?",
     ]},
    {"key": "out_of_term", "gold": "reject", "weight": 3,
     "prov": {"applecare": "AC-LIM-term"}, "plans": ["applecare"], "m": (27, 34),
     "tpl": [
        "My iPhone won't power on. I've had it {m} months and bought AppleCare+ with it. Can you repair it?",
        "iPhone stopped working after {m} months. I had AppleCare+ from the start — still good?",
     ]},
    {"key": "intentional", "gold": "reject", "weight": 2,
     "prov": {"applecare": "AC-EXC-intentional", "samsung": "SS-EXC-intentional", "allstate": "AL-EXC-intentional"},
     "plans": ["applecare", "samsung", "allstate"], "m": (1, 16),
     "tpl": [
        "I got frustrated and smashed my {dev} on the desk on purpose, now the screen's gone. I have the accident plan.",
        "Honestly I threw my {dev} in anger and it broke. {m} months old, I have coverage — is that covered?",
     ]},
    {"key": "commercial", "gold": "reject", "weight": 3,
     "prov": {"allstate": "AL-EXC-commercial"}, "plans": ["allstate"], "m": (1, 16),
     "tpl": [
        "One of the {dev}s we use to run our shop registers stopped working. We have Allstate plans on the store devices.",
        "My {dev} broke during a delivery run — I use it all day for my courier business. I have the Allstate plan.",
     ]},
    {"key": "eligibility", "gold": "reject", "weight": 3,
     "prov": {"allstate": "AL-COND-30day"}, "plans": ["allstate"], "m": (1, 12),
     "tpl": [
        "I dropped my {dev} and cracked it. I added the Allstate plan about two months after I bought the device. Covered?",
        "Bought a refurbished {dev} and added the Allstate accident plan at checkout, then cracked the screen. Can you fix it?",
     ]},
    {"key": "limit", "gold": "reject", "weight": 2,
     "prov": {"samsung": "SS-LIM-adh3"}, "plans": ["samsung"], "m": (8, 12),
     "tpl": [
        "I've already had three accidental repairs on my {dev} this year and it slipped again — screen cracked. Samsung Care+.",
        "This is the fourth screen I've cracked on my {dev} in 12 months. Samsung Care+, can you sort another repair?",
     ]},
    # ---- needs_info ----
    {"key": "insufficient", "gold": "needs_info", "weight": 6,
     "prov": {"applecare": "none", "samsung": "none", "allstate": "none"},
     "plans": ["applecare", "samsung", "allstate"], "m": (1, 20),
     "tpl": [
        "Hi, my {dev} is broken and I need help. I have a plan.",
        "Something's wrong with my {dev}. Can you help? I'm covered.",
        "My {dev} isn't working right. I have the protection plan.",
     ]},
    {"key": "battery_nomeasure", "gold": "needs_info", "weight": 3,
     "prov": {"applecare": "AC-COV-battery"}, "plans": ["applecare"], "m": (12, 22),
     "tpl": [
        "My iPhone battery drains way faster than it used to, can't get through the afternoon. {m} months old, AppleCare+. I haven't checked the exact battery health number.",
        "iPhone battery seems weak lately after {m} months. AppleCare+. Not sure what the health percentage is.",
     ]},
    {"key": "wear_ambiguous", "gold": "needs_info", "weight": 2,
     "prov": {"applecare": "AC-EXC-wear"}, "plans": ["applecare"], "m": (18, 23),
     "tpl": [
        "The charging port on my iPhone has gotten loose after about {m} months — sometimes the cable won't connect unless I wiggle it. AppleCare+, still in coverage.",
        "After {m} months my iPhone's charging port is finicky, cable only works at an angle. AppleCare+, within term.",
     ]},
    {"key": "degradation_gap", "gold": "needs_info", "weight": 2,
     "prov": {"samsung": "SS-GAP-degradation"}, "plans": ["samsung"], "m": (10, 16),
     "tpl": [
        "My {dev}'s battery life has gotten noticeably worse over the last year — no damage, nothing spilled, it just doesn't last. Samsung Care+.",
        "Over {m} months my {dev} battery just degraded, no accident, no defect I can point to. Samsung Care+ please.",
     ]},
    # ---- traps (hidden disqualifier) ----
    {"key": "battery_leak", "gold": "reject", "weight": 3,
     "prov": {"samsung": "SS-EXC-batteryleak"}, "plans": ["samsung"], "m": (6, 14),
     "tpl": [
        "My {dev}'s battery has swollen up and there's crusty residue near it, barely holds a charge now. {m} months old, Samsung Care+.",
        "The battery in my {dev} is bulging and leaking a bit. {m} months old, I have Samsung Care+.",
     ]},
    {"key": "liquid_is_leak", "gold": "reject", "weight": 2,
     "prov": {"samsung": "SS-EXC-batteryleak"}, "plans": ["samsung"], "m": (6, 16),
     "tpl": [
        "There's liquid corrosion inside my {dev} and a repair shop said the battery itself leaked internally. Samsung Care+.",
        "My {dev} has internal corrosion — the shop says it's from the battery leaking. {m} months old, Samsung Care+.",
     ]},
    {"key": "loss_base", "gold": "reject", "weight": 1,
     "prov": {"applecare": "AC-EXC-theft"}, "plans": ["applecare"], "m": (1, 20),
     "tpl": [
        "I left my iPhone in a taxi and never got it back. I have AppleCare+ — can you send a replacement?",
        "Lost my iPhone somewhere and can't find it. AppleCare+, can I get a new one?",
     ]},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    weighted = [(s, p) for s in SCENARIOS for p in s["plans"]]
    weights = [s["weight"] for s, _p in weighted]

    rows = []
    counts: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for i in range(1, args.n + 1):
        scen, plan = rng.choices(weighted, weights=weights, k=1)[0]
        dev = rng.choice(DEVICES[plan])
        m = rng.randint(*scen["m"])
        text = rng.choice(scen["tpl"]).format(dev=dev, m=m)
        rows.append({
            "id": f"S{i:04d}",
            "plan": plan,
            "is_trap": scen["key"] in ("battery_leak", "liquid_is_leak", "loss_base"),
            "trap_type": scen["key"],
            "gold_outcome": scen["gold"],
            "controlling_provision": scen["prov"].get(plan, "none"),
            "raw_input": text,
            "claim_value": estimate_claim_value(text),
        })
        counts[scen["key"]] = counts.get(scen["key"], 0) + 1
        gold_counts[scen["gold"]] = gold_counts.get(scen["gold"], 0) + 1

    out = EXP / "synth_claims.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    print(f"[gen] wrote {out} — {len(rows)} claims (seed {args.seed})")
    print("[gen] gold distribution:")
    for g, c in sorted(gold_counts.items()):
        print(f"        {g:<11} {c:>4}  ({c/len(rows):.0%})")
    print("[gen] by scenario:")
    for k, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"        {k:<20} {c:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
