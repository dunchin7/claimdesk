"""Build the two-sided claims P&L + crossover sweep.

Prices each model tier's decisions through the cost model — making the
asymmetry visible in dollars (the model rarely leaks; the cheap tier
*false-denies* valid claims) — and sweeps the auto-resolve confidence threshold
to find each tier's crossover: the point where routing more claims to humans
stops being worth the labor it costs.

Prefers the confidence-scored books (data/atlas_experiment/book_<tier>.json,
from score_book.py); falls back to the experiment results (system A) when a
confidence book isn't present.

    uv run --project backend python scripts/build_ledger.py

No LLM calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.ledger import (  # noqa: E402
    CostModel,
    ScoredClaim,
    estimate_claim_value,
    price_book,
    sweep_threshold,
)
from app.ledger.ledger import BookPnL  # noqa: E402

EXP = ROOT / "data" / "atlas_experiment"
TIERS = [("cheap (gpt-4o-mini)", "cheap", "results_gpt4omini.json"),
         ("frontier (gpt-5.2)", "frontier", "results_gpt52.json")]


def _load_book(tier_key: str, fallback_results: str, claims: dict[str, dict], tag: str = "") -> tuple[list[ScoredClaim], str] | None:
    prefix = f"{tag}_" if tag else ""
    book_file = EXP / f"book_{prefix}{tier_key}.json"
    if book_file.exists():
        rows = json.loads(book_file.read_text())
        sc = [
            ScoredClaim(
                id=r["id"], decision=r["decision"], gold=r["gold"],
                claim_value=r.get("claim_value") or estimate_claim_value(claims.get(r["id"], {}).get("raw_input", "")),
                plan=r.get("plan", ""), confidence=r.get("confidence", 1.0),
            )
            for r in rows
        ]
        return sc, "confidence-scored"
    res_file = EXP / fallback_results
    if res_file.exists():
        rows = json.loads(res_file.read_text())
        sc = [
            ScoredClaim(
                id=r["id"], decision=r["A"], gold=r["gold"],
                claim_value=estimate_claim_value(claims.get(r["id"], {}).get("raw_input", "")),
                plan=claims.get(r["id"], {}).get("plan", ""), confidence=1.0,
            )
            for r in rows
        ]
        return sc, "experiment system A (no confidence)"
    return None


def _money(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def _print_compare(books: list[BookPnL]) -> None:
    rows = [
        ("auto-resolve rate", lambda b: f"{b.auto_resolve_rate:.0%}"),
        ("leakage events", lambda b: str(b.n_leakage_events)),
        ("false-denial events", lambda b: str(b.n_false_denial_events)),
        ("LAE saved", lambda b: _money(b.lae_saved)),
        ("leakage $", lambda b: _money(b.leakage)),
        ("false-denial liability $", lambda b: _money(b.false_denial)),
        ("NET", lambda b: _money(b.net)),
        ("NET / 1,000 claims", lambda b: _money(b.per_1000(b.net))),
        ("false-denial / 1,000", lambda b: _money(b.per_1000(b.false_denial))),
    ]
    w0 = 26
    header = f"{'metric (full auto-resolve)':<{w0}}" + "".join(f"{b.label:>26}" for b in books)
    print("\n" + header)
    print("-" * len(header))
    for name, fn in rows:
        print(f"{name:<{w0}}" + "".join(f"{fn(b):>26}" for b in books))


def _print_sweep(label: str, claims: list[ScoredClaim], cm: CostModel) -> dict:
    curve = sweep_threshold(claims, cm)
    best = max(curve, key=lambda p: p.net)
    print(f"\n  crossover sweep — {label}")
    print(f"  {'thresh':>7} {'auto%':>7} {'LAE saved':>11} {'false-denial':>13} {'NET':>10}")
    for p in curve:
        mark = "  <- best NET" if p.threshold == best.threshold else ""
        print(f"  {p.threshold:>7.2f} {p.auto_resolve_rate:>6.0%} {_money(p.lae_saved):>11} "
              f"{_money(p.false_denial):>13} {_money(p.net):>10}{mark}")
    full = curve[0]  # lowest threshold = full auto
    if best.net <= 0:
        verdict = "no threshold makes automation net-positive on this tier — confidence can't separate good from bad"
    elif best.threshold <= full.threshold + 1e-9:
        verdict = "full automation is already optimal"
    else:
        verdict = (f"optimal at conf>={best.threshold:.2f}: auto-resolve {best.auto_resolve_rate:.0%}, "
                   f"net {_money(best.net)} (vs {_money(full.net)} at full auto)")
    print(f"  → {verdict}")
    return {"label": label, "curve": [p.as_dict() for p in curve],
            "best_threshold": best.threshold, "best_net": round(best.net, 2), "verdict": verdict}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="dataset tag: '' = 30-claim stress set; 'synth' = realistic book")
    ap.add_argument("--claims", default="claims.jsonl", help="claim set jsonl (for claim text/values)")
    args = ap.parse_args()
    claims_file = "synth_claims.jsonl" if (args.tag == "synth" and not args.claims) else args.claims
    if args.tag == "synth" and args.claims == "claims.jsonl":
        claims_file = "synth_claims.jsonl"

    claims = {r["id"]: r for r in
              (json.loads(l) for l in (EXP / claims_file).read_text().splitlines() if l.strip())}
    cm = CostModel()

    print("=== Reserve & Leakage Ledger — two-sided claims P&L ===")
    print(f"dataset: {args.tag or 'stress (30 traps)'}  |  book: {len(claims)} gold-labeled claims")
    print("coefficients (editable): "
          f"review_labor=${cm.review_labor:.0f}  leakage×{cm.leakage_multiplier:.1f}  "
          f"dispute_ev=${cm.dispute_ev:.0f}  churn=${cm.churn_cost:.0f}")

    loaded: list[tuple[str, list[ScoredClaim]]] = []
    for label, key, fallback in TIERS:
        got = _load_book(key, fallback, claims, tag=args.tag)
        if got is None:
            print(f"  [skip] no book or results for {key}")
            continue
        sc, src = got
        loaded.append((label, sc))
        print(f"  loaded {label}: {len(sc)} claims ({src})")

    if not loaded:
        print("No tier data — run score_book.py or the experiment first.")
        return 1

    books = [price_book(sc, cm, label=label) for label, sc in loaded]
    _print_compare(books)

    if len(books) == 2:
        gap = books[0].false_denial - books[1].false_denial
        print(f"\n  → {books[0].label} carries {_money(gap)} more false-denial liability than "
              f"{books[1].label}\n    ({_money(books[0].per_1000(gap))} per 1,000 claims) — the side nobody prices.")

    sweeps = [_print_sweep(label, sc, cm) for label, sc in loaded]

    out = {
        "coefficients": {"review_labor": cm.review_labor, "leakage_multiplier": cm.leakage_multiplier,
                         "dispute_ev": cm.dispute_ev, "churn_cost": cm.churn_cost},
        "books": [b.summary() for b in books],
        "sweeps": sweeps,
    }
    ledger_file = EXP / (f"ledger_{args.tag}.json" if args.tag else "ledger.json")
    ledger_file.write_text(json.dumps(out, indent=2))
    print(f"\n[ledger] wrote {ledger_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
