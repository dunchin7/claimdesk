"""Policy X-ray — drop in ANY product-protection policy, get a grounded
coverage map back.

    uv run --project backend python scripts/xray_policy.py <path-or-url> [--name NAME]

`<path-or-url>` can be a local file (.pdf / .txt / .md / .html), an http(s)
URL, or a path to pasted text. The policy is ingested to clean text, normalized
onto the canonical coverage schema, and every cell is grounded against a
verbatim clause from the source (so you can trust the map, not just the model).

This is the foundation of the X-ray: the coverage map + grounding integrity.
The landmine report (gaps / mis-pay traps / contradictions) builds on this and
is added next.

Needs an LLM key (OPENAI_API_KEY; optionally OPENAI_REASONER_MODEL).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.atlas.extract import extract_coverage_profile, verify_profile_grounding  # noqa: E402
from app.atlas.ingest import ingest_policy  # noqa: E402
from app.atlas.schema import PERIL_FIELDS, CoverageProfile  # noqa: E402

_GLYPH = {"covered": "✅", "conditional": "🟠", "excluded": "❌", "not_addressed": "➖"}


def _cell(item) -> str:  # noqa: ANN001
    glyph = _GLYPH.get(item.status, "?")
    detail = f" — {item.detail}" if item.detail else ""
    return f"{glyph} {item.status}{detail}"


def _render(profile: CoverageProfile, grounded: int, total: int, ungrounded: list[str]) -> str:
    out: list[str] = []
    out.append("")
    out.append(f"  POLICY X-RAY · {profile.plan_name}")
    out.append(f"  source: {profile.source}")
    pct = (grounded / total * 100) if total else 0.0
    out.append(f"  grounding: {grounded}/{total} cited cells verified verbatim ({pct:.0f}%)")
    if ungrounded:
        out.append(f"  ⚠ ungrounded cells (clause not found in source): {', '.join(ungrounded)}")
    out.append("")
    out.append("  COVERAGE")
    for name in PERIL_FIELDS:
        out.append(f"    {name.replace('_', ' '):<20} {_cell(getattr(profile, name))}")
    out.append("")
    out.append("  TERMS")
    for term in ("term_length", "deductible_or_fee", "claim_limit", "transferable"):
        out.append(f"    {term.replace('_', ' '):<20} {_cell(getattr(profile, term))}")
    if profile.exclusions:
        out.append("")
        out.append("  KEY EXCLUSIONS")
        for ex in profile.exclusions[:8]:
            line = ex.detail or ex.clause
            out.append(f"    ❌ {line[:100]}")
    if profile.resolution_types:
        out.append("")
        out.append(f"  RESOLUTION: {', '.join(profile.resolution_types)}")
    out.append("")
    return "\n".join(out)


async def main() -> int:
    ap = argparse.ArgumentParser(description="X-ray any product-protection policy.")
    ap.add_argument("source", help="file path (.pdf/.txt/.md/.html), http(s) URL, or text file")
    ap.add_argument("--name", default="", help="override the plan name")
    args = ap.parse_args()

    print(f"[xray] ingesting {args.source} ...", flush=True)
    policy = ingest_policy(args.source)
    print(f"[xray] ingested {policy.fmt}, {len(policy.text):,} chars — '{policy.title}'", flush=True)

    plan_name = args.name or policy.title
    print(f"[xray] extracting coverage for '{plan_name}' ...", flush=True)
    profile = await extract_coverage_profile(plan_name, policy.source, policy.text)
    grounded, total, ungrounded = verify_profile_grounding(profile, policy.text)

    print(_render(profile, grounded, total, ungrounded))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
