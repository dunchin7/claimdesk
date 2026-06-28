"""Integrity test: the dashboard's in-browser JS cost model must produce the
same P&L as the Python cost model, so the demo numbers are trustworthy.

Runs the JS that's embedded in the generated dashboard via node and compares to
`price_book`. Skips cleanly if node or the generated artifacts aren't present.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from app.ledger import CostModel, ScoredClaim, price_book

ROOT = Path(__file__).resolve().parent.parent.parent
DASH = ROOT / "docs" / "ledger" / "index.html"
# dataset key -> book-file prefix (mirrors build_dashboard.DATASETS)
DS_PREFIX = {"realistic": "synth_", "stress": ""}
BOOK_DIR = ROOT / "data" / "atlas_experiment"

NODE_PARITY = r"""
import fs from 'fs';
const h = fs.readFileSync(process.argv[2],'utf8');
const grab = (n)=>JSON.parse(h.match(new RegExp('const '+n+' = (.*?);\\n'))[1]);
const BOOKS=grab('BOOKS'), cm=grab('DEFAULTS');
const AMBIG=JSON.parse(h.match(/const AMBIG_W = (.*?);/)[1]);
function price(d,gold,val){
  const auto=d==='approve'||d==='reject';
  if(!auto) return [0,0,0];
  if(d===gold) return [cm.review_labor,0,0];
  const w = gold==='needs_info'?AMBIG:1.0;
  if(d==='approve') return [cm.review_labor,val*cm.leakage_multiplier*w,0];
  const owed = gold==='approve'?val:0;
  return [cm.review_labor,0,(owed+cm.dispute_ev+cm.churn_cost)*w];
}
const out={};
for(const ds of Object.keys(BOOKS)){ out[ds]={};
  for(const t of Object.keys(BOOKS[ds])){ let lae=0,leak=0,deny=0;
    for(const r of BOOKS[ds][t]){ const [a,b,c]=price(r.decision,r.gold,r.claim_value); lae+=a;leak+=b;deny+=c; }
    out[ds][t]={lae,leak,deny}; } }
console.log(JSON.stringify(out));
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
@pytest.mark.skipif(not DASH.exists(), reason="dashboard not generated (run build_dashboard.py)")
def test_dashboard_js_matches_python(tmp_path: Path) -> None:
    script = tmp_path / "parity.mjs"
    script.write_text(NODE_PARITY)
    res = subprocess.run(
        ["node", str(script), str(DASH)],
        capture_output=True, text=True, timeout=30,
    )
    assert res.returncode == 0, res.stderr
    js = json.loads(res.stdout)

    cm = CostModel()
    checked = 0
    for ds, tiers in js.items():
        prefix = DS_PREFIX.get(ds, "")
        for tier, vals in tiers.items():
            path = BOOK_DIR / f"book_{prefix}{tier}.json"
            assert path.exists(), f"missing {path}"
            rows = json.loads(path.read_text())
            sc = [ScoredClaim(id=r["id"], decision=r["decision"], gold=r["gold"],
                              claim_value=r["claim_value"], confidence=r["confidence"]) for r in rows]
            b = price_book(sc, cm)
            assert vals["lae"] == pytest.approx(b.lae_saved), (ds, tier)
            assert vals["leak"] == pytest.approx(b.leakage), (ds, tier)
            assert vals["deny"] == pytest.approx(b.false_denial), (ds, tier)
            checked += 1
    assert checked > 0, "no datasets verified"
