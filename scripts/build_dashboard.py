"""Generate the Reserve & Leakage Ledger dashboard — a single, self-contained
HTML file (no server, no build step) that embeds the scored books and recomputes
the two-sided P&L and crossover curve live as you edit the cost coefficients,
drag the auto-resolve threshold, and switch model tier / dataset.

    uv run --project backend python scripts/build_dashboard.py
    open docs/ledger/index.html

Two datasets are embedded when present:
  - realistic  -> book_synth_<tier>.json  (a realistically-distributed book)
  - stress     -> book_<tier>.json        (the adversarial 30-claim trap set)

The JS cost model mirrors backend/app/ledger/cost_model.py exactly (verified by
test_dashboard_js_matches_python), so the browser numbers equal the Python ones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app.ledger import CostModel  # noqa: E402

EXP = ROOT / "data" / "atlas_experiment"
OUT = ROOT / "docs" / "ledger" / "index.html"

TIERS = [("cheap", "gpt-4o-mini"), ("frontier", "gpt-5.2")]
DATASETS = [("realistic", "realistic book", "synth_"), ("stress", "adversarial stress", "")]


def _notes() -> dict[str, str]:
    notes: dict[str, str] = {}
    for fn in ("claims.jsonl", "synth_claims.jsonl"):
        p = EXP / fn
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    notes[r["id"]] = r.get("raw_input", "")
    return notes


def _load() -> tuple[dict, list]:
    notes = _notes()
    books: dict = {}
    present: list = []
    for dkey, dlabel, prefix in DATASETS:
        per_tier: dict = {}
        for tkey, _tl in TIERS:
            f = EXP / f"book_{prefix}{tkey}.json"
            if not f.exists():
                continue
            rows = json.loads(f.read_text())
            per_tier[tkey] = [
                {
                    "id": r["id"], "decision": r["decision"], "gold": r["gold"],
                    "claim_value": r["claim_value"], "confidence": r.get("confidence", 1.0),
                    "note": notes.get(r["id"], "")[:130],
                }
                for r in rows
            ]
        if per_tier:
            books[dkey] = per_tier
            present.append({"key": dkey, "label": f"{dlabel} ({len(next(iter(per_tier.values())))})"})
    return books, present


HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Reserve &amp; Leakage Ledger</title>
<style>
  :root {
    --bg:#0b0f14; --panel:#121821; --line:#1f2935; --ink:#e6edf3; --mut:#8b97a7;
    --green:#3fb950; --red:#f85149; --amber:#d29922; --accent:#58a6ff;
    --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
  .wrap { max-width:1080px; margin:0 auto; padding:28px 20px 60px; }
  h1 { font-size:22px; margin:0 0 2px; letter-spacing:-.2px; }
  .sub { color:var(--mut); font-size:13.5px; margin:0 0 22px; max-width:780px; line-height:1.5; }
  .row { display:flex; gap:16px; flex-wrap:wrap; }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px; }
  .controls { flex:1 1 320px; }
  .controls h3 { margin:0 0 8px; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--mut); }
  .seg { display:inline-flex; border:1px solid var(--line); border-radius:8px; overflow:hidden; margin-bottom:14px; }
  .seg button { background:transparent; color:var(--mut); border:0; padding:8px 13px; font-size:13px; cursor:pointer; }
  .seg button.on { background:var(--accent); color:#06101f; font-weight:600; }
  .coef { display:flex; align-items:center; justify-content:space-between; gap:10px; margin:9px 0; }
  .coef label { font-size:13px; color:var(--ink); }
  .coef .src { font-size:11px; color:var(--mut); display:block; margin-top:1px; max-width:240px; }
  .coef input[type=number] { width:96px; background:#0b1119; color:var(--ink); border:1px solid var(--line);
    border-radius:6px; padding:6px 8px; font-family:var(--mono); font-size:13px; text-align:right; }
  .thr { margin-top:8px; }
  .thr input[type=range] { width:100%; }
  .thrval { font-family:var(--mono); color:var(--accent); }
  .counters { flex:1 1 380px; display:flex; flex-direction:column; gap:14px; }
  .twoup { display:flex; gap:14px; }
  .counter { flex:1; background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; }
  .counter .k { font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--mut); }
  .counter .v { font-family:var(--mono); font-size:26px; font-weight:600; margin-top:6px; }
  .counter .n { font-size:11px; color:var(--mut); margin-top:4px; }
  .saved .v { color:var(--green); } .liab .v { color:var(--red); }
  .net { text-align:center; padding:18px; }
  .net .v { font-size:34px; }
  .net.pos .v { color:var(--green); } .net.neg .v { color:var(--red); }
  .net .n { font-size:12px; }
  .banner { margin-top:14px; border-radius:10px; padding:12px 14px; font-size:13.5px; line-height:1.5; border:1px solid; }
  .banner.pos { background:rgba(63,185,80,.08); border-color:rgba(63,185,80,.35); }
  .banner.neg { background:rgba(248,81,73,.08); border-color:rgba(248,81,73,.35); }
  .chart { margin-top:20px; }
  .chart h3 { margin:0 0 8px; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:var(--mut); }
  .legend { font-size:12px; color:var(--mut); display:flex; gap:16px; margin-bottom:6px; }
  .legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }
  table { width:100%; border-collapse:collapse; margin-top:18px; font-size:12.5px; }
  th,td { text-align:left; padding:7px 8px; border-bottom:1px solid var(--line); }
  th { color:var(--mut); font-weight:500; text-transform:uppercase; font-size:10.5px; letter-spacing:.06em; }
  td.mono { font-family:var(--mono); }
  .pill { font-size:10.5px; padding:2px 7px; border-radius:999px; }
  .pill.leak { background:rgba(210,153,34,.15); color:var(--amber); }
  .pill.deny { background:rgba(248,81,73,.15); color:var(--red); }
  .foot { color:var(--mut); font-size:11.5px; margin-top:26px; line-height:1.5; }
</style></head>
<body><div class="wrap">
  <h1>Reserve &amp; Leakage Ledger</h1>
  <p class="sub">Every claims-AI vendor prices the dollars you lose to wrong <b>approvals</b> (leakage).
  This is the other half of the P&amp;L: the dollars you lose to wrong <b>denials</b>. Pick a model tier and
  claim book, drag the auto-resolve threshold, edit the cost assumptions — the crossover is computed live.</p>

  <div class="row">
    <div class="panel controls">
      <h3>Claim book</h3>
      <div class="seg" id="dsseg"></div>
      <h3>Model tier</h3>
      <div class="seg" id="tierseg"></div>
      <h3>Cost assumptions (editable — price YOUR book)</h3>
      <div id="coefs"></div>
      <div class="thr">
        <div class="coef"><label>Auto-resolve threshold</label><span class="thrval" id="thrlbl"></span></div>
        <input type="range" id="thr" min="0.50" max="1.00" step="0.01" value="0.50">
      </div>
    </div>

    <div class="counters">
      <div class="twoup">
        <div class="counter saved"><div class="k">LAE saved</div><div class="v" id="c_saved"></div><div class="n" id="c_saved_n"></div></div>
        <div class="counter liab"><div class="k">False-denial liability</div><div class="v" id="c_liab"></div><div class="n" id="c_liab_n"></div></div>
      </div>
      <div class="counter net" id="netbox"><div class="k">Net value of automation</div><div class="v" id="c_net"></div><div class="n" id="c_net_n"></div></div>
      <div class="banner" id="banner"></div>
    </div>
  </div>

  <div class="panel chart">
    <h3>Crossover — value vs auto-resolve threshold</h3>
    <div class="legend">
      <span><i style="background:var(--green)"></i>LAE saved</span>
      <span><i style="background:var(--red)"></i>False-denial liability</span>
      <span><i style="background:var(--accent)"></i>Net</span>
      <span><i style="background:var(--mut)"></i>current threshold</span>
    </div>
    <svg id="svg" viewBox="0 0 1000 320" width="100%" height="320"></svg>
  </div>

  <table id="errtbl"><thead><tr><th>flagged at this threshold</th><th>plan/decision</th><th>gold</th><th>$ cost</th><th>type</th></tr></thead><tbody></tbody></table>

  <p class="foot" id="foot"></p>
</div>
<script>
const BOOKS = __BOOKS__;
const DEFAULTS = __DEFAULTS__;
const META = __META__;
const AMBIG_W = __AMBIG__;
let ds = META.datasets[0].key;
let tier = META.tiers[0].key;
let cm = Object.assign({}, DEFAULTS);

const COEF_META = [
  ["review_labor", "Review labor / touch ($)", "fully-loaded LAE per manual claim touch"],
  ["leakage_multiplier", "Leakage multiplier (×)", "fraction of a wrongful payout you eat"],
  ["dispute_ev", "Dispute EV / denial ($)", "appeal + DOI complaint + bad-faith (UCSPA/NAIC)"],
  ["churn_cost", "Churn / denial ($)", "lost LTV per wrongly-denied valid customer"],
];

function fmt(x){ const s = x<0?'-':''; return s+'$'+Math.round(Math.abs(x)).toLocaleString(); }

function priceClaim(d, gold, val){
  const auto = d==='approve'||d==='reject';
  if(!auto) return {auto:false,lae:0,leak:0,deny:0,err:null};
  if(d===gold) return {auto:true,lae:cm.review_labor,leak:0,deny:0,err:null};
  const w = gold==='needs_info'?AMBIG_W:1.0;
  if(d==='approve') return {auto:true,lae:cm.review_labor,leak:val*cm.leakage_multiplier*w,deny:0,err:'leakage'};
  const owed = gold==='approve'?val:0;
  return {auto:true,lae:cm.review_labor,leak:0,deny:(owed+cm.dispute_ev+cm.churn_cost)*w,err:'false_denial'};
}
function rows(){ return BOOKS[ds][tier]; }
function priceBook(thr){
  let lae=0,leak=0,deny=0,auto=0,nl=0,nd=0,flagged=[];
  for(const r of rows()){
    const d = r.confidence>=thr ? r.decision : 'needs_info';
    const c = priceClaim(d, r.gold, r.claim_value);
    lae+=c.lae; leak+=c.leak; deny+=c.deny; if(c.auto) auto++;
    if(c.err==='leakage'){nl++; flagged.push({r,c,d});}
    if(c.err==='false_denial'){nd++; flagged.push({r,c,d});}
  }
  const n=rows().length;
  return {n,lae,leak,deny,auto,nl,nd,flagged,net:lae-leak-deny,
          per1000:v=>n?v/n*1000:0, autorate:n?auto/n:0};
}
function sweep(){ const pts=[]; for(let t=50;t<=100;t+=5){ const thr=t/100;
  let lae=0,leak=0,deny=0; for(const r of rows()){ const d=r.confidence>=thr?r.decision:'needs_info';
    const c=priceClaim(d,r.gold,r.claim_value); lae+=c.lae;leak+=c.leak;deny+=c.deny; }
  pts.push({thr,lae,liab:leak+deny,deny,net:lae-leak-deny}); } return pts; }

function render(){
  const thr = parseFloat(document.getElementById('thr').value);
  document.getElementById('thrlbl').textContent = 'conf ≥ '+thr.toFixed(2);
  const b = priceBook(thr);
  document.getElementById('c_saved').textContent = fmt(b.lae);
  document.getElementById('c_saved_n').textContent = (b.autorate*100).toFixed(0)+'% auto-resolved';
  document.getElementById('c_liab').textContent = fmt(b.deny);
  document.getElementById('c_liab_n').textContent =
    b.nd+' false denials'+(b.nl? ' · +'+fmt(b.leak)+' leakage ('+b.nl+')' : ' · $0 leakage');
  const netbox = document.getElementById('netbox');
  netbox.className = 'counter net '+(b.net>=0?'pos':'neg');
  document.getElementById('c_net').textContent = fmt(b.net);
  document.getElementById('c_net_n').textContent = fmt(b.per1000(b.net))+' per 1,000 claims';

  const banner = document.getElementById('banner');
  if(b.net>=0){ banner.className='banner pos';
    banner.innerHTML = 'At conf ≥ '+thr.toFixed(2)+', automating '+(b.autorate*100).toFixed(0)+
      '% of claims nets <b>'+fmt(b.net)+'</b> ('+fmt(b.per1000(b.net))+' / 1,000). Automation pays.';
  } else { banner.className='banner neg';
    const liabTxt = b.leak>0 ? 'error liability ('+fmt(b.deny)+' false-denial + '+fmt(b.leak)+' leakage)'
                             : 'false-denial liability ('+fmt(b.deny)+')';
    banner.innerHTML = 'At conf ≥ '+thr.toFixed(2)+', automation costs you <b>'+fmt(-b.net)+'</b> ('+
      fmt(b.per1000(b.net))+' / 1,000): '+liabTxt+' outweighs the '+fmt(b.lae)+' of labor saved.';
  }

  const tb = document.querySelector('#errtbl tbody'); tb.innerHTML='';
  b.flagged.sort((x,y)=>(y.c.deny+y.c.leak)-(x.c.deny+x.c.leak));
  for(const f of b.flagged.slice(0,12)){
    const cost=f.c.deny+f.c.leak, type=f.c.err==='leakage'?'leak':'deny';
    tb.innerHTML += '<tr><td>'+f.r.id+' · '+f.r.note.replace(/</g,'&lt;').slice(0,68)+
      '…</td><td class="mono">'+f.d+'</td><td class="mono">'+f.r.gold+'</td><td class="mono">'+fmt(cost)+
      '</td><td><span class="pill '+type+'">'+(f.c.err==='leakage'?'leakage':'false denial')+'</span></td></tr>';
  }
  if(!b.flagged.length) tb.innerHTML='<tr><td colspan="5" style="color:var(--mut)">no auto-resolved errors at this threshold</td></tr>';
  else if(b.flagged.length>12) tb.innerHTML += '<tr><td colspan="5" style="color:var(--mut)">+ '+(b.flagged.length-12)+' more…</td></tr>';

  const sw=sweep(); drawChart(sw, thr);
  const best = sw.reduce((a,c)=>c.net>a.net?c:a);
  document.getElementById('foot').innerHTML =
    'Book: <b>'+META.datasets.find(d=>d.key===ds).label+'</b> · tier <b>'+
    META.tiers.find(t=>t.key===tier).label+'</b> · '+rows().length+' claims. Best net at conf ≥ '+
    best.thr.toFixed(2)+' ('+fmt(best.net)+'). Coefficients are editable assumptions — point this at your real loss runs to price your book.';
}

function drawChart(pts, thr){
  const W=1000,H=320,pad=44;
  const all=pts.flatMap(p=>[p.lae,p.liab,p.net]);
  let lo=Math.min(0,...all), hi=Math.max(0,...all); if(hi===lo) hi=lo+1;
  const X=t=>pad+(t-0.50)/(0.50)*(W-2*pad);
  const Y=v=>H-pad-(v-lo)/(hi-lo)*(H-2*pad);
  const path=(key,col)=>{ let d=''; pts.forEach((p,i)=>{ d+=(i?'L':'M')+X(p.thr).toFixed(1)+' '+Y(p[key]).toFixed(1)+' '; });
    return '<path d="'+d+'" fill="none" stroke="'+col+'" stroke-width="2.5"/>'; };
  let g='';
  g+='<line x1="'+pad+'" y1="'+Y(0).toFixed(1)+'" x2="'+(W-pad)+'" y2="'+Y(0).toFixed(1)+'" stroke="#2a3441" stroke-dasharray="3 3"/>';
  g+='<line x1="'+X(thr).toFixed(1)+'" y1="'+pad+'" x2="'+X(thr).toFixed(1)+'" y2="'+(H-pad)+'" stroke="#8b97a7" stroke-dasharray="4 4"/>';
  for(let t=50;t<=100;t+=10){ g+='<text x="'+X(t/100).toFixed(1)+'" y="'+(H-pad+18)+'" fill="#8b97a7" font-size="11" text-anchor="middle">'+(t/100).toFixed(2)+'</text>'; }
  g+=path('lae','#3fb950')+path('liab','#f85149')+path('net','#58a6ff');
  document.getElementById('svg').innerHTML=g;
}

function seg(id, items, get, set){
  const el=document.getElementById(id);
  items.forEach(it=>{ const b=document.createElement('button'); b.textContent=it.label;
    b.className=it.key===get()?'on':''; b.onclick=()=>{ set(it.key);
      [...el.children].forEach(c=>c.className=''); b.className='on'; render(); }; el.appendChild(b); });
}
seg('dsseg', META.datasets, ()=>ds, k=>ds=k);
seg('tierseg', META.tiers, ()=>tier, k=>tier=k);
const cdiv=document.getElementById('coefs');
COEF_META.forEach(([k,lab,src])=>{ const row=document.createElement('div'); row.className='coef';
  row.innerHTML='<div><label>'+lab+'</label><span class="src">'+src+'</span></div>'+
    '<input type="number" step="'+(k==='leakage_multiplier'?'0.1':'1')+'" value="'+cm[k]+'" id="cf_'+k+'">';
  cdiv.appendChild(row);
  row.querySelector('input').addEventListener('input',e=>{ cm[k]=parseFloat(e.target.value)||0; render(); }); });
document.getElementById('thr').addEventListener('input',render);
render();
</script>
</body></html>
"""


def main() -> int:
    books, datasets = _load()
    if not books:
        print("No book_*.json files — run scripts/score_book.py first.")
        return 1
    cm = CostModel()
    defaults = {"review_labor": cm.review_labor, "leakage_multiplier": cm.leakage_multiplier,
                "dispute_ev": cm.dispute_ev, "churn_cost": cm.churn_cost}
    tiers = [{"key": k, "label": f"{k} ({lbl})"} for k, lbl in TIERS
             if any(k in books[d] for d in books)]
    meta = {"datasets": datasets, "tiers": tiers}
    html = (HTML
            .replace("__BOOKS__", json.dumps(books))
            .replace("__DEFAULTS__", json.dumps(defaults))
            .replace("__META__", json.dumps(meta))
            .replace("__AMBIG__", json.dumps(cm.ambiguous_error_weight)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"[dashboard] wrote {OUT}  ({len(html):,} bytes)")
    print(f"[dashboard] datasets: {[d['label'] for d in datasets]}  tiers: {[t['key'] for t in tiers]}")
    print(f"[dashboard] open {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
