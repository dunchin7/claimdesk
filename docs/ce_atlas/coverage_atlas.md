# Consumer-Electronics Protection — Coverage Atlas

A clause-cited, normalized comparison of the major device-protection plans, plus an engine that adjudicates a claim against the **real** terms of each. Built to answer a question nobody maintains a clean answer to: *what do these plans actually cover, where do they differ, and how confidently can a claim be decided against them?*

> **Method.** Each plan's published terms are normalized onto one canonical schema (`backend/app/atlas/schema.py`). Every cell is **grounded** — it links to a verbatim clause from the source, or is explicitly marked `(plan summary)` when the master contract defers the detail to a per-device schedule. The ClaimDesk engine then adjudicates sample claims against the real terms, citing the governing clause. Quotes are short, fair-use excerpts for analysis; full terms live with the publishers (linked).
>
> **Not legal or purchasing advice.** Terms vary by device, region, and date; always read the current contract.

## Plans compared

| Plan | Source |
|---|---|
| **AppleCare+ for iPhone** (single-pay, North America terms) — *manufacturer* | [apple.com/legal · AppleCare+ T&C](https://www.apple.com/legal/sales-support/applecare/applecareplus/docs/applecareplusnaen.html) · [fees](https://www.apple.com/legal/applecare/fees-deductibles/) |
| **Samsung Care+** (US service contract) — *manufacturer* | [samsung.com/us · Service Contract T&C](https://www.samsung.com/us/support/samsung-care-plus/service-contract-terms-and-conditions/) |
| **Allstate Protection Plan / SquareTrade** — *third-party (sold at retail)* | [squaretrade.com · Coverage](https://www.squaretrade.com/coverage/) · [Terms](https://www.squaretrade.com/terms/) |

## Coverage matrix (perils)

| Peril | AppleCare+ | Samsung Care+ | Allstate Protection (SquareTrade) |
|---|---|---|---|
| **Mechanical breakdown / defect** | ✅ no fee — *"repair the defect at no charge…"* (§3.1) | ✅ — *"a mechanical or electrical failure or other defect"* | ✅ — *"Standard Plans cover product breakdowns and malfunctions during normal use"* |
| **Accidental damage (ADH)** | 🟠 w/ fee — *"accidental damage from handling… unexpected and unintentional external event"* (§3.2) | 🟠 w/ fee (ADH tier) — *"a failure due to accidental damage from handling…"* | 🟠 Accident Plan only — *"life's accidents like drops and spills"* |
| **Liquid damage** | ✅ within ADH — *"drops and damage caused by liquid contact"* (§3.2) | ✅ within ADH *(summary)* — note: *"leaking Product battery… is excluded"* | ✅ within Accident Plan — *"drops and spills"* + immersion *(summary)* |
| **Screen / back glass** | 🟠 **$29** screen / $99 other — *"Screen-Only Damage: US$29; All Other Damage: US$99"* (§3.2) | 🟠 **$29** screen *(summary)* | 🟠 no screen tier; flat **$149** phone deductible *(summary)* |
| **Battery (degradation)** | ✅ no fee at **<80%** — *"less than eighty percent (80%)"* (§3.1) | ❌ leak excluded; degradation **not addressed** | ➖ **not addressed** on coverage terms |
| **Theft & loss** | 🟠 Theft & Loss tier only — *"replace Covered Equipment that is lost or stolen"* (§4.1c) | 🟠 Theft & Loss tier only — *"lost, stolen…"* | ❌ **excluded** — *"Neither plan covers intentional damage, loss, theft, or commercial use"* |
| **Cosmetic damage** | ❌ excluded (normal wear) | ❌ — *"Cosmetic damage… scratches, dents, or housing cracks"* | ❌ excluded (cosmetic / normal wear) |
| **Intentional damage** | ❌ — *"reckless, abusive, willful or intentional conduct"* (§4.1) | ❌ — *"Any intentional dishonest, fraudulent or criminal act"* | ❌ — *"intentional damage… or commercial use"* |

✅ covered · 🟠 conditional (fee/tier/limit) · ❌ excluded · ➖ not addressed

## Commercial terms

| Term | AppleCare+ | Samsung Care+ | Allstate Protection |
|---|---|---|---|
| **Coverage length** | 2 yrs single-pay — *"your Plan Term is two (2) years"* (§2) | per Coverage Confirmation | per plan confirmation; **must buy within 30 days** of item purchase |
| **Service fee / deductible** | $0 defect · $29 screen · $99 other (§3.2) | $0 defect · $29 screen · **$99–$199** ADH *(summary)* | **$149 flat** (phones); 100% parts & labor, deductible may apply *(summary)* |
| **Claim limit** | ⚠️ master terms say **2 service events** (§3.2) — but Apple's current summaries say *unlimited* | **3 ADH claims / 12 mo** *(summary)* | up to item purchase price; ➖ count not stated on terms |
| **Resolution** | repair or exchange | repair or replace | repair, or **full item price / cash settlement** if unrepairable |

## What this surfaces that a list-of-plans wouldn't

- **The fee model is the real differentiator, not the coverage list.** All three cover drops and spills; what a claimant *feels* is AppleCare+'s flat $29/$99 vs Samsung's $99–$199 ADH deductible vs Allstate's $149 flat — wildly different economics for the same cracked screen.
- **Manufacturer vs third-party split on theft.** Apple and Samsung gate theft/loss behind a separate tier; Allstate **flatly excludes** it. Same word, three different meanings — exactly the nuance a generic "do they cover theft?" comparison gets wrong.
- **Battery is an asymmetry.** AppleCare+ names an explicit 80% capacity threshold; Samsung excludes *leakage* and is silent on degradation; Allstate's coverage terms don't address battery at all. Three different coverage gaps at quote time.
- **A live discrepancy inside AppleCare+'s own documents:** the master terms state a **2-service-event** ADH cap while Apple's current summaries say **unlimited**. The version/source drift that bites a TPA onboarding a program — caught by clause-grounding, invisible to a brochure comparison.

## Adjudication on the real terms

The ClaimDesk engine decides a claim against the actual plan text, citing the governing clause and routing by confidence. Run it live with `scripts/adjudicate_ce_demo.py` (a matrix of real claims × all three plans) or rebuild the normalized data with `scripts/build_ce_atlas.py`. Illustrative:

> **Claim:** *"Dropped my iPhone, screen cracked, otherwise fine. Have AppleCare+, no prior claims."*
> **→ AppleCare+:** `approve` · resolution `repair` · cited clause §3.2 *"Screen-Only Damage: US$29"* · high confidence → **auto-resolve** (unambiguous fee tier).

> **Claim:** *"My Samsung phone battery swelled and won't hold charge after 14 months."*
> **→ Samsung Care+:** `needs_info / review` · the master terms exclude battery *leakage* but don't address *degradation*; cause (defect vs wear) is ambiguous → **route to human** (don't auto-pay where the clause doesn't clearly cover it — leakage control).

That second case is the point: the engine **doesn't auto-pay where the policy is silent or ambiguous** — it cites what it can and routes the rest to a person. Coverage gaps become review items, not leakage.

---

*Generated by the ClaimDesk Coverage Atlas — three plans across the manufacturer / third-party split. Next to add: Asurion carrier device protection, Amazon product protection, manufacturer base warranties — each normalized onto the same schema (`backend/app/atlas/schema.py`).*
