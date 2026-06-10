# Self-Consistency Audit

Sample: **40 claims** (8/stratum × 5 strata) · **3 runs** each · **T=0.3**

## Headline

| Metric | Count | Rate |
|---|---:|---:|
| Stable (all 3 runs agree) | 40 / 40 | 100% |
| Stable AND unanimously correct | 35 / 40 | 88% |
| Stable but unanimously WRONG | 5 / 40 | 12% |
| Flippy (≥1 run disagreed) | 0 / 40 | 0% |
| Majority-correct (≥2 of 3 right) | 35 / 40 | 88% |

## By Stratum

| Stratum | n | Stable | Stable + correct |
|---|---:|---:|---:|
| clear_approve | 8 | 8 (100%) | 8 (100%) |
| clear_reject | 8 | 8 (100%) | 8 (100%) |
| fraud_suspect | 8 | 8 (100%) | 7 (88%) |
| gray | 8 | 8 (100%) | 4 (50%) |
| needs_info | 8 | 8 (100%) | 8 (100%) |

## Flippy claims (≥1 run disagreed)

_None — model is fully stable on this sample._

## Interpretation rule

- **High stability + low accuracy** → model reasons consistently but the *prompt or data* is the bottleneck. A heavier model probably won't help on these claims. Iterate on the prompt or labels.
- **Low stability** → model is making decisions on weak signal. A heavier model (gpt-4o, Claude-Sonnet, etc.) is a meaningful upgrade candidate.
- **High stability + high accuracy** → at the ceiling. Don't spend on a heavier model; spend on real-claim data or new features.

**Verdict: at the ceiling for gpt-4o-mini.** Stability is 100%; the residual errors are stable wrong-answers, which means a heavier model won't recover them without different priors. Spend on prompts, labels, and real data, not on the model.

