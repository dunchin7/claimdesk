"""Eval framework schemas (Week 4).

`GoldenClaim` is the source-of-truth claim+expected-decision record;
`EvalRun` records a single eval execution with metrics and per-claim results.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class GoldenClaim(BaseModel):
    claim_id: str
    customer_text: str
    photo_descriptions: list[str] = []
    sku: str
    purchase_date: datetime
    expected_decision: str          # approve / reject / needs_info
    expected_citation: str          # exact policy quote justifying the decision
    expected_resolution: str        # refund / replacement / repair / store_credit
    is_fraud: bool
    notes: str = ""
    tags: list[str] = []


class EvalRun(BaseModel):
    run_id: str
    timestamp: datetime
    prompt_version: str
    retrieval_version: str
    model_aliases: dict[str, str]
    metrics: dict[str, float]       # accuracy, precision, recall, etc.
    per_claim_results: list[dict[str, Any]]
