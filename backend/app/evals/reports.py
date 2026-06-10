"""Markdown report generation for eval runs (Week 4)."""

from __future__ import annotations

from pathlib import Path

from app.evals.runners import EvalRunReport


def render_markdown(report: EvalRunReport) -> str:
    lines: list[str] = []
    lines.append(f"# Eval Run · {report.run_id[:8]}")
    lines.append("")
    lines.append(f"- **Timestamp:** {report.timestamp}")
    lines.append(f"- **Claims:** {report.n_claims} (errors: {report.n_errors})")
    lines.append("- **Prompt versions:**")
    for role, prompt in report.prompt_versions.items():
        lines.append(f"  - `{role}` → `{prompt}`")
    lines.append("")
    lines.append("## Headline Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Decision accuracy | **{report.decision_accuracy:.1%}** |")
    lines.append(f"| Citation verbatim rate | {report.citation_verbatim_rate:.1%} |")
    lines.append(
        f"| Citation precision on correct decisions | "
        f"{report.citation_precision_on_correct:.1%} |"
    )
    lines.append(f"| Faithfulness (judged on misses) | {report.faithfulness_rate:.1%} |")
    if report.email_avg_score:
        lines.append(f"| Email avg quality (1–5, sampled) | {report.email_avg_score} |")
    lines.append(f"| Latency p50 / p95 | {report.p50_latency_ms:.0f} ms / {report.p95_latency_ms:.0f} ms |")
    lines.append(f"| Avg cost / claim | ${report.avg_cost_usd:.6f} |")
    lines.append(
        f"| Cost / accurate decision | ${report.cost_per_accurate_decision_usd:.6f} |"
    )
    lines.append("")
    lines.append("## Accuracy by Stratum")
    lines.append("")
    lines.append("| Stratum | n | Accuracy |")
    lines.append("|---|---:|---:|")
    for k, v in report.by_stratum.items():
        lines.append(f"| {k} | {v['n']} | {v['accuracy']:.1%} |")
    lines.append("")
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append("| Expected → Predicted | Count |")
    lines.append("|---|---:|")
    for k, v in report.confusion_matrix.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## Calibration")
    lines.append("")
    lines.append("| Confidence | n | Accuracy |")
    lines.append("|---|---:|---:|")
    for conf, info in report.calibration.items():
        lines.append(f"| {conf} | {info['n']} | {info['accuracy']:.1%} |")
    lines.append("")
    if report.by_route:
        lines.append("## Routing (Week 17)")
        lines.append("")
        lines.append("| Route | n | Share | Accuracy |")
        lines.append("|---|---:|---:|---:|")
        for route, info in report.by_route.items():
            lines.append(
                f"| {route} | {info['n']} | {info['share']:.1%} | {info['accuracy']:.1%} |"
            )
        lines.append("")
        lines.append(
            f"**Auto-resolve rate:** {report.auto_resolve_rate:.1%} · "
            f"**Accuracy on auto-resolved:** {report.accuracy_on_auto_resolved:.1%}"
        )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_report(report: EvalRunReport, output_dir: Path) -> tuple[Path, Path]:
    """Write `<date>_<run_id>.md` and `.json`. Returns the two paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report.timestamp[:10]}_{report.run_id[:8]}"
    md_path = output_dir / f"{stem}.md"
    json_path = output_dir / f"{stem}.json"

    md_path.write_text(render_markdown(report), encoding="utf-8")

    import json as _json
    from dataclasses import asdict

    json_path.write_text(
        _json.dumps(asdict(report), indent=2, default=str), encoding="utf-8"
    )
    return md_path, json_path
