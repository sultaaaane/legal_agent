"""
Report writer node.

Assembles all analysis results into a final structured markdown report.
This is a pure assembly node — no LLM calls except the executive summary.
"""
from langchain_core.messages import SystemMessage, HumanMessage

from src.graph.state import ContractState
from src.utils.llm import llm_fast
from src.utils.prompts import REPORT_EXECUTIVE_SUMMARY
from src.agents.flag_detector import CONTRACT_FLAGS_KEY


def report_writer_node(state: ContractState) -> dict:
    """
    Build the final markdown report from all analysis data in state.
    Sections:
      1. Header + metadata
      2. Executive summary (LLM-generated)
      3. Critical cross-clause flags
      4. Clause-by-clause analysis (sorted by risk score, high→low)
         - Each clause: risk score, flags, plain English, reasoning
         - Priority clauses also show: suggested edit + negotiation note
    """
    analyses      = state.get("analyses", {})
    clauses       = state.get("clauses", [])
    contract_type = state.get("contract_type", "Unknown Contract")
    clause_map    = {c["clause_id"]: c for c in clauses}

    # Separate contract-level flags from clause-level analyses
    contract_flags = analyses.get(CONTRACT_FLAGS_KEY, {})
    clause_analyses = {
        k: v for k, v in analyses.items()
        if k != CONTRACT_FLAGS_KEY
    }

    # Sort clauses by risk score, highest first
    sorted_analyses = sorted(
        clause_analyses.values(),
        key=lambda x: x.get("risk_score", 0),
        reverse=True,
    )

    # Count risk levels
    n_high   = sum(1 for a in clause_analyses.values() if a.get("risk_score", 0) >= 7)
    n_medium = sum(1 for a in clause_analyses.values() if 4 <= a.get("risk_score", 0) <= 6)
    n_low    = sum(1 for a in clause_analyses.values() if a.get("risk_score", 0) <= 3)
    n_priority = len(state.get("priority_clauses", []))

    # Generate executive summary via LLM
    summary_context = (
        f"Contract type: {contract_type}\n"
        f"Total clauses: {len(clauses)}\n"
        f"High risk (7–10): {n_high}\n"
        f"Medium risk (4–6): {n_medium}\n"
        f"Low risk (1–3): {n_low}\n"
        f"Critical cross-clause flags: {len(contract_flags.get('flags', []))}\n\n"
        f"Top risks:\n" + "\n".join(
            f"- {a['clause_id']} ({clause_map.get(a['clause_id'], {}).get('clause_type','?')}): "
            f"score {a.get('risk_score',0)}/10 — {', '.join(a.get('flags',[]))}"
            for a in sorted_analyses[:5]
        )
    )

    try:
        summary_response = llm_fast.invoke([
            SystemMessage(REPORT_EXECUTIVE_SUMMARY),
            HumanMessage(summary_context),
        ])
        executive_summary = summary_response.content.strip()
    except Exception:
        executive_summary = (
            f"This {contract_type} contains {len(clauses)} clauses. "
            f"{n_high} clauses are high risk and should be reviewed before signing."
        )

    # ---------------------------------------------------------------------------
    # Build the report
    # ---------------------------------------------------------------------------
    lines = []

    # Header
    lines += [
        f"# Contract Review Report",
        f"",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Contract type | {contract_type} |",
        f"| Total clauses | {len(clauses)} |",
        f"| 🔴 High risk (7–10) | {n_high} |",
        f"| 🟡 Medium risk (4–6) | {n_medium} |",
        f"| 🟢 Low risk (1–3) | {n_low} |",
        f"| Clauses negotiated | {n_priority} |",
        f"",
    ]

    # Executive summary
    lines += [
        f"## Executive Summary",
        f"",
        executive_summary,
        f"",
    ]

    # Critical cross-clause flags
    if contract_flags.get("flags"):
        lines += [
            f"## ⚠️ Critical Cross-Clause Issues",
            f"",
            f"> These risks only appear when reading the contract as a whole.",
            f"",
        ]
        for flag in contract_flags["flags"]:
            lines.append(f"- {flag}")
        if contract_flags.get("risk_reasoning"):
            lines += ["", contract_flags["risk_reasoning"]]
        lines.append("")

    # Clause-by-clause analysis
    lines += [
        f"## Clause-by-Clause Analysis",
        f"",
        f"Clauses are ordered by risk score (highest first).",
        f"",
    ]

    for analysis in sorted_analyses:
        cid    = analysis.get("clause_id", "unknown")
        clause = clause_map.get(cid, {})
        score  = analysis.get("risk_score", 0)
        flags  = analysis.get("flags", [])

        # Risk emoji
        if score >= 7:
            emoji = "🔴"
        elif score >= 4:
            emoji = "🟡"
        else:
            emoji = "🟢"

        clause_type = clause.get("clause_type", "unknown").replace("_", " ").title()

        lines += [
            f"---",
            f"",
            f"### {emoji} {cid} — {clause_type}",
            f"",
            f"**Risk score:** {score}/10  ",
        ]

        if flags:
            flag_badges = " · ".join(f"`{f}`" for f in flags)
            lines.append(f"**Flags:** {flag_badges}  ")

        lines.append("")

        # Plain English
        if analysis.get("plain_english"):
            lines += [
                f"**Plain English**",
                f"",
                f"{analysis['plain_english']}",
                f"",
            ]

        # Risk reasoning
        if analysis.get("risk_reasoning"):
            lines += [
                f"**Why it's risky**",
                f"",
                f"{analysis['risk_reasoning']}",
                f"",
            ]

        # Original clause (collapsed)
        if clause.get("original"):
            lines += [
                f"<details>",
                f"<summary>Original clause text</summary>",
                f"",
                f"```",
                clause["original"],
                f"```",
                f"",
                f"</details>",
                f"",
            ]

        # Negotiation suggestion (priority clauses only)
        if analysis.get("suggested_edit"):
            lines += [
                f"**Suggested Edit**",
                f"",
                f"{analysis['suggested_edit']}",
                f"",
            ]

    # Footer
    lines += [
        f"---",
        f"",
        f"*Report generated by Legal Contract Reviewer · "
        f"This report is for informational purposes only and does not constitute legal advice. "
        f"Always consult a qualified attorney before signing any contract.*",
    ]

    report = "\n".join(lines)

    return {
        "report":        report,
        "current_phase": "done",
    }
