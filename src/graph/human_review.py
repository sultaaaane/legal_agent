"""
Human review node — the interrupt point.

Execution pauses here. The caller sees a risk summary and selects
which clauses to negotiate. On resume, priority_clauses is written
to state and routing continues to the negotiator.
"""
import ast
from langgraph.types import interrupt

from src.graph.state import ContractState
from src.agents.flag_detector import CONTRACT_FLAGS_KEY


def human_review_node(state: ContractState) -> dict:
    """
    1. Builds a markdown risk summary from current analyses.
    2. Calls interrupt() — execution pauses, summary sent to caller.
    3. On resume, receives user's selected clause IDs.
    4. Marks those clauses as priority in analyses.
    5. Returns updated state to continue to negotiator.
    """
    analyses = state.get("analyses", {})

    # Sort clause analyses by risk score (highest first)
    clause_analyses = sorted(
        [v for k, v in analyses.items() if k != CONTRACT_FLAGS_KEY],
        key=lambda x: x.get("risk_score", 0),
        reverse=True,
    )

    # Build the risk summary markdown
    lines = [
        f"## Risk Summary — {state.get('contract_type', 'Contract')}",
        f"",
        f"Review the clauses below and select which ones you want negotiated.",
        f"",
    ]

    # Cross-clause flags first
    contract_flags = analyses.get(CONTRACT_FLAGS_KEY, {})
    if contract_flags.get("flags"):
        lines += [
            "### ⚠️ Cross-Clause Issues",
            "",
        ]
        for flag in contract_flags["flags"]:
            lines.append(f"- {flag}")
        lines.append("")

    # Per-clause table
    lines += [
        "### Clause Risk Scores",
        "",
        "| Clause ID | Type | Risk | Flags |",
        "|-----------|------|------|-------|",
    ]

    for a in clause_analyses:
        cid   = a.get("clause_id", "?")
        score = a.get("risk_score", 0)
        flags = ", ".join(a.get("flags", [])) or "—"
        emoji = "🔴" if score >= 7 else "🟡" if score >= 4 else "🟢"

        # Look up clause type from state
        clause_type = "unknown"
        for c in state.get("clauses", []):
            if c["clause_id"] == cid:
                clause_type = c["clause_type"].replace("_", " ")
                break

        lines.append(f"| `{cid}` | {clause_type} | {emoji} {score}/10 | {flags} |")

    lines += ["", "### Plain English Summaries", ""]

    for a in clause_analyses:
        if a.get("plain_english"):
            cid   = a.get("clause_id", "?")
            score = a.get("risk_score", 0)
            emoji = "🔴" if score >= 7 else "🟡" if score >= 4 else "🟢"
            lines += [
                f"**{emoji} {cid}** — {a.get('plain_english', '')}",
                "",
            ]

    summary = "\n".join(lines)

    # Build the payload sent to the caller when interrupted
    interrupt_payload = {
        "summary":     summary,
        "instruction": (
            "Enter the clause IDs you want negotiated as a Python list, "
            "e.g. ['clause_003', 'clause_007']. "
            "Press Enter with no input to auto-select all clauses scored 7 or higher."
        ),
        "clauses": [
            {
                "id":    a.get("clause_id"),
                "score": a.get("risk_score", 0),
                "flags": a.get("flags", []),
            }
            for a in clause_analyses
        ],
        "high_risk_ids": [
            a["clause_id"] for a in clause_analyses
            if a.get("risk_score", 0) >= 7
        ],
    }

    # --- PAUSE HERE — caller must resume with priority_clauses ---
    user_input = interrupt(interrupt_payload)

    # --- RESUMED — process user input ---
    priorities = _parse_priorities(user_input, interrupt_payload["high_risk_ids"])

    # Mark selected clauses as priority in analyses
    updated_analyses = {}
    for cid, analysis in analyses.items():
        updated_analyses[cid] = {
            **analysis,
            "priority": cid in priorities,
        }

    return {
        "priority_clauses": priorities,
        "analyses":         updated_analyses,
        "current_phase":    "negotiate",
    }


def _parse_priorities(user_input, high_risk_ids: list[str]) -> list[str]:
    """
    Parse user input into a list of clause_ids.
    Handles: list, comma-separated string, empty string (auto-select high risk).
    """
    if user_input is None or user_input == "" or user_input == []:
        return high_risk_ids

    if isinstance(user_input, list):
        return [str(x).strip() for x in user_input if x]

    if isinstance(user_input, str):
        text = user_input.strip()
        if not text:
            return high_risk_ids
        # Try parsing as Python list literal
        if text.startswith("["):
            try:
                return ast.literal_eval(text)
            except (ValueError, SyntaxError):
                pass
        # Fall back to comma-separated
        return [x.strip() for x in text.split(",") if x.strip()]

    return high_risk_ids
