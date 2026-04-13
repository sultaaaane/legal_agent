"""
Supervisor node + router.

The supervisor reads the current state, decides the next phase,
and the router translates that phase into the next graph node to run.
"""
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import END
from pydantic import BaseModel

from src.graph.state import ContractState
from src.utils.llm import llm_fast
from src.utils.prompts import SUPERVISOR
from src.agents.flag_detector import CONTRACT_FLAGS_KEY


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class SupervisorDecision(BaseModel):
    next_action: Literal[
        "extract",
        "analyze_parallel",
        "wait_for_human",
        "negotiate",
        "finish",
        "done",
    ]
    reasoning: str


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------

def supervisor_node(state: ContractState) -> dict:
    """
    Reads current state and decides what phase comes next.
    Uses rule-based logic first (fast and deterministic),
    falls back to LLM only when the state is ambiguous.
    """
    clauses          = state.get("clauses", [])
    analyses         = state.get("analyses", {})
    current_phase    = state.get("current_phase", "start")
    priority_clauses = state.get("priority_clauses", [])
    report           = state.get("report")

    # --- Rule-based routing (deterministic, no LLM cost) ---

    # Already done
    if report or current_phase == "done":
        return {"current_phase": "done"}

    # No clauses yet — need to extract
    if not clauses:
        return {"current_phase": "extract"}

    # Clauses exist but not all analyzed
    clause_ids      = {c["clause_id"] for c in clauses}
    analyzed_ids    = {k for k in analyses if k != CONTRACT_FLAGS_KEY}
    all_have_score  = all(
        analyses.get(cid, {}).get("risk_score") for cid in clause_ids
    )
    all_have_plain  = all(
        analyses.get(cid, {}).get("plain_english") for cid in clause_ids
    )
    fully_analyzed  = all_have_score and all_have_plain

    if not fully_analyzed:
        return {"current_phase": "analyze_parallel"}

    # All analyzed, waiting for human input
    if not priority_clauses:
        return {"current_phase": "wait_for_human"}

    # Human gave input, need to negotiate
    if current_phase in ("wait_for_human", "waiting", "analyzing"):
        any_negotiated = any(
            analyses.get(cid, {}).get("suggested_edit")
            for cid in priority_clauses
        )
        if not any_negotiated:
            return {"current_phase": "negotiate"}

    # Negotiation done, write report
    if current_phase in ("negotiate", "negotiating", "finish") and not report:
        return {"current_phase": "finish"}

    # --- LLM fallback for any ambiguous state ---
    structured = llm_fast.with_structured_output(SupervisorDecision)

    status_summary = (
        f"current_phase:      {current_phase}\n"
        f"clauses_found:      {len(clauses)}\n"
        f"clauses_analyzed:   {len(analyzed_ids)}\n"
        f"all_scored:         {all_have_score}\n"
        f"all_plain_english:  {all_have_plain}\n"
        f"priority_clauses:   {priority_clauses}\n"
        f"report_written:     {bool(report)}"
    )

    decision = structured.invoke([
        SystemMessage(SUPERVISOR),
        HumanMessage(status_summary),
    ])

    return {"current_phase": decision.next_action}


# ---------------------------------------------------------------------------
# Router — translates phase name to graph node name
# ---------------------------------------------------------------------------

PHASE_TO_NODE = {
    "extract":          "extractor",
    "analyze_parallel": "analyze_parallel",
    "wait_for_human":   "human_review",
    "negotiate":        "negotiator",
    "finish":           "report_writer",
    "done":             END,
}

def route_supervisor(state: ContractState) -> str:
    phase = state.get("current_phase", "extract")
    return PHASE_TO_NODE.get(phase, "extractor")
