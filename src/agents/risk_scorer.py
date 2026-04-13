"""
Risk scorer node.

Reads every clause from state, scores each one 1-10 for risk,
and writes results into state["analyses"] via the merge_clause_results reducer.
"""
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from src.graph.state import ContractState
from src.utils.llm import llm
from src.utils.prompts import RISK_SCORER


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class RiskResult(BaseModel):
    risk_score:     int       = Field(ge=1, le=10, description="Risk score 1-10")
    risk_reasoning: str       = Field(description="Specific reasoning referencing exact clause language")
    flags:          list[str] = Field(description="List of flag names from the allowed set")


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def risk_scorer_node(state: ContractState) -> dict:
    """
    Score every clause for risk.
    Skips clauses that already have a risk_score (idempotent — safe to re-run).
    """
    structured    = llm.with_structured_output(RiskResult)
    contract_type = state.get("contract_type", "contract")
    existing      = state.get("analyses", {})
    updates       = {}

    for clause in state["clauses"]:
        cid = clause["clause_id"]

        # Skip if already scored
        if cid in existing and existing[cid].get("risk_score"):
            continue

        try:
            result = structured.invoke([
                SystemMessage(RISK_SCORER.format(contract_type=contract_type)),
                HumanMessage(
                    f"Clause type: {clause['clause_type']}\n\n"
                    f"Clause text:\n{clause['original']}"
                ),
            ])

            updates[cid] = {
                "clause_id":      cid,
                "risk_score":     result.risk_score,
                "risk_reasoning": result.risk_reasoning,
                "flags":          result.flags,
                # Fields filled by other workers — initialise to defaults
                "plain_english":  existing.get(cid, {}).get("plain_english",  ""),
                "suggested_edit": existing.get(cid, {}).get("suggested_edit", ""),
                "priority":       existing.get(cid, {}).get("priority",       False),
            }

        except Exception as e:
            # Don't crash the whole pipeline on one bad clause
            updates[cid] = {
                "clause_id":      cid,
                "risk_score":     5,
                "risk_reasoning": f"Scoring failed: {e}",
                "flags":          ["scoring_error"],
                "plain_english":  "",
                "suggested_edit": "",
                "priority":       False,
            }

    return {"analyses": updates}
