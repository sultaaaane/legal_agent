"""
Flag detector node.

Looks for contract-level red flags that only appear from reading
multiple clauses together — risks invisible when reading clause-by-clause.

Writes results under the special key "__contract_flags__" in analyses.
"""
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel

from src.graph.state import ContractState
from src.utils.llm import llm
from src.utils.prompts import FLAG_DETECTOR


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class FlagResult(BaseModel):
    critical_flags: list[str]  # plain-English description of each cross-clause issue
    clause_ids:     list[str]  # all clause_ids involved across all flags
    explanation:    str        # overall summary of the cross-clause risks


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

CONTRACT_FLAGS_KEY = "__contract_flags__"

def flag_detector_node(state: ContractState) -> dict:
    """
    Detect cross-clause red flags by reading the entire contract at once.
    Sends up to 500 chars per clause to stay within context limits.
    """
    contract_type = state.get("contract_type", "contract")
    clauses       = state.get("clauses", [])

    if not clauses:
        return {"analyses": {}}

    # Build a compact representation of all clauses for the model
    clause_block = "\n\n".join(
        f"[{c['clause_id']} — {c['clause_type']}]:\n{c['original'][:500]}"
        for c in clauses
    )

    structured = llm.with_structured_output(FlagResult)

    try:
        result = structured.invoke([
            SystemMessage(FLAG_DETECTOR.format(contract_type=contract_type)),
            HumanMessage(clause_block),
        ])

        contract_flags_entry = {
            "clause_id":      CONTRACT_FLAGS_KEY,
            "flags":          result.critical_flags,
            "risk_reasoning": result.explanation,
            "clause_ids":     result.clause_ids,
            "plain_english":  "",
            "risk_score":     0,    # not a per-clause score
            "suggested_edit": "",
            "priority":       False,
        }

    except Exception as e:
        contract_flags_entry = {
            "clause_id":      CONTRACT_FLAGS_KEY,
            "flags":          [f"Flag detection failed: {e}"],
            "risk_reasoning": str(e),
            "clause_ids":     [],
            "plain_english":  "",
            "risk_score":     0,
            "suggested_edit": "",
            "priority":       False,
        }

    return {"analyses": {CONTRACT_FLAGS_KEY: contract_flags_entry}}
