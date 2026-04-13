"""
Negotiator node.

Generates alternative clause wording ONLY for clauses the user
has marked as priority. Leaves all other clauses untouched.
"""
from langchain_core.messages import SystemMessage, HumanMessage

from src.graph.state import ContractState
from src.utils.llm import llm
from src.utils.prompts import NEGOTIATOR


def negotiator_node(state: ContractState) -> dict:
    """
    For each priority clause, generate a suggested alternative and
    a negotiation note explaining why the change is reasonable.

    Writes suggested_edit back into analyses via the merge reducer.
    """
    priorities    = state.get("priority_clauses", [])
    contract_type = state.get("contract_type", "contract")
    clause_map    = {c["clause_id"]: c for c in state["clauses"]}
    existing      = state.get("analyses", {})
    updates       = {}

    if not priorities:
        return {"analyses": {}, "current_phase": "finish"}

    for cid in priorities:
        clause   = clause_map.get(cid)
        analysis = existing.get(cid, {})

        if not clause:
            continue

        flags     = analysis.get("flags", [])
        flags_str = ", ".join(flags) if flags else "general risk concerns"

        try:
            response = llm.invoke([
                SystemMessage(NEGOTIATOR.format(
                    contract_type = contract_type,
                    flags         = flags_str,
                )),
                HumanMessage(
                    f"Clause type: {clause['clause_type']}\n\n"
                    f"Original clause:\n{clause['original']}"
                ),
            ])

            suggested = response.content.strip()

            # Validate expected format is present; add header if missing
            if "SUGGESTED EDIT:" not in suggested:
                suggested = f"SUGGESTED EDIT:\n{suggested}\n\nNEGOTIATION NOTE:\nThis revision balances both parties' interests."

        except Exception as e:
            suggested = (
                f"SUGGESTED EDIT:\n[Generation failed: {e}]\n\n"
                f"NEGOTIATION NOTE:\nPlease review this clause manually."
            )

        updates[cid] = {
            **analysis,
            "clause_id":      cid,
            "suggested_edit": suggested,
            "priority":       True,
        }

    return {
        "analyses":      updates,
        "current_phase": "finish",
    }
