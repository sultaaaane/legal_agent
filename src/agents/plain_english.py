"""
Plain English rewriter node.

Rewrites each clause in plain language and merges the result
into state["analyses"] via the merge_clause_results reducer.
"""
from langchain_core.messages import SystemMessage, HumanMessage

from src.graph.state import ContractState
from src.utils.llm import llm_fast
from src.utils.prompts import PLAIN_ENGLISH


def plain_english_node(state: ContractState) -> dict:
    """
    Rewrite every clause in plain English.
    Skips clauses that already have a plain_english value.
    Uses the fast/cheap model — high volume, lower stakes.
    """
    existing = state.get("analyses", {})
    updates  = {}

    for clause in state["clauses"]:
        cid = clause["clause_id"]

        # Skip if already written
        if existing.get(cid, {}).get("plain_english"):
            continue

        try:
            response = llm_fast.invoke([
                SystemMessage(PLAIN_ENGLISH),
                HumanMessage(clause["original"]),
            ])

            plain = response.content.strip()

            # Ensure the "This means:" prefix is present
            if not plain.lower().startswith("this means"):
                plain = "This means: " + plain

            updates[cid] = {
                "clause_id":     cid,
                "plain_english": plain,
            }

        except Exception as e:
            updates[cid] = {
                "clause_id":     cid,
                "plain_english": f"This means: [rewrite failed — {e}]",
            }

    return {"analyses": updates}
