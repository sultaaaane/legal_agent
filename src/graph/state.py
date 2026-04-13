from __future__ import annotations
from typing import TypedDict, Annotated, Optional


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------

def merge_clause_results(existing: dict, new: dict) -> dict:
    """
    Custom reducer for the `analyses` field.

    Parallel workers each write their own keys into analyses
    (risk_score, plain_english, flags …).  Without this reducer,
    the second writer would blow away the first writer's keys.

    This function merges at the clause level AND at the field level:
      - New clause_id  → added to merged dict
      - Existing clause_id → fields merged (new keys win, existing kept)
    """
    merged = existing.copy()
    for clause_id, analysis in new.items():
        if clause_id in merged:
            merged[clause_id] = {**merged[clause_id], **analysis}
        else:
            merged[clause_id] = analysis
    return merged


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

class Clause(TypedDict):
    clause_id:   str   # "clause_001"
    clause_type: str   # "indemnification", "termination", etc.
    original:    str   # verbatim text from the contract
    position:    int   # order in document (1-based)


class ClauseAnalysis(TypedDict):
    clause_id:      str
    plain_english:  str        # 1-3 sentence plain-language summary
    risk_score:     int        # 1-10
    risk_reasoning: str        # why this score was given
    flags:          list[str]  # e.g. ["one_sided", "uncapped_liability"]
    suggested_edit: str        # negotiation alternative (empty until negotiator runs)
    priority:       bool       # True = user flagged for negotiation


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class ContractState(TypedDict):
    # ---- Input ----
    raw_text:         str            # full contract text loaded from file
    contract_type:    str            # detected: "NDA", "SaaS Agreement", etc.

    # ---- Extraction ----
    clauses:          list[Clause]   # structured list of clauses

    # ---- Analysis ----
    # clause_id → ClauseAnalysis
    # Annotated with merge_clause_results so parallel workers safely merge
    analyses:         Annotated[dict, merge_clause_results]

    # ---- Supervisor control ----
    # Tracks where we are in the pipeline so the supervisor can route correctly
    # Values: "start" | "extracting" | "analyzing" | "waiting" | "negotiating" | "done"
    current_phase:    str

    # ---- Human input ----
    priority_clauses: list[str]      # clause_ids the user wants negotiated

    # ---- Output ----
    report:           Optional[str]  # final markdown report (None until done)
