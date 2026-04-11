from __future__ import annotations
from typing import TypedDict, Annotated, Optional


def merge_clause_results(existing: dict, new: dict) -> dict:
    """
    Custom reducer for the analyses field.
    Merges new clause results into existing dict by clause_id key.
    Prevents parallel workers from overwriting each other's results.
    """
    merged = existing.copy()
    for clause_id, analysis in new.items():
        if clause_id in merged:
            # merge fields — don't replace the whole entry
            merged[clause_id] = {**merged[clause_id], **analysis}
        else:
            merged[clause_id] = analysis
    return merged


class Clause(TypedDict):
    clause_id: str
    clause_type: str
    original: str
    position: int


class ClauseAnalysis(TypedDict):
    clause_id: str
    plain_english: str
    risk_score: int
    risk_reasoning: str
    flags: list[str]
    suggested_edit: str
    priority: bool


class ContractState(TypedDict):
    # Input
    raw_text: str
    contract_type: str

    # Extraction output
    clauses: list[Clause]

    # Analysis output — clause_id → ClauseAnalysis
    # merge_clause_results ensures parallel writes don't collide
    analyses: Annotated[dict, merge_clause_results]

    # Supervisor routing
    current_phase: str

    # Human input
    priority_clauses: list[str]

    # Final output
    report: Optional[str]
