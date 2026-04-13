"""
Extractor subgraph — runs as a compiled sub-graph inside the main graph.

Nodes (in order):
  detect_type → chunk → classify

Input:  raw_text (str)
Output: contract_type (str), clauses (list[Clause])
"""

import json
from typing import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from src.graph.state import Clause
from src.utils.llm import llm, llm_fast
from src.utils.prompts import (
    DETECT_CONTRACT_TYPE,
    CHUNK_CONTRACT,
    CLASSIFY_CLAUSES,
)


# ---------------------------------------------------------------------------
# Extractor-local state
# (Separate from ContractState — subgraph has its own isolated state.
#  Fields that share a name with ContractState flow in/out automatically.)
# ---------------------------------------------------------------------------


class ExtractorState(TypedDict):
    raw_text: str
    contract_type: str
    clauses: list[Clause]


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------


class ChunkResult(BaseModel):
    clauses: list[dict]  # [{"clause_id": "...", "text": "...", "position": 1}]


class ClassifyResult(BaseModel):
    classifications: list[dict]  # [{"clause_id": "...", "type": "..."}]


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def detect_type_node(state: ExtractorState) -> dict:
    """Detect the contract type from the first 2000 characters."""
    sample = state["raw_text"][:2000]
    response = llm_fast.invoke(
        [
            SystemMessage(DETECT_CONTRACT_TYPE),
            HumanMessage(sample),
        ]
    )
    contract_type = response.content.strip().strip(".")
    return {"contract_type": contract_type}


def chunk_node(state: ExtractorState) -> dict:
    """
    Split the raw contract text into individual clauses.
    Handles contracts up to ~30 pages by chunking at 12 000 chars
    (safe buffer under GPT-4o's context window).
    """
    raw = state["raw_text"]

    # For very long contracts, process in windows and deduplicate
    MAX_CHARS = 12_000
    all_clauses = []
    offset = 0
    position = 1

    while offset < len(raw):
        window = raw[offset : offset + MAX_CHARS]
        response = llm.invoke(
            [
                SystemMessage(CHUNK_CONTRACT),
                HumanMessage(window),
            ]
        )

        try:
            # Strip markdown code fences if model wraps the JSON
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            data = json.loads(content)
        except json.JSONDecodeError:
            # If parsing fails, treat the whole window as one clause
            data = {
                "clauses": [
                    {
                        "clause_id": f"clause_{position:03d}",
                        "text": window,
                        "position": position,
                    }
                ]
            }

        for c in data.get("clauses", []):
            all_clauses.append(
                Clause(
                    clause_id=f"clause_{position:03d}",
                    clause_type="unknown",
                    original=c["text"].strip(),
                    position=position,
                )
            )
            position += 1

        # Advance window; overlap 500 chars to avoid cutting mid-clause
        offset += MAX_CHARS - 500
        if offset >= len(raw):
            break

    return {"clauses": all_clauses}


def classify_node(state: ExtractorState) -> dict:
    """
    Classify each clause by legal type.
    Sends clauses in batches of 20 to stay within token limits.
    """
    clauses = state["clauses"]
    BATCH_SIZE = 20
    type_map = {}

    for i in range(0, len(clauses), BATCH_SIZE):
        batch = clauses[i : i + BATCH_SIZE]

        clause_list = "\n\n".join(
            f"[{c['clause_id']}]:\n{c['original'][:400]}" for c in batch
        )

        structured = llm_fast.with_structured_output(ClassifyResult)
        result = structured.invoke(
            [
                SystemMessage(CLASSIFY_CLAUSES),
                HumanMessage(clause_list),
            ]
        )

        for item in result.classifications:
            type_map[item["clause_id"]] = item["type"]

    # Merge classifications back into clauses
    updated_clauses = [
        {**clause, "clause_type": type_map.get(clause["clause_id"], "other")}
        for clause in clauses
    ]

    return {"clauses": updated_clauses}


# ---------------------------------------------------------------------------
# Build and compile the extractor subgraph
# ---------------------------------------------------------------------------


def build_extractor_subgraph():
    graph = StateGraph(ExtractorState)

    graph.add_node("detect_type", detect_type_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("classify", classify_node)

    graph.add_edge(START, "detect_type")
    graph.add_edge("detect_type", "chunk")
    graph.add_edge("chunk", "classify")
    graph.add_edge("classify", END)

    return graph.compile()
