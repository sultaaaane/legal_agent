"""
Graph builder — the single file that assembles the full LangGraph.

Import build_graph() from here. Nothing else should call StateGraph directly.
"""
import traceback
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import ContractState
from src.graph.supervisor import supervisor_node, route_supervisor
from src.graph.human_review import human_review_node

from src.agents.extractor import build_extractor_subgraph
from src.agents.risk_scorer import risk_scorer_node
from src.agents.plain_english import plain_english_node
from src.agents.flag_detector import flag_detector_node
from src.agents.negotiator import negotiator_node
from src.agents.report_writer import report_writer_node


# ---------------------------------------------------------------------------
# Node wrapper — catches and surfaces errors instead of silently failing
# ---------------------------------------------------------------------------

def _wrap(name: str, fn):
    """
    Wraps a node function so any exception prints the node name,
    full traceback, and re-raises — making silent failures impossible.
    """
    def wrapped(state: ContractState) -> dict:
        print(f"  [node:{name}] starting...")
        try:
            result = fn(state)
            # Print a brief summary of what the node produced
            if isinstance(result, dict):
                phase     = result.get("current_phase", "")
                n_clauses = len(result.get("clauses", []))
                n_analyses= len(result.get("analyses", {}))
                ctype     = result.get("contract_type", "")
                parts = []
                if ctype:                parts.append(f"type={ctype!r}")
                if n_clauses:            parts.append(f"clauses={n_clauses}")
                if n_analyses:           parts.append(f"analyses={n_analyses}")
                if phase:                parts.append(f"phase={phase!r}")
                if result.get("report"): parts.append("report=ready")
                print(f"  [node:{name}] done  {' | '.join(parts) if parts else '(no key fields changed)'}")
            return result
        except Exception as e:
            print(f"\n  [node:{name}] FAILED: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            raise
    wrapped.__name__ = name
    return wrapped


# ---------------------------------------------------------------------------
# Parallel analysis wrapper
# ---------------------------------------------------------------------------

def analyze_parallel_node(state: ContractState) -> dict:
    """
    Runs the three analysis workers in sequence.
    Each worker reads the current state and writes only its own fields.
    The merge_clause_results reducer ensures their outputs are safely combined.
    """
    n = len(state.get("clauses", []))
    print(f"  [analyze_parallel] scoring {n} clauses across 3 workers...")

    # Worker 1: risk scores
    print(f"  [analyze_parallel] worker 1/3 — risk_scorer")
    result_1 = risk_scorer_node(state)

    merged_analyses = {**state.get("analyses", {})}
    for cid, analysis in result_1["analyses"].items():
        merged_analyses[cid] = {**merged_analyses.get(cid, {}), **analysis}
    state_after_1 = {**state, "analyses": merged_analyses}

    # Worker 2: plain English
    print(f"  [analyze_parallel] worker 2/3 — plain_english")
    result_2 = plain_english_node(state_after_1)

    for cid, analysis in result_2["analyses"].items():
        merged_analyses[cid] = {**merged_analyses.get(cid, {}), **analysis}
    state_after_2 = {**state_after_1, "analyses": merged_analyses}

    # Worker 3: contract-level flag detection
    print(f"  [analyze_parallel] worker 3/3 — flag_detector")
    result_3 = flag_detector_node(state_after_2)

    combined_analyses = {}
    for r in [result_1, result_2, result_3]:
        for cid, analysis in r["analyses"].items():
            combined_analyses[cid] = {**combined_analyses.get(cid, {}), **analysis}

    scored = sum(1 for a in combined_analyses.values() if a.get("risk_score", 0) > 0)
    print(f"  [analyze_parallel] complete — {scored}/{n} clauses scored")

    return {
        "analyses":      combined_analyses,
        "current_phase": "wait_for_human",
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(use_persistent_memory: bool = False):
    """
    Assemble and compile the full contract review graph.

    Args:
        use_persistent_memory: If True, use MemorySaver checkpointing.

    Returns:
        Compiled LangGraph app ready for .invoke()
    """
    graph = StateGraph(ContractState)

    # Register nodes — all wrapped for visible error reporting
    graph.add_node("supervisor",       _wrap("supervisor",       supervisor_node))
    graph.add_node("extractor",        build_extractor_subgraph())   # subgraph wraps itself
    graph.add_node("analyze_parallel", _wrap("analyze_parallel", analyze_parallel_node))
    graph.add_node("human_review",     _wrap("human_review",     human_review_node))
    graph.add_node("negotiator",       _wrap("negotiator",       negotiator_node))
    graph.add_node("report_writer",    _wrap("report_writer",    report_writer_node))

    # Wire edges
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_supervisor)
    graph.add_edge("extractor",        "supervisor")
    graph.add_edge("analyze_parallel", "supervisor")
    graph.add_edge("human_review",     "negotiator")
    graph.add_edge("negotiator",       "report_writer")
    graph.add_edge("report_writer",    END)

    checkpointer = MemorySaver() if use_persistent_memory else None

    app = graph.compile(
        checkpointer     = checkpointer,
        interrupt_before = ["human_review"],
    )

    return app
