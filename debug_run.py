"""
Debug runner — runs the graph with full verbose logging so you can
see exactly which node fails and what the error is.

Usage:
  python debug_run.py tests/fixtures/sample_nda.txt
"""
import sys
import traceback
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.graph.builder import build_graph
from src.graph.state import ContractState
from src.utils.pdf_loader import load_contract

contract_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/sample_nda.txt")

print(f"\n=== DEBUG RUN: {contract_path} ===\n")

# ── Load contract ──────────────────────────────────────────────────────────
raw_text = load_contract(contract_path)
print(f"[loader]  {len(raw_text):,} chars loaded\n")

# ── Build graph ────────────────────────────────────────────────────────────
graph  = build_graph(use_persistent_memory=True)
config = {"configurable": {"thread_id": "debug-001"}}

initial_state: ContractState = {
    "raw_text":         raw_text,
    "contract_type":    "",
    "clauses":          [],
    "analyses":         {},
    "current_phase":    "start",
    "priority_clauses": [],
    "report":           None,
}

# ── Stream node-by-node so we see exactly where it stops ──────────────────
print("[graph]  streaming node events...\n")

try:
    for event in graph.stream(initial_state, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            print(f"[node: {node_name}]")

            if "__interrupt__" in str(node_name):
                print("  → INTERRUPT fired (human_review pause)")
                break

            # Print key state fields after each node
            if isinstance(node_output, dict):
                phase    = node_output.get("current_phase", "—")
                n_clauses = len(node_output.get("clauses", []))
                n_analyses = len(node_output.get("analyses", {}))
                report_ready = bool(node_output.get("report"))

                if phase != "—":
                    print(f"  current_phase  = {phase}")
                if n_clauses:
                    print(f"  clauses        = {n_clauses}")
                if n_analyses:
                    print(f"  analyses keys  = {n_analyses}")
                if "contract_type" in node_output and node_output["contract_type"]:
                    print(f"  contract_type  = {node_output['contract_type']}")
                if report_ready:
                    print(f"  report         = ready ({len(node_output['report'])} chars)")

                # Show any analyses with errors
                for cid, analysis in node_output.get("analyses", {}).items():
                    score = analysis.get("risk_score", 0)
                    flags = analysis.get("flags", [])
                    if "scoring_error" in flags or "flag_detection_failed" in str(flags):
                        print(f"  [WARN] {cid}: scoring failed — {analysis.get('risk_reasoning','')}")
                    else:
                        print(f"  {cid}: score={score} flags={flags[:2]}")
            print()

except KeyboardInterrupt:
    print("\n[interrupted by user]")
    sys.exit(0)

except Exception as e:
    print(f"\n[ERROR in graph]\n{type(e).__name__}: {e}\n")
    print("=== Full traceback ===")
    traceback.print_exc()
    sys.exit(1)

print("\n=== Stream complete ===")
