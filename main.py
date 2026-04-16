"""
Legal Contract Reviewer — CLI entry point.

Usage:
  python main.py contracts/my_nda.pdf
  python main.py contracts/my_nda.pdf --output outputs/nda_review.md
  python main.py contracts/my_nda.pdf --thread-id session-42
"""

import traceback
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich.rule import Rule

load_dotenv()


def _sanitize_langsmith_env() -> bool:
    """
    Disable LangSmith tracing automatically when tracing is enabled but
    the API key is missing or clearly a placeholder.

    Returns True if tracing was auto-disabled.
    """
    tracing_raw = os.getenv("LANGCHAIN_TRACING_V2", "").strip().lower()
    tracing_on = tracing_raw in {"1", "true", "yes", "on"}

    if not tracing_on:
        return False

    api_key = os.getenv("LANGCHAIN_API_KEY", "").strip()
    is_placeholder = (
        not api_key
        or api_key in {"ls__...", "..."}
        or "..." in api_key
        or api_key.lower().startswith("changeme")
    )

    if is_placeholder:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return True

    return False


_LANGSMITH_DISABLED = _sanitize_langsmith_env()

from src.graph.builder import build_graph
from src.utils.pdf_loader import load_contract
from langgraph.types import Command

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def review(
    contract_path: Path = typer.Argument(
        ...,
        help="Path to the contract file (.txt or .pdf)",
        exists=True,
    ),
    thread_id: str = typer.Option(
        "default",
        "--thread-id",
        "-t",
        help="Session ID — use the same ID to resume a previous review",
    ),
    output: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save the markdown report (default: outputs/<filename>_review.md)",
    ),
    no_persist: bool = typer.Option(
        False,
        "--no-persist",
        help="Disable checkpointing (faster for one-shot runs)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show node-by-node progress logs",
    ),
):
    """
    Review a legal contract and generate a risk report with optional
    negotiation suggestions for clauses you flag as priority.
    """
    console.print()
    console.print(Rule("[bold]Legal Contract Reviewer[/bold]"))
    if _LANGSMITH_DISABLED:
        console.print(
            "[dim]LangSmith tracing disabled (missing/placeholder API key).[/dim]"
        )

    # --- Load contract ---
    console.print(f"\n📄 Loading: [cyan]{contract_path}[/cyan]")
    try:
        raw_text = load_contract(contract_path)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1)
    console.print(f"   {len(raw_text):,} characters loaded\n")

    # --- Build graph ---
    graph = build_graph(use_persistent_memory=not no_persist)
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "raw_text": raw_text,
        "contract_type": "",
        "clauses": [],
        "analyses": {},
        "current_phase": "start",
        "priority_clauses": [],
        "report": None,
    }

    # --- Run Phase 1: extraction + analysis ---
    console.print("[bold]Phase 1 — Extracting and analysing clauses...[/bold]")

    if verbose:
        # Stream events so node logs print as they happen
        result = _stream_with_logs(graph, initial_state, config, console)
    else:
        # Invoke normally — node wrapper still prints to stdout
        try:
            result = graph.invoke(initial_state, config=config)
        except Exception as e:
            console.print(
                f"\n[red bold]Graph error:[/red bold] {type(e).__name__}: {e}"
            )
            console.print(traceback.format_exc())
            raise typer.Exit(code=1)

    if result is None:
        console.print(
            "[red]Graph returned None — check the logs above for the error.[/red]"
        )
        raise typer.Exit(code=1)
   # --- Handle human-in-the-loop interrupt ---
    state_snapshot = graph.get_state(config)
    is_interrupted = bool(state_snapshot.next)

    if is_interrupted:
        # Reconstruct payload from the saved state values
        analyses = state_snapshot.values.get("analyses", {})
        clauses  = state_snapshot.values.get("clauses", [])
        ctype    = state_snapshot.values.get("contract_type", "Contract")

        from src.agents.flag_detector import CONTRACT_FLAGS_KEY
        clause_analyses = sorted(
            [v for k, v in analyses.items() if k != CONTRACT_FLAGS_KEY],
            key=lambda x: x.get("risk_score", 0),
            reverse=True,
        )
        contract_flags = analyses.get(CONTRACT_FLAGS_KEY, {})

        # Build summary inline
        lines = [f"## Risk Summary — {ctype}", ""]
        if contract_flags.get("flags"):
            lines += ["### ⚠️ Cross-Clause Issues", ""]
            for flag in contract_flags["flags"]:
                lines.append(f"- {flag}")
            lines.append("")
        lines += ["### Clause Risk Scores", "",
                  "| Clause ID | Risk | Flags |",
                  "|-----------|------|-------|"]
        for a in clause_analyses:
            cid   = a.get("clause_id", "?")
            score = a.get("risk_score", 0)
            flags = ", ".join(a.get("flags", [])) or "—"
            emoji = "🔴" if score >= 7 else "🟡" if score >= 4 else "🟢"
            lines.append(f"| `{cid}` | {emoji} {score}/10 | {flags} |")

        high_risk_ids = [a["clause_id"] for a in clause_analyses if a.get("risk_score", 0) >= 7]

        payload = {
            "summary":      "\n".join(lines),
            "instruction":  "Enter clause IDs to negotiate, e.g. clause_003, clause_006",
            "clauses":      [{"id": a["clause_id"], "score": a.get("risk_score", 0)} for a in clause_analyses],
            "high_risk_ids": high_risk_ids,
        }

        console.print("\n")
        console.print(Markdown(payload["summary"]))
        console.print()
        console.print(Rule())
        console.print(
            "\n[bold yellow]Which clauses do you want to negotiate?[/bold yellow]\n"
            f"[dim]{payload['instruction']}[/dim]\n"
        )

        high_risk = payload.get("high_risk_ids", [])
        if high_risk:
            console.print(
                f"[dim]High-risk clauses (auto-selected if you press Enter): "
                f"{', '.join(high_risk)}[/dim]\n"
            )

        raw_input_str = Prompt.ask("Your selection", default="").strip()

        if not raw_input_str:
            priorities = high_risk
            if priorities:
                console.print(f"[dim]Auto-selected: {priorities}[/dim]")
            else:
                console.print(
                    "[yellow]No high-risk clauses found — skipping negotiation.[/yellow]"
                )
        else:
            priorities = [
                x.strip()
                for x in raw_input_str.replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .split(",")
                if x.strip()
            ]

        # --- Run Phase 2: negotiation + report ---
        console.print(
            f"\n[bold]Phase 2 — Generating suggestions for: "
            f"{', '.join(priorities) if priorities else 'none (writing report only)'}[/bold]\n"
        )

        try:
            final = graph.invoke(Command(resume=priorities), config=config)
        except Exception as e:
            console.print(
                f"\n[red bold]Phase 2 error:[/red bold] {type(e).__name__}: {e}"
            )
            console.print(traceback.format_exc())
            raise typer.Exit(code=1)

    else:
        # Graph completed without interrupt (shouldn't happen in normal flow,
        # but handle gracefully)
        final = result

    # --- Output report ---
    report = (final or {}).get("report", "")
    if not report:
        console.print(
            "\n[red bold]No report was generated.[/red bold]\n"
            "Run with [bold]--verbose[/bold] or check the node logs printed above.\n"
            "Common causes:\n"
            "  • Ollama is not running  →  run: [bold]ollama serve[/bold]\n"
            "  • Model not pulled       →  run: [bold]ollama pull qwen2.5:14b[/bold]\n"
            "  • Wrong model name in .env  →  check PRIMARY_MODEL / FAST_MODEL\n"
            "  • Graph loop exited early   →  run: [bold]python debug_run.py[/bold]"
        )
        raise typer.Exit(code=1)

    console.print("\n")
    console.print(Rule("[bold]Review Report[/bold]"))
    console.print()
    console.print(Markdown(report))

    # --- Save report ---
    save_path = output or Path("outputs") / f"{contract_path.stem}_review.md"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(report, encoding="utf-8")

    console.print()
    console.print(Rule())
    console.print(f"\n✅ Report saved to [bold green]{save_path}[/bold green]\n")


def _stream_with_logs(graph, initial_state, config, console) -> dict:
    """Stream graph events and print each node as it completes."""
    result = {}
    try:
        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, node_output in event.items():
                if node_name == "__interrupt__":
                    result["__interrupt__"] = node_output
                else:
                    if isinstance(node_output, dict):
                        result.update(node_output)
    except Exception as e:
        console.print(f"\n[red bold]Stream error:[/red bold] {type(e).__name__}: {e}")
        console.print(traceback.format_exc())
        return None
    return result


if __name__ == "__main__":
    app()
