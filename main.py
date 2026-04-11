import typer
from pathlib import Path
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from dotenv import load_dotenv

load_dotenv()

from src.graph.builder import build_graph
from src.utils.pdf_loader import load_contract
from langgraph.types import Command

app = typer.Typer()
console = Console()


@app.command()
def review(
    contract_path: Path = typer.Argument(
        ..., help="Path to contract file (.txt or .pdf)"
    ),
    thread_id: str = typer.Option("default", help="Session ID for checkpointing"),
    output: Path = typer.Option(None, help="Save report to this path"),
):
    """Review a legal contract and generate a risk report with negotiation suggestions."""

    # Load contract text
    console.print(f"\n[bold]Loading contract:[/bold] {contract_path}")
    raw_text = load_contract(contract_path)
    console.print(f"[green]✓[/green] Loaded {len(raw_text):,} characters\n")

    # Build and run graph
    graph = build_graph()
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

    console.print("[bold]Starting analysis...[/bold]")
    with console.status("Extracting and analyzing clauses..."):
        result = graph.invoke(initial_state, config=config)

    # Handle human-in-the-loop interrupt
    interrupt_data = result.get("__interrupt__")
    if interrupt_data:
        payload = interrupt_data[0].value

        console.print("\n")
        console.print(Markdown(payload["summary"]))
        console.print(
            "\n[bold yellow]Which clauses do you want negotiated?[/bold yellow]"
        )
        console.print(
            "[dim]Enter clause IDs separated by commas (e.g. clause_003, clause_007)[/dim]"
        )
        console.print(
            "[dim]Press Enter with no input to negotiate all high-risk clauses (7+)[/dim]\n"
        )

        raw_input = Prompt.ask("Clause IDs").strip()

        if raw_input == "":
            # Default: all clauses with risk score >= 7
            priorities = [c["id"] for c in payload["clauses"] if c["score"] >= 7]
            console.print(f"[dim]Selecting all high-risk clauses: {priorities}[/dim]")
        else:
            priorities = [x.strip() for x in raw_input.split(",")]

        console.print("\n[bold]Generating negotiation suggestions...[/bold]")
        with console.status("Writing counter-clauses..."):
            final = graph.invoke(Command(resume=priorities), config=config)
    else:
        final = result

    # Output report
    report = final.get("report", "No report generated.")
    console.print("\n")
    console.print(Markdown(report))

    # Save if requested
    save_path = output or Path("outputs") / f"{contract_path.stem}_review.md"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(report)
    console.print(f"\n[green]✓[/green] Report saved to [bold]{save_path}[/bold]")


if __name__ == "__main__":
    app()
