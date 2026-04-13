"""
Basic tests for the legal contract reviewer.
Run with: pytest tests/ -v
"""
import pytest
from pathlib import Path

from src.graph.state import ContractState, merge_clause_results
from src.utils.pdf_loader import load_contract


# ---------------------------------------------------------------------------
# State / reducer tests (no API calls)
# ---------------------------------------------------------------------------

class TestMergeClauseResults:
    def test_new_keys_added(self):
        existing = {}
        new      = {"clause_001": {"clause_id": "clause_001", "risk_score": 7}}
        result   = merge_clause_results(existing, new)
        assert "clause_001" in result
        assert result["clause_001"]["risk_score"] == 7

    def test_existing_keys_merged_not_replaced(self):
        existing = {"clause_001": {"clause_id": "clause_001", "risk_score": 7, "flags": ["one_sided"]}}
        new      = {"clause_001": {"clause_id": "clause_001", "plain_english": "This means: ..."}}
        result   = merge_clause_results(existing, new)
        # Both fields should be present
        assert result["clause_001"]["risk_score"]    == 7
        assert result["clause_001"]["flags"]         == ["one_sided"]
        assert result["clause_001"]["plain_english"] == "This means: ..."

    def test_new_key_wins_on_conflict(self):
        existing = {"clause_001": {"risk_score": 5}}
        new      = {"clause_001": {"risk_score": 8}}
        result   = merge_clause_results(existing, new)
        assert result["clause_001"]["risk_score"] == 8

    def test_contract_flags_key_preserved(self):
        existing = {"__contract_flags__": {"flags": ["flag_a"]}}
        new      = {"clause_001": {"risk_score": 3}}
        result   = merge_clause_results(existing, new)
        assert "__contract_flags__" in result
        assert "clause_001"         in result


# ---------------------------------------------------------------------------
# PDF loader tests (no API calls)
# ---------------------------------------------------------------------------

class TestPdfLoader:
    def test_load_txt_file(self):
        path = Path("tests/fixtures/sample_nda.txt")
        text = load_contract(path)
        assert len(text) > 100
        assert "NON-DISCLOSURE" in text

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_contract("does_not_exist.txt")

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported file type"):
            load_contract("contract.docx")


# ---------------------------------------------------------------------------
# Extractor tests (requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestExtractor:
    def test_extractor_produces_clauses(self):
        """Integration test — requires API key."""
        from src.agents.extractor import build_extractor_subgraph

        subgraph = build_extractor_subgraph()
        sample   = Path("tests/fixtures/sample_nda.txt").read_text()

        result = subgraph.invoke({
            "raw_text":      sample,
            "contract_type": "",
            "clauses":       [],
        })

        assert len(result["clauses"]) >= 5, "Should extract at least 5 clauses from sample NDA"
        assert result["contract_type"] != "", "Should detect contract type"

        for clause in result["clauses"]:
            assert clause["clause_id"].startswith("clause_")
            assert len(clause["original"]) > 10
            assert clause["clause_type"] != ""


# ---------------------------------------------------------------------------
# Full graph smoke test (requires OPENAI_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestFullGraph:
    def test_graph_reaches_interrupt(self):
        """
        Smoke test: graph should run extraction + analysis and pause at human_review.
        Does NOT resume — just verifies the interrupt fires correctly.
        """
        from src.graph.builder import build_graph

        graph  = build_graph(use_persistent_memory=True)
        config = {"configurable": {"thread_id": "test-smoke-001"}}
        sample = Path("tests/fixtures/sample_nda.txt").read_text()

        result = graph.invoke({
            "raw_text":         sample,
            "contract_type":    "",
            "clauses":          [],
            "analyses":         {},
            "current_phase":    "start",
            "priority_clauses": [],
            "report":           None,
        }, config=config)

        # Should have paused at human_review
        assert "__interrupt__" in result, "Graph should pause at human_review"

        payload = result["__interrupt__"][0].value
        assert "summary"  in payload
        assert "clauses"  in payload
        assert len(payload["clauses"]) >= 5
