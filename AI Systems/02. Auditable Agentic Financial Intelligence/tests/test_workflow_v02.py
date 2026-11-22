from pathlib import Path
from app.graph.workflow import build_graph
from app.tools.research_tool import FixtureProvider

def test_graph_adds_research_evidence():
    csv_path = Path(__file__).parents[1] / "data" / "samples" / "example_financials.csv"
    provider = FixtureProvider({"Rival Corp": [{
        "title": "Annual results", "url": "https://example.com/results",
        "content": "FY2025 revenue grew 8.0%. Gross margin was 35.0%."
    }]})
    result = build_graph().invoke({
        "csv_path": str(csv_path), "company": "Example Corp",
        "competitors": ["Rival Corp"], "research_provider": provider,
    })
    assert not result.get("errors")
    assert result["metrics"]
    assert len(result["evidence"]) == 2
