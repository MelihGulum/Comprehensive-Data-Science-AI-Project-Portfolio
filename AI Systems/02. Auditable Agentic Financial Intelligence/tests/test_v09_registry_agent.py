from app.services.company_registry import COMPANIES, by_ticker
from app.agents.research_agent import ResearchAgent

def test_registry_resolves_company():
    assert COMPANIES["Apple Inc. (AAPL)"]["ticker"] == "AAPL"
    assert by_ticker("msft")["name"] == "Microsoft Corp."

def test_research_agent_selects_tools():
    plan=ResearchAgent().plan(competitors=["MSFT"], competitor_financials={}, include_news=True)
    assert "competitor_financials" in plan.tools
    assert "web_research" in plan.tools
    assert "news_context" in plan.tools
