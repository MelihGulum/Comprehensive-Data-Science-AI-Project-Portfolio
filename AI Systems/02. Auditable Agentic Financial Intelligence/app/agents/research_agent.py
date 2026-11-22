from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ResearchPlan:
    tools: list[str]
    rationale: list[str]

class ResearchAgent:
    """Bounded tool-selection policy for research.

    The agent chooses which capabilities are necessary from state rather than
    treating every connector as an agent. The actual providers remain tools.
    """
    def plan(self, *, competitors: list[str], competitor_financials: dict | None, include_news: bool=True) -> ResearchPlan:
        tools=[]; why=[]
        if competitors and not competitor_financials:
            tools.append("competitor_financials"); why.append("Comparable fundamentals are missing.")
        if competitors:
            tools.append("web_research"); why.append("External claims need source-backed evidence.")
        if include_news:
            tools.append("news_context"); why.append("Recent context can explain material movements without changing deterministic metrics.")
        return ResearchPlan(tools=tools, rationale=why)
