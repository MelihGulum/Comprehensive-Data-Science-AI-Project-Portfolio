from __future__ import annotations
from app.agents.api_agent import APIAgent
from app.agents.scraping_agent import ScrapingAgent

class YahooFinanceProvider(APIAgent):
    """Market/fundamental data provider. Compatibility wrapper around the v0.8 adapter."""
    name = "yahoo_finance"

class NewsScraper(ScrapingAgent):
    """Web/news extraction tool. It is a tool, not a reasoning agent."""
    name = "news_scraper"
