from __future__ import annotations
import pandas as pd
from app.services.providers import YahooFinanceProvider, NewsScraper
from app.tools.financial_calculator import validate_financials

class DataIngestionService:
    def __init__(self, api_agent=None, scraping_agent=None):
        self.api=api_agent or YahooFinanceProvider(); self.scraper=scraping_agent or NewsScraper()

    def from_ticker(self, ticker: str, years: int=4, include_market=True, include_news=True) -> dict:
        ticker=ticker.strip().upper()
        financials=self.api.get_financial_statements(ticker, years)
        if financials is None or financials.empty: raise ValueError(f'No usable annual financial statements returned for {ticker}.')
        rows=[r.model_dump() for r in validate_financials(financials)]
        market=self.api.get_market_data([ticker]).get(ticker) if include_market else None
        news=self.scraper.scrape_news([f'https://finance.yahoo.com/quote/{ticker}/news/']) if include_news else []
        return {'rows':rows,'market_data':self.api.serialize_market_data({ticker:market}).get(ticker,[]) if market is not None else [],'news':news,'ingestion':{'mode':'automatic','provider':'Yahoo Finance via yfinance','ticker':ticker,'years':years}}


    def ingest_competitors(self, tickers: list[str], years: int = 4) -> dict[str, list[dict]]:
        out = {}
        for ticker in tickers:
            df = self.api.get_financial_statements(ticker, years)
            if df is not None and not df.empty:
                out[ticker.upper()] = [r.model_dump() for r in validate_financials(df)]
        return out

    def from_csv(self, df: pd.DataFrame) -> dict:
        rows=[r.model_dump() for r in validate_financials(df)]
        return {'rows':rows,'market_data':[],'news':[],'ingestion':{'mode':'csv','provider':'User upload'}}
