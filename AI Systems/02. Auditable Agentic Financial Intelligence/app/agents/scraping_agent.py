from __future__ import annotations
import logging
from datetime import datetime
import yfinance as yf
from newspaper import Article
logger=logging.getLogger(__name__)

class ScrapingAgent:
    """Collects recent market-news metadata/content. Direct article scraping is a fallback and failures are non-fatal."""
    def scrape_news(self, urls, timeout=10):
        articles=[]
        for url in urls:
            if 'finance.yahoo.com/quote/' in url:
                try:
                    symbol=url.split('/quote/')[1].split('/')[0]
                    for raw in (yf.Ticker(symbol).news or [])[:5]:
                        item=raw.get('content', raw) if isinstance(raw,dict) else {}
                        title=item.get('title') or raw.get('title','Market Update')
                        summary=item.get('summary') or raw.get('summary') or title
                        canonical=item.get('canonicalUrl') or {}
                        link=canonical.get('url') if isinstance(canonical,dict) else raw.get('link')
                        published=item.get('pubDate') or raw.get('providerPublishTime')
                        articles.append({'title':title,'text':summary,'url':link or url,'publish_date':str(published) if published else None,'source':'yfinance_news'})
                    if articles: continue
                except Exception as exc: logger.warning('yfinance news failed: %s',exc)
            try:
                a=Article(url); a.download(); a.parse()
                if a.text and len(a.text.strip())>50:
                    articles.append({'title':a.title or 'Market Article','text':a.text,'url':url,'publish_date':a.publish_date.isoformat() if a.publish_date else None,'source':'newspaper'})
            except Exception as exc: logger.warning('Article scrape failed for %s: %s',url,exc)
        return articles
