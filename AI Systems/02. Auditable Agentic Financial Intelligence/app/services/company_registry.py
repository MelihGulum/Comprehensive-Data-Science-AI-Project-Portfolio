from __future__ import annotations

COMPANIES = {
    "Apple Inc. (AAPL)": {"name":"Apple Inc.","ticker":"AAPL","sector":"Technology","competitors":["MSFT","GOOGL","AMZN"]},
    "Microsoft Corp. (MSFT)": {"name":"Microsoft Corp.","ticker":"MSFT","sector":"Technology","competitors":["GOOGL","AAPL","AMZN"]},
    "Alphabet Inc. (GOOGL)": {"name":"Alphabet Inc.","ticker":"GOOGL","sector":"Technology","competitors":["MSFT","META","AMZN"]},
    "Amazon.com Inc. (AMZN)": {"name":"Amazon.com Inc.","ticker":"AMZN","sector":"Consumer / Cloud","competitors":["MSFT","GOOGL","WMT"]},
    "Meta Platforms Inc. (META)": {"name":"Meta Platforms Inc.","ticker":"META","sector":"Technology","competitors":["GOOGL","SNAP","PINS"]},
    "NVIDIA Corp. (NVDA)": {"name":"NVIDIA Corp.","ticker":"NVDA","sector":"Semiconductors","competitors":["AMD","INTC","AVGO"]},
    "Tesla Inc. (TSLA)": {"name":"Tesla Inc.","ticker":"TSLA","sector":"Automotive","competitors":["GM","F","TM"]},
}

def by_ticker(ticker: str) -> dict:
    t=ticker.upper()
    for item in COMPANIES.values():
        if item["ticker"] == t: return item
    return {"name":t,"ticker":t,"sector":"Unknown","competitors":[]}

def label_for_ticker(ticker: str) -> str:
    item=by_ticker(ticker); return f'{item["name"]} ({item["ticker"]})'
