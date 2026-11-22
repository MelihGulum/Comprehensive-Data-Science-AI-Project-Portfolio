from __future__ import annotations
import logging
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class APIAgent:
    """Yahoo Finance ingestion adapter. yfinance is an unofficial Yahoo client, so outputs are normalized and validated downstream."""
    def __init__(self): self.cache = {}

    def get_market_data(self, symbols, start_date=None, end_date=None):
        end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        start_date = start_date or (datetime.now()-timedelta(days=30)).strftime('%Y-%m-%d')
        data = {}
        for symbol in symbols:
            try:
                key=(symbol.upper(), start_date, end_date)
                if key not in self.cache:
                    hist=yf.Ticker(symbol).history(start=start_date,end=end_date)
                    if hist.empty: continue
                    hist=hist.reset_index(); hist['Date']=hist['Date'].astype(str); self.cache[key]=hist
                data[symbol.upper()]=self.cache[key]
            except Exception as exc: logger.warning('Market data failed for %s: %s',symbol,exc)
        return data

    def get_financial_statements(self, symbol: str, years: int = 4) -> pd.DataFrame | None:
        """Normalize annual Yahoo income-statement fields to the project's canonical schema."""
        try:
            t=yf.Ticker(symbol)
            stmt=t.income_stmt
            if stmt is None or stmt.empty: stmt=t.financials
            if stmt is None or stmt.empty: return None
            aliases={
                'revenue':['Total Revenue','Operating Revenue'],
                'gross_profit':['Gross Profit'],
                'net_income':['Net Income','Net Income Common Stockholders'],
            }
            found={}
            for target,names in aliases.items():
                for name in names:
                    if name in stmt.index:
                        found[target]=stmt.loc[name]; break
            if set(found) != set(aliases):
                logger.warning('Required statement rows missing for %s: %s',symbol,set(aliases)-set(found)); return None
            currency=getattr(t,'fast_info',{}).get('currency','USD') if getattr(t,'fast_info',None) else 'USD'
            rows=[]
            for col in list(stmt.columns)[:years]:
                vals={k: pd.to_numeric(v.get(col), errors='coerce') for k,v in found.items()}
                if any(pd.isna(x) for x in vals.values()): continue
                rows.append({'year':int(pd.Timestamp(col).year),'revenue':float(vals['revenue']),'gross_profit':float(vals['gross_profit']),'net_income':float(vals['net_income']),'currency':currency or 'USD'})
            return pd.DataFrame(rows).sort_values('year').reset_index(drop=True) if rows else None
        except Exception as exc:
            logger.exception('Financial statement ingestion failed for %s: %s',symbol,exc); return None

    def get_earnings(self, symbol):
        df=self.get_financial_statements(symbol)
        if df is None: return None
        return df[['year','net_income']].rename(columns={'year':'Year','net_income':'Earnings'})

    def serialize_market_data(self, data):
        serialized={}
        for symbol,df in data.items():
            records=[]
            for record in df.to_dict(orient='records'):
                records.append({k:(None if pd.isna(v) else v if isinstance(v,(int,float,str,bool,type(None))) else str(v)) for k,v in record.items()})
            serialized[symbol]=records
        return serialized
