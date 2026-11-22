import pandas as pd
from app.schemas.models import FinancialRow, Metric

REQUIRED = {"year", "revenue", "gross_profit", "net_income"}

def validate_financials(df: pd.DataFrame) -> list[FinancialRow]:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")
    if df["year"].duplicated().any():
        raise ValueError("Duplicate years are not allowed")
    rows = [FinancialRow(**r) for r in df.to_dict("records")]
    currencies = {r.currency.upper() for r in rows}
    if len(currencies) != 1:
        raise ValueError("MVP requires a single currency")
    return sorted(rows, key=lambda r: r.year)

def calculate_metrics(rows: list[FinancialRow]) -> list[Metric]:
    out = []
    previous_revenue = None
    for r in rows:
        growth = None if previous_revenue is None else (r.revenue / previous_revenue - 1)
        out.append(Metric(
            year=r.year,
            revenue_growth=growth,
            gross_margin=r.gross_profit / r.revenue,
            net_margin=r.net_income / r.revenue,
        ))
        previous_revenue = r.revenue
    return out
