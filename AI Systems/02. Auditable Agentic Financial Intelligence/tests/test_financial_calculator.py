import pandas as pd
import pytest
from app.tools.financial_calculator import validate_financials, calculate_metrics

def test_metrics():
    df = pd.DataFrame([
        {"year": 2024, "revenue": 100, "gross_profit": 40, "net_income": 10, "currency": "USD"},
        {"year": 2025, "revenue": 120, "gross_profit": 54, "net_income": 15, "currency": "USD"},
    ])
    m = calculate_metrics(validate_financials(df))
    assert m[0].revenue_growth is None
    assert m[1].revenue_growth == pytest.approx(0.20)
    assert m[1].gross_margin == pytest.approx(0.45)
    assert m[1].net_margin == pytest.approx(0.125)

def test_missing_column():
    with pytest.raises(ValueError):
        validate_financials(pd.DataFrame([{"year": 2025, "revenue": 1}]))

def test_metric_units_are_decimal_ratios_for_ui_and_agents():
    df = pd.DataFrame([
        {"year": 2024, "revenue": 100, "gross_profit": 40, "net_income": 10, "currency": "USD"},
        {"year": 2025, "revenue": 120, "gross_profit": 54, "net_income": 15, "currency": "USD"},
    ])
    latest = calculate_metrics(validate_financials(df))[-1]
    assert latest.revenue_growth == pytest.approx(0.20)
    assert latest.gross_margin == pytest.approx(0.45)
    assert latest.net_margin == pytest.approx(0.125)
    # UI formatting with .1% now yields 20.0%, 45.0%, 12.5%, not 2000% etc.
    assert f"{latest.gross_margin:.1%}" == "45.0%"
