from typing import TypedDict
class FinanceState(TypedDict, total=False):
    csv_path: str; company: str; ticker: str; competitors: list[str]; rows: list[dict]; metrics: list[dict]
    research: list[dict]; evidence: list[dict]; research_provider: object; analysis: dict
    verification: dict; verification_attempts: int; max_revision_attempts: int
    market_data: list[dict]; news: list[dict]; ingestion: dict; competitor_financials: dict; research_plan: dict; report_artifact: dict; llm_provider: object; llm_usage: list[dict]; errors: list[str]; warnings: list[str]; report: str
