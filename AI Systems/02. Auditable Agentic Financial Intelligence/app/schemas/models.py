from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field, HttpUrl, field_validator

class FinancialRow(BaseModel):
    year: int
    revenue: float = Field(gt=0)
    gross_profit: float
    net_income: float
    currency: str = "USD"

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return value.strip().upper()

class Metric(BaseModel):
    year: int
    revenue_growth: float | None
    gross_margin: float
    net_margin: float

class ResearchSource(BaseModel):
    title: str
    url: HttpUrl
    published_at: str | None = None
    retrieved_at: datetime
    snippet: str = ""
    provider: str

class Evidence(BaseModel):
    company: str
    claim: str
    metric: str | None = None
    value: float | None = None
    period: str | None = None
    currency: str | None = None
    source_url: HttpUrl
    source_title: str
    source_date: str | None = None
    retrieved_at: datetime
    confidence: Literal["low", "medium", "high"] = "medium"
    raw_snippet: str = ""

class ResearchResult(BaseModel):
    company: str
    query: str
    sources: list[ResearchSource] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class Claim(BaseModel):
    """One report assertion with machine-checkable provenance."""
    claim_id: str
    text: str
    kind: Literal["internal_metric", "competitor_evidence", "comparison"]
    company: str
    metric: str | None = None
    period: str | None = None
    value: float | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    calculation_ref: str | None = None
    support: Literal["supported", "partial", "unsupported"] = "supported"

class Insight(BaseModel):
    """Higher-level interpretation grounded only in verified claim IDs."""
    insight_id: str
    category: Literal["profitability", "growth", "peer_positioning", "watch_item", "other"] = "other"
    title: str
    narrative: str
    claim_ids: list[str] = Field(default_factory=list)
    evidence_status: Literal["sufficient", "insufficient"] = "sufficient"

class AnalysisResult(BaseModel):
    executive_summary: list[str] = Field(default_factory=list)
    executive_analysis: str | None = None
    insights: list[Insight] = Field(default_factory=list)
    internal_claims: list[Claim] = Field(default_factory=list)
    competitor_claims: list[Claim] = Field(default_factory=list)
    comparison_claims: list[Claim] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class ReportArtifact(BaseModel):
    company: str
    generated_at: datetime
    markdown: str
    claim_count: int
    supported_claim_count: int
    evidence_count: int

class VerificationIssue(BaseModel):
    claim_id: str
    code: Literal["missing_calculation", "calculation_mismatch", "missing_evidence", "evidence_mismatch", "unsupported_claim", "sanity_check_failed"]
    message: str

class VerificationResult(BaseModel):
    passed: bool
    checked_claims: int
    supported_claims: int
    issues: list[VerificationIssue] = Field(default_factory=list)
    unsupported_claim_rate: float = 0.0

class EvaluationResult(BaseModel):
    numerical_accuracy: float
    citation_support_rate: float
    unsupported_claim_rate: float
    latency_ms: float
    estimated_cost_usd: float = 0.0
