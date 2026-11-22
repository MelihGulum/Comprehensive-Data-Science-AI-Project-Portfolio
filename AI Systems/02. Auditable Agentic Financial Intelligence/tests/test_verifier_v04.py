from app.agents.analyst import analyze
from app.agents.verifier import verify, revise
from app.schemas.models import AnalysisResult, Claim

def test_verifier_passes_reproducible_claims():
    m=[{'year':2025,'revenue_growth':.1,'gross_margin':.4,'net_margin':.12}]
    e=[{'company':'Rival','claim':'Rival gross margin: 35.0% (FY2025)','metric':'gross_margin','value':.35,'period':'FY2025','source_url':'https://example.com','source_title':'Annual'}]
    a=analyze('Acme',m,e); v=verify(a,m,e)
    assert v.passed and v.unsupported_claim_rate == 0

def test_verifier_detects_and_revises_tampered_value():
    m=[{'year':2025,'revenue_growth':.1,'gross_margin':.4,'net_margin':.12}]
    a=analyze('Acme',m,[]); a.internal_claims[0].value=.999
    v=verify(a,m,[]); assert not v.passed
    repaired=revise(a,v); assert all(c.claim_id != v.issues[0].claim_id for c in repaired.internal_claims)

def test_verifier_rejects_percentage_scale_bug():
    analysis = AnalysisResult(internal_claims=[Claim(
        claim_id="scaled", text="Gross margin was 4500%.", kind="internal_metric",
        company="Demo", metric="gross_margin", period="2025", value=45.0,
        calculation_ref="metrics[2025].gross_margin",
    )])
    result = verify(analysis, [{"year": 2025, "revenue_growth": 0.2, "gross_margin": 45.0, "net_margin": 0.1}], [])
    assert not result.passed
    assert "sanity_check_failed" in {issue.code for issue in result.issues}
