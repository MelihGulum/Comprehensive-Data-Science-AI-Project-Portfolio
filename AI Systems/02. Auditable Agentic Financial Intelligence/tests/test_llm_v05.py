from app.agents.analyst import analyze
from app.agents.llm_analyst import enrich_with_llm
from app.services.llm import FixtureLLM

def test_llm_can_rewrite_but_not_change_structured_value():
    a=analyze('Acme',[{'year':2025,'revenue_growth':.1,'gross_margin':.4,'net_margin':.2}],[])
    cid=a.internal_claims[0].claim_id; old=a.internal_claims[0].value
    llm=FixtureLLM({'executive_analysis':'Acme revenue growth was 10.0% in 2025.', 'insights':[], 'claim_rewrites':{cid:'Acme revenue growth was 10.0% in 2025.'}})
    out,usage=enrich_with_llm(a,llm)
    assert out.internal_claims[0].value == old
    assert out.executive_summary == ['Acme revenue growth was 10.0% in 2025.']
    assert out.executive_analysis == 'Acme revenue growth was 10.0% in 2025.'
    assert usage['input_tokens'] == 10

def test_unknown_claim_ids_are_ignored():
    a=analyze('Acme',[{'year':2025,'revenue_growth':.1,'gross_margin':.4,'net_margin':.2}],[])
    out,_=enrich_with_llm(a,FixtureLLM({'executive_summary':['fake'], 'claim_rewrites':{'fake':'invented'}}))
    assert all(c.claim_id != 'fake' for c in out.internal_claims)
