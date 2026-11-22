import pandas as pd
from langgraph.graph import StateGraph, START, END
from app.graph.state import FinanceState
from app.tools.financial_calculator import validate_financials, calculate_metrics
from app.tools.research_tool import default_provider, research_company
from app.agents.analyst import analyze
from app.agents.llm_analyst import enrich_with_llm
from app.services.llm import default_llm
from app.agents.reporter import build_report
from app.agents.verifier import verify, revise
from app.agents.research_agent import ResearchAgent
from app.schemas.models import AnalysisResult, FinancialRow, VerificationResult

def validate_node(state):
    try:
        if state.get('rows'):
            return {'rows':[r.model_dump() for r in validate_financials(pd.DataFrame(state['rows']))], 'errors':[]}
        return {'rows':[r.model_dump() for r in validate_financials(pd.read_csv(state['csv_path']))], 'errors':[]}
    except Exception as exc: return {'errors':[str(exc)]}
def metric_node(state): return {'metrics':[m.model_dump() for m in calculate_metrics([FinancialRow(**r) for r in state['rows']])]}
def research_node(state):
    plan=ResearchAgent().plan(competitors=state.get('competitors',[]), competitor_financials=state.get('competitor_financials'), include_news=bool(state.get('news')))
    if not state.get('competitors'): return {'research':[],'evidence':[],'warnings':[],'research_plan':plan.__dict__}
    try:
        p=state.get('research_provider') or default_provider(); rs=[research_company(c,p) for c in state['competitors']]
        return {'research':[r.model_dump(mode='json') for r in rs], 'evidence':[e.model_dump(mode='json') for r in rs for e in r.evidence], 'warnings':[w for r in rs for w in r.warnings], 'research_plan':plan.__dict__}
    except Exception as exc:
        if state.get('competitor_financials'):
            warning = f'Web research unavailable: {exc}. Direct competitor fundamentals remain available.'
        else:
            warning = f'Research unavailable: {exc}'
        return {'research':[],'evidence':[],'warnings':[warning],'research_plan':plan.__dict__}
def analysis_node(state):
    r=analyze(state.get('company','Company'),state.get('metrics',[]),state.get('evidence',[]))
    return {'analysis':r.model_dump(mode='json'),'warnings':state.get('warnings',[])+r.warnings,'verification_attempts':0}
def llm_analysis_node(state):
    llm=state.get('llm_provider') or default_llm()
    if llm is None: return {'llm_usage':[], 'warnings':state.get('warnings',[])+['LLM unavailable; deterministic analysis retained.']}
    try:
        a,u=enrich_with_llm(AnalysisResult(**state['analysis']),llm)
        return {'analysis':a.model_dump(mode='json'),'llm_usage':[u]}
    except Exception as exc:
        return {'llm_usage':[], 'warnings':state.get('warnings',[])+[f'LLM analysis unavailable: {exc}']}

def verify_node(state):
    a=AnalysisResult(**state['analysis']); v=verify(a,state.get('metrics',[]),state.get('evidence',[]))
    return {'verification':v.model_dump(mode='json'),'verification_attempts':state.get('verification_attempts',0)+1}
def revise_node(state):
    a=revise(AnalysisResult(**state['analysis']),VerificationResult(**state['verification']))
    return {'analysis':a.model_dump(mode='json'),'warnings':state.get('warnings',[])+[w for w in a.warnings if w not in state.get('warnings',[])]}
def report_node(state):
    a=AnalysisResult(**state['analysis']); artifact=build_report(state.get('company','Company'),a,state.get('evidence',[]),VerificationResult(**state['verification']))
    return {'report':artifact.markdown,'report_artifact':artifact.model_dump(mode='json')}
def route_validation(state): return 'stop' if state.get('errors') else 'calculate'
def route_verification(state):
    if state['verification']['passed']: return 'report'
    return 'revise' if state.get('verification_attempts',0) <= state.get('max_revision_attempts',2) else 'report'
def build_graph():
    g=StateGraph(FinanceState)
    for n,f in [('validate',validate_node),('calculate',metric_node),('research',research_node),('analyze',analysis_node),('llm_analyze',llm_analysis_node),('verify',verify_node),('revise',revise_node),('report',report_node)]: g.add_node(n,f)
    g.add_edge(START,'validate'); g.add_conditional_edges('validate',route_validation,{'calculate':'calculate','stop':END})
    g.add_edge('calculate','research'); g.add_edge('research','analyze'); g.add_edge('analyze','llm_analyze'); g.add_edge('llm_analyze','verify')
    g.add_conditional_edges('verify',route_verification,{'report':'report','revise':'revise'}); g.add_edge('revise','verify'); g.add_edge('report',END)
    return g.compile()
