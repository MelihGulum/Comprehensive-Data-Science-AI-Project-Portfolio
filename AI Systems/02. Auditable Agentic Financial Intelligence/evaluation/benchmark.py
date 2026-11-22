from __future__ import annotations
import json, statistics, time
from pathlib import Path
from typing import Callable
from app.schemas.models import AnalysisResult
from app.agents.verifier import verify
from evaluation.metrics import evaluate

DEFAULT_CASES = Path(__file__).parent / 'datasets' / 'benchmark_cases.json'

def load_cases(path: str | Path = DEFAULT_CASES) -> list[dict]:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def source_quality(evidence: list[dict]) -> float:
    """Simple transparent rubric: official/regulator=1, reputable secondary=.75, other=.5."""
    if not evidence: return 1.0
    vals=[]
    for e in evidence:
        url=str(e.get('source_url','')).lower()
        title=str(e.get('source_title','')).lower()
        if any(x in url for x in ('.gov','sec.gov','investor','annualreport')) or any(x in title for x in ('annual report','10-k','investor relations')): vals.append(1.0)
        elif any(x in url for x in ('reuters.com','bloomberg.com','ft.com','wsj.com')): vals.append(.75)
        else: vals.append(.5)
    return sum(vals)/len(vals)

def run_case(case: dict) -> dict:
    start=time.perf_counter()
    analysis=AnalysisResult(**case['analysis'])
    verification=verify(analysis, case.get('metrics',[]), case.get('evidence',[]))
    claims=[c.model_dump() for c in analysis.internal_claims+analysis.competitor_claims+analysis.comparison_claims]
    ev=evaluate(verification, claims, (time.perf_counter()-start)*1000)
    detected={i.code for i in verification.issues}
    expected=set(case.get('expected_issue_codes',[]))
    detection=(len(expected & detected)/len(expected)) if expected else float(not detected)
    return {'case_id':case['case_id'],'category':case['category'],'passed_expectation': expected==detected,
            'expected_issues':sorted(expected),'detected_issues':sorted(detected),'fault_detection_rate':detection,
            'source_quality_score':source_quality(case.get('evidence',[])), **ev.model_dump()}

def run_benchmark(path: str | Path = DEFAULT_CASES) -> dict:
    rows=[run_case(c) for c in load_cases(path)]
    def avg(k): return statistics.fmean(float(r[k]) for r in rows) if rows else 0.0
    return {'summary':{'cases':len(rows),'expectation_pass_rate':avg('passed_expectation'),
        'fault_detection_rate':avg('fault_detection_rate'),'numerical_accuracy':avg('numerical_accuracy'),
        'citation_support_rate':avg('citation_support_rate'),'unsupported_claim_rate':avg('unsupported_claim_rate'),
        'source_quality_score':avg('source_quality_score'),'avg_latency_ms':avg('latency_ms')},'cases':rows}

def write_report(result: dict, path: str | Path) -> None:
    s=result['summary']; lines=['# AI Agent Finance Benchmark','',f"Cases: **{s['cases']}**",'', '| Metric | Score |','|---|---:|']
    for k in ('expectation_pass_rate','fault_detection_rate','numerical_accuracy','citation_support_rate','unsupported_claim_rate','source_quality_score'):
        lines.append(f"| {k} | {s[k]:.3f} |")
    lines += ['', '## Cases', '', '| Case | Category | Expected behavior | Fault detection |','|---|---|---:|---:|']
    for r in result['cases']: lines.append(f"| {r['case_id']} | {r['category']} | {'PASS' if r['passed_expectation'] else 'FAIL'} | {r['fault_detection_rate']:.3f} |")
    Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')
