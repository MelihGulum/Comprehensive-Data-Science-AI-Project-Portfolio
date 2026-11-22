from evaluation.benchmark import run_benchmark, source_quality

def test_adversarial_benchmark_detects_all_seeded_faults():
    r=run_benchmark()
    assert r['summary']['cases'] >= 5
    assert r['summary']['expectation_pass_rate'] == 1.0
    assert r['summary']['fault_detection_rate'] == 1.0

def test_source_quality_prefers_primary_source():
    assert source_quality([{'source_url':'https://www.sec.gov/x','source_title':'10-K'}]) == 1.0
    assert source_quality([{'source_url':'https://example.com/x','source_title':'Blog'}]) == .5
