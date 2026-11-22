# AI Agent Finance Benchmark

Cases: **5**

| Metric | Score |
|---|---:|
| expectation_pass_rate | 1.000 |
| fault_detection_rate | 1.000 |
| numerical_accuracy | 0.600 |
| citation_support_rate | 0.400 |
| unsupported_claim_rate | 0.800 |
| source_quality_score | 1.000 |

## Cases

| Case | Category | Expected behavior | Fault detection |
|---|---|---:|---:|
| clean_internal | clean | PASS | 1.000 |
| tampered_number | numerical_hallucination | PASS | 1.000 |
| missing_citation | citation_failure | PASS | 1.000 |
| evidence_mismatch | evidence_mismatch | PASS | 1.000 |
| bad_comparison | comparison_hallucination | PASS | 1.000 |
