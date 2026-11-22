import json
from pathlib import Path
from evaluation.benchmark import run_benchmark, write_report
if __name__ == '__main__':
    result=run_benchmark(); out=Path('evaluation/results'); out.mkdir(parents=True,exist_ok=True)
    (out/'benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    write_report(result,out/'benchmark.md')
    print(json.dumps(result['summary'],indent=2))
