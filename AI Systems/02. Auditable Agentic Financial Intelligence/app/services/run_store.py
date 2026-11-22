from __future__ import annotations
import json, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path

class RunStore:
    def __init__(self, path='data/runs/finance_runs.db'):
        self.path=path; Path(path).parent.mkdir(parents=True, exist_ok=True); self._init()
    def _connect(self): return sqlite3.connect(self.path)
    def _init(self):
        with self._connect() as c:
            c.execute('''CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT, company TEXT, status TEXT, latency_ms REAL, estimated_cost_usd REAL DEFAULT 0, errors_json TEXT, warnings_json TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, ts TEXT, stage TEXT, payload_json TEXT)''')
    def start(self, company):
        rid='run_'+uuid.uuid4().hex[:12]
        with self._connect() as c: c.execute('INSERT INTO runs(run_id,started_at,company,status) VALUES(?,?,?,?)',(rid,datetime.now(timezone.utc).isoformat(),company,'running'))
        return rid
    def event(self, run_id, stage, payload):
        with self._connect() as c: c.execute('INSERT INTO events(run_id,ts,stage,payload_json) VALUES(?,?,?,?)',(run_id,datetime.now(timezone.utc).isoformat(),stage,json.dumps(payload,default=str)))
    def finish(self, run_id, status, latency_ms, errors=None, warnings=None, estimated_cost_usd=0.0):
        with self._connect() as c: c.execute('UPDATE runs SET finished_at=?,status=?,latency_ms=?,estimated_cost_usd=?,errors_json=?,warnings_json=? WHERE run_id=?',(datetime.now(timezone.utc).isoformat(),status,latency_ms,estimated_cost_usd,json.dumps(errors or []),json.dumps(warnings or []),run_id))
