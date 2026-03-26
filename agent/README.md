# AI Agent Dashboard App

- FastAPI backend
- Browser dashboard
- Orchestrator plus specialist agent routing
- Read/write tool boundaries
- Human approval gate for write actions
- Working, episodic, and semantic memory layers
- Audit log and execution trace

## Run

```bash
. .venv/bin/activate
uv sync
uvicorn app.main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Notes

- The agent uses realistic demo hospital data and a deterministic planning loop so the app works locally without extra API keys.
- Write actions never execute immediately. They always create a pending approval that must be approved from the dashboard.
- Episodic memory and the audit trail are stored in SQLite at `data/agent_demo.db`.
