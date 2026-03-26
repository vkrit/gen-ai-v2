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
- Agent Card discovery is available at `/api/agents/cards` and `/api/agents/cards/{agent_id}`.
- In-process A2A task handoff is available at `/api/a2a/tasks` for the orchestrator, pharmacy, scheduling, and documentation agents.
- Persisted A2A task history is available at `GET /api/a2a/tasks`.
- Approvals, A2A task records, session messages, episodic memory, and the audit trail are all stored in SQLite at `data/agent_demo.db`.
- Each agent is also mounted as its own service surface under `/agents/orchestrator`, `/agents/pharmacy-agent`, `/agents/scheduling-agent`, and `/agents/documentation-agent`, each exposing `GET /card` and `POST /tasks`.
- `A2A_TRANSPORT_MODE=http` with `A2A_BASE_URL=http://127.0.0.1:8000` switches the orchestrator transport to HTTP instead of local in-process dispatch.
