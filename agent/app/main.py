from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.agent_registry import (
    DOCUMENTATION_AGENT_ID,
    ORCHESTRATOR_AGENT_ID,
    PHARMACY_AGENT_ID,
    SCHEDULING_AGENT_ID,
    agent_mount_path,
)
from app.agent import HospitalAgentRuntime
from app.service_apps import create_agent_service_app
from app.schemas import (
    AgentCard,
    AgentTaskRequest,
    AgentTaskResult,
    ApprovalDecisionRequest,
    ApprovalDecisionResponse,
    ApprovalRequest,
    BootstrapResponse,
    ChatRequest,
    ChatResponse,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
DB_PATH = BASE_DIR.parent / "data" / "agent_demo.db"

app = FastAPI(title="AI Agent Dashboard App", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

runtime = HospitalAgentRuntime(DB_PATH)
app.mount(
    agent_mount_path(ORCHESTRATOR_AGENT_ID),
    create_agent_service_app(
        agent_id=ORCHESTRATOR_AGENT_ID,
        title="Orchestrator Agent Service",
        get_card=runtime.get_agent_card,
        handle_task=runtime.handle_agent_task,
    ),
)
app.mount(
    agent_mount_path(PHARMACY_AGENT_ID),
    create_agent_service_app(
        agent_id=PHARMACY_AGENT_ID,
        title="Pharmacy Agent Service",
        get_card=runtime.get_agent_card,
        handle_task=runtime.handle_agent_task,
    ),
)
app.mount(
    agent_mount_path(SCHEDULING_AGENT_ID),
    create_agent_service_app(
        agent_id=SCHEDULING_AGENT_ID,
        title="Scheduling Agent Service",
        get_card=runtime.get_agent_card,
        handle_task=runtime.handle_agent_task,
    ),
)
app.mount(
    agent_mount_path(DOCUMENTATION_AGENT_ID),
    create_agent_service_app(
        agent_id=DOCUMENTATION_AGENT_ID,
        title="Documentation Agent Service",
        get_card=runtime.get_agent_card,
        handle_task=runtime.handle_agent_task,
    ),
)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.svg")


@app.get("/api/bootstrap", response_model=BootstrapResponse)
def bootstrap() -> dict:
    return runtime.bootstrap()


@app.get("/api/agents/cards", response_model=list[AgentCard])
def list_agent_cards() -> list[AgentCard]:
    return runtime.list_agent_cards()


@app.get("/api/agents/cards/{agent_id}", response_model=AgentCard)
def get_agent_card(agent_id: str) -> AgentCard:
    try:
        return runtime.get_agent_card(agent_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/a2a/tasks", response_model=AgentTaskResult)
def submit_agent_task(payload: AgentTaskRequest) -> AgentTaskResult:
    if payload.patient_id not in runtime.patient_lookup:
        raise HTTPException(status_code=404, detail="Unknown patient_id.")
    try:
        return runtime.handle_agent_task(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/a2a/tasks", response_model=list[dict])
def list_agent_tasks() -> list[dict]:
    return runtime.memory.list_agent_tasks()


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> dict:
    if payload.patient_id not in runtime.patient_lookup:
        raise HTTPException(status_code=404, detail="Unknown patient_id.")
    return runtime.chat(
        session_id=payload.session_id,
        nurse_id=payload.nurse_id,
        patient_id=payload.patient_id,
        message=payload.message,
    )


@app.get("/api/approvals", response_model=list[ApprovalRequest])
def approvals() -> list[ApprovalRequest]:
    return runtime.list_approvals()


@app.post("/api/approvals/{approval_id}", response_model=ApprovalDecisionResponse)
def decide_approval(approval_id: str, payload: ApprovalDecisionRequest) -> dict:
    try:
        return runtime.decide_approval(approval_id, payload.nurse_id, payload.decision)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/audit-log", response_model=list[dict])
def audit_log() -> list[dict]:
    return runtime.memory.recent_audit_log()
