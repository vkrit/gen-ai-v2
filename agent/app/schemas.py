from __future__ import annotations

from uuid import uuid4
from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    nurse_id: str = Field(min_length=1, default="nurse-demo")
    patient_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class TraceStep(BaseModel):
    kind: Literal["memory", "route", "tool", "approval", "response"]
    title: str
    detail: str
    agent_id: str | None = None


class ApprovalRequest(BaseModel):
    approval_id: str
    session_id: str
    nurse_id: str
    patient_id: str
    tool_name: str
    reason: str
    payload: dict[str, Any]
    status: Literal["pending", "approved", "rejected"]
    agent_id: str | None = None
    task_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    trace: list[TraceStep]
    approvals: list[ApprovalRequest]
    audit_log: list[dict[str, Any]]


class ApprovalDecisionRequest(BaseModel):
    nurse_id: str = Field(min_length=1, default="nurse-demo")
    decision: Literal["approved", "rejected"]


class ApprovalDecisionResponse(BaseModel):
    message: str
    approvals: list[ApprovalRequest]
    audit_log: list[dict[str, Any]]


class BootstrapResponse(BaseModel):
    patients: list[dict[str, Any]]
    approvals: list[ApprovalRequest]
    audit_log: list[dict[str, Any]]


class AgentCapability(BaseModel):
    id: str
    name: str
    description: str


class AgentCard(BaseModel):
    agent_id: str
    name: str
    version: str
    role: Literal["orchestrator", "specialist"]
    description: str
    card_path: str
    endpoint: str
    capabilities: list[AgentCapability]
    accepts: list[str] = Field(default_factory=list)
    returns: list[str] = Field(default_factory=list)
    approval_required_actions: list[str] = Field(default_factory=list)
    downstream_agents: list[str] = Field(default_factory=list)


class AgentTaskRequest(BaseModel):
    task_id: str = Field(default_factory=lambda: f"TASK-{uuid4().hex[:10].upper()}")
    requester_agent_id: str
    target_agent_id: str
    session_id: str = Field(min_length=1)
    nurse_id: str = Field(min_length=1)
    patient_id: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    message: str = Field(min_length=1)
    context: dict[str, Any] = Field(default_factory=dict)


class AgentTaskResult(BaseModel):
    task_id: str
    requester_agent_id: str
    responder_agent_id: str
    status: Literal["completed", "needs_approval", "rejected"]
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    trace: list[TraceStep] = Field(default_factory=list)
    approval: ApprovalRequest | None = None
