from __future__ import annotations

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


class ApprovalRequest(BaseModel):
    approval_id: str
    session_id: str
    nurse_id: str
    patient_id: str
    tool_name: str
    reason: str
    payload: dict[str, Any]
    status: Literal["pending", "approved", "rejected"]


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
