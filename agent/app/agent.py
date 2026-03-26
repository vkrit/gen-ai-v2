from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from app.agent_registry import (
    DOCUMENTATION_AGENT_ID,
    ORCHESTRATOR_AGENT_ID,
    PHARMACY_AGENT_ID,
    SCHEDULING_AGENT_ID,
    build_agent_cards,
)
from app.documentation_agent import DocumentationAgent
from app.memory import MemoryStore
from app.orchestrator_agent import OrchestratorAgent
from app.pharmacy_agent import PharmacyAgent
from app.scheduling_agent import SchedulingAgent
from app.schemas import AgentCard, AgentTaskRequest, AgentTaskResult, ApprovalRequest
from app.transport import A2ATransport
from app.tools import HospitalTools
from app.data import PATIENTS


class HospitalAgentRuntime:
    def __init__(self, db_path: Path) -> None:
        self.memory = MemoryStore(db_path)
        self.patient_lookup = {patient["patient_id"]: patient for patient in PATIENTS}
        self.agent_cards = build_agent_cards()
        self.tools = HospitalTools(self.memory, self.patient_lookup)
        self.transport = A2ATransport(self.get_agent_card)

        self.pharmacy_agent = PharmacyAgent(self.tools)
        self.scheduling_agent = SchedulingAgent(self.tools, self._create_approval)
        self.documentation_agent = DocumentationAgent(self.tools, self._create_approval)
        self.orchestrator_agent = OrchestratorAgent(self.memory, self.tools, self.transport.submit_task)

        self.transport.register_local_handler(PHARMACY_AGENT_ID, self.handle_agent_task)
        self.transport.register_local_handler(SCHEDULING_AGENT_ID, self.handle_agent_task)
        self.transport.register_local_handler(DOCUMENTATION_AGENT_ID, self.handle_agent_task)
        self.transport.register_local_handler(ORCHESTRATOR_AGENT_ID, self.handle_agent_task)

    def bootstrap(self) -> dict[str, Any]:
        return {
            "patients": PATIENTS,
            "approvals": self.list_approvals(),
            "audit_log": self.memory.recent_audit_log(),
        }

    def list_agent_cards(self) -> list[AgentCard]:
        ordered = [
            ORCHESTRATOR_AGENT_ID,
            PHARMACY_AGENT_ID,
            SCHEDULING_AGENT_ID,
            DOCUMENTATION_AGENT_ID,
        ]
        return [self.agent_cards[agent_id] for agent_id in ordered]

    def get_agent_card(self, agent_id: str) -> AgentCard:
        card = self.agent_cards.get(agent_id)
        if card is None:
            raise KeyError(f"Unknown agent_id: {agent_id}")
        return card

    def list_approvals(self) -> list[ApprovalRequest]:
        return self.memory.list_approvals(status="pending")

    def _create_approval(
        self,
        *,
        session_id: str,
        nurse_id: str,
        patient_id: str,
        tool_name: str,
        reason: str,
        payload: dict[str, Any],
        agent_id: str | None = None,
        task_id: str | None = None,
    ) -> ApprovalRequest:
        approval = ApprovalRequest(
            approval_id=f"APR-{uuid.uuid4().hex[:8].upper()}",
            session_id=session_id,
            nurse_id=nurse_id,
            patient_id=patient_id,
            tool_name=tool_name,
            reason=reason,
            payload=payload,
            status="pending",
            agent_id=agent_id,
            task_id=task_id,
        )
        self.memory.save_approval(approval)
        self.memory.log_action(
            nurse_id=nurse_id,
            patient_id=patient_id,
            action=tool_name,
            detail=payload,
            approval_status="pending",
        )
        return approval

    def handle_pharmacy_task(self, task: AgentTaskRequest) -> AgentTaskResult:
        return self.pharmacy_agent.handle_task(task)

    def handle_scheduling_task(self, task: AgentTaskRequest) -> AgentTaskResult:
        return self.scheduling_agent.handle_task(task)

    def handle_documentation_task(self, task: AgentTaskRequest) -> AgentTaskResult:
        return self.documentation_agent.handle_task(task)

    def handle_orchestrator_task(self, task: AgentTaskRequest) -> AgentTaskResult:
        self.memory.save_session_message(
            session_id=task.session_id,
            nurse_id=task.nurse_id,
            patient_id=task.patient_id,
            role="user",
            content=task.message,
        )
        orchestrator_response = self.orchestrator_agent.chat(
            session_id=task.session_id,
            nurse_id=task.nurse_id,
            patient_id=task.patient_id,
            message=task.message,
        )
        self.memory.save_session_message(
            session_id=task.session_id,
            nurse_id=task.nurse_id,
            patient_id=task.patient_id,
            role="assistant",
            content=orchestrator_response["reply"],
        )
        self.memory.save_episode(
            task.patient_id,
            task.nurse_id,
            f"Query: {task.message[:80]} | Outcome: {orchestrator_response['reply'][:120]}",
        )
        return AgentTaskResult(
            task_id=task.task_id,
            requester_agent_id=task.requester_agent_id,
            responder_agent_id=ORCHESTRATOR_AGENT_ID,
            status="completed",
            summary=orchestrator_response["reply"],
            data={
                "reply": orchestrator_response["reply"],
                "approvals": [approval.model_dump() for approval in self.list_approvals()],
                "audit_log": self.memory.recent_audit_log(),
            },
            trace=orchestrator_response["trace"],
        )

    def handle_agent_task(self, task: AgentTaskRequest) -> AgentTaskResult:
        self.get_agent_card(task.target_agent_id)
        self.memory.record_agent_task(task)

        if task.target_agent_id == ORCHESTRATOR_AGENT_ID:
            result = self.handle_orchestrator_task(task)
        elif task.target_agent_id == PHARMACY_AGENT_ID:
            result = self.handle_pharmacy_task(task)
        elif task.target_agent_id == SCHEDULING_AGENT_ID:
            result = self.handle_scheduling_task(task)
        elif task.target_agent_id == DOCUMENTATION_AGENT_ID:
            result = self.handle_documentation_task(task)
        else:
            raise KeyError(f"Unknown agent_id: {task.target_agent_id}")

        self.memory.save_agent_task_result(result)
        return result

    def chat(self, *, session_id: str, nurse_id: str, patient_id: str, message: str) -> dict[str, Any]:
        result = self.handle_agent_task(
            AgentTaskRequest(
                requester_agent_id="nurse-ui",
                target_agent_id=ORCHESTRATOR_AGENT_ID,
                session_id=session_id,
                nurse_id=nurse_id,
                patient_id=patient_id,
                intent="nurse_query",
                message=message,
            )
        )
        return {
            "session_id": session_id,
            "reply": result.summary,
            "trace": result.trace,
            "approvals": self.list_approvals(),
            "audit_log": self.memory.recent_audit_log(),
        }

    def decide_approval(self, approval_id: str, nurse_id: str, decision: str) -> dict[str, Any]:
        approval = self.memory.get_approval(approval_id)
        if approval is None:
            raise KeyError(f"Approval {approval_id} was not found.")

        updated = self.memory.update_approval_status(approval_id, decision)
        if updated is None:
            raise KeyError(f"Approval {approval_id} was not found.")

        self.memory.update_task_status_from_approval(updated.task_id, decision)
        self.memory.log_action(
            nurse_id=nurse_id,
            patient_id=updated.patient_id,
            action=updated.tool_name,
            detail=updated.payload,
            approval_status=decision,
        )
        return {
            "message": f"{updated.tool_name} {decision} for {updated.patient_id}.",
            "approvals": self.list_approvals(),
            "audit_log": self.memory.recent_audit_log(),
        }
