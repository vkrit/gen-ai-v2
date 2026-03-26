from __future__ import annotations

from collections.abc import Callable

from app.agent_registry import DOCUMENTATION_AGENT_ID
from app.schemas import AgentTaskRequest, AgentTaskResult, ApprovalRequest, TraceStep
from app.tools import HospitalTools


ApprovalFactory = Callable[..., ApprovalRequest]


class DocumentationAgent:
    def __init__(self, tools: HospitalTools, create_approval: ApprovalFactory) -> None:
        self.tools = tools
        self.create_approval = create_approval

    def handle_task(self, task: AgentTaskRequest) -> AgentTaskResult:
        patient_tool = self.tools.get_patient_record(task.patient_id)
        note = self.tools.draft_soap_note(task.message, patient_tool.result)
        tool = self.tools.finalize_soap_note(task.patient_id, note)
        approval = self.create_approval(
            session_id=task.session_id,
            nurse_id=task.nurse_id,
            patient_id=task.patient_id,
            tool_name=tool.name,
            reason=tool.reason,
            payload=tool.result,
            agent_id=DOCUMENTATION_AGENT_ID,
            task_id=task.task_id,
        )
        return AgentTaskResult(
            task_id=task.task_id,
            requester_agent_id=task.requester_agent_id,
            responder_agent_id=DOCUMENTATION_AGENT_ID,
            status="needs_approval",
            summary="SOAP note draft is ready and awaiting approval.",
            data={"draft_note": note, "request": tool.result},
            approval=approval,
            trace=[
                TraceStep(
                    kind="approval",
                    title="Documentation agent approval",
                    detail=f"Drafted SOAP note and paused finalization for approval {approval.approval_id}.",
                    agent_id=DOCUMENTATION_AGENT_ID,
                )
            ],
        )
