from __future__ import annotations

from collections.abc import Callable

from app.agent_registry import SCHEDULING_AGENT_ID
from app.schemas import AgentTaskRequest, AgentTaskResult, ApprovalRequest, TraceStep
from app.tools import HospitalTools


ApprovalFactory = Callable[..., ApprovalRequest]


class SchedulingAgent:
    def __init__(self, tools: HospitalTools, create_approval: ApprovalFactory) -> None:
        self.tools = tools
        self.create_approval = create_approval

    def handle_task(self, task: AgentTaskRequest) -> AgentTaskResult:
        specialty = str(task.context.get("specialty") or self.tools.specialty_from_message(task.message))
        tool = self.tools.schedule_appointment(task.patient_id, specialty)
        approval = self.create_approval(
            session_id=task.session_id,
            nurse_id=task.nurse_id,
            patient_id=task.patient_id,
            tool_name=tool.name,
            reason=tool.reason,
            payload=tool.result,
            agent_id=SCHEDULING_AGENT_ID,
            task_id=task.task_id,
        )
        return AgentTaskResult(
            task_id=task.task_id,
            requester_agent_id=task.requester_agent_id,
            responder_agent_id=SCHEDULING_AGENT_ID,
            status="needs_approval",
            summary=f"Appointment booking is pending approval for {specialty}.",
            data={"specialty": specialty, "request": tool.result},
            approval=approval,
            trace=[
                TraceStep(
                    kind="approval",
                    title="Scheduling agent approval",
                    detail=(
                        f"Prepared {specialty} appointment request and paused for approval "
                        f"{approval.approval_id}."
                    ),
                    agent_id=SCHEDULING_AGENT_ID,
                )
            ],
        )
