from __future__ import annotations

from app.agent_registry import PHARMACY_AGENT_ID
from app.schemas import AgentTaskRequest, AgentTaskResult, TraceStep
from app.tools import HospitalTools


class PharmacyAgent:
    def __init__(self, tools: HospitalTools) -> None:
        self.tools = tools

    def handle_task(self, task: AgentTaskRequest) -> AgentTaskResult:
        candidate_drug = str(
            task.context.get("candidate_drug") or self.tools.candidate_drug_from_message(task.message)
        )
        patient_tool = self.tools.get_patient_record(task.patient_id)
        interaction_tool = self.tools.check_drug_interaction(task.patient_id, candidate_drug)
        top = interaction_tool.result["interactions"][0]

        return AgentTaskResult(
            task_id=task.task_id,
            requester_agent_id=task.requester_agent_id,
            responder_agent_id=PHARMACY_AGENT_ID,
            status="completed",
            summary=(
                f"{top['severity']} interaction risk for {top['candidate_drug']} with "
                f"{top['current_drug']}. {top['effect']}"
            ),
            data={
                "patient": patient_tool.result,
                "interaction_review": interaction_tool.result,
                "candidate_drug": candidate_drug,
            },
            trace=[
                TraceStep(
                    kind="tool",
                    title="Pharmacy agent review",
                    detail=f"Reviewed {candidate_drug} for patient {task.patient_id}.",
                    agent_id=PHARMACY_AGENT_ID,
                )
            ],
        )
