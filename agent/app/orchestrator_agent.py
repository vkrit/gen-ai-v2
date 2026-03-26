from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.agent_registry import (
    DOCUMENTATION_AGENT_ID,
    ORCHESTRATOR_AGENT_ID,
    PHARMACY_AGENT_ID,
    SCHEDULING_AGENT_ID,
)
from app.memory import MemoryStore
from app.schemas import AgentTaskRequest, AgentTaskResult, TraceStep
from app.tools import HospitalTools, ToolExecution

HOSPITAL_SYSTEM_PROMPT = """You are a hospital operations AI assistant.
Follow the Perceive -> Reason -> Act -> Observe loop.
Use read tools freely when they help. Never execute write actions without approval.
Summaries should be operational, concise, and explicit about risk."""

TaskDispatcher = Callable[[AgentTaskRequest], AgentTaskResult]


class OrchestratorAgent:
    def __init__(self, memory: MemoryStore, tools: HospitalTools, dispatch_task: TaskDispatcher) -> None:
        self.memory = memory
        self.tools = tools
        self.dispatch_task = dispatch_task

    def _build_context(
        self, session_id: str, patient_id: str, nurse_query: str, nurse_id: str
    ) -> tuple[str, TraceStep]:
        system = HOSPITAL_SYSTEM_PROMPT
        protocols = self.memory.search_protocols(nurse_query, top_k=2)
        if protocols:
            protocol_text = "\n".join(f"- {item['title']}: {item['content']}" for item in protocols)
            system += f"\n\nRelated hospital protocols:\n{protocol_text}"

        history = self.memory.get_patient_context(patient_id, limit=3)
        system += f"\n\n{history}"

        pref = self.memory.get_nurse_preference(nurse_id)
        if pref:
            system += f"\n\nNurse preference: {pref}"

        prior_messages = self.memory.get_session_messages(session_id=session_id, limit=4)
        if prior_messages:
            system += "\n\nPrior session context loaded."

        return system, TraceStep(
            kind="memory",
            title="Memory injection",
            detail="Combined procedural prompt, semantic protocols, episodic history, and nurse preference.",
            agent_id=ORCHESTRATOR_AGENT_ID,
        )

    def route(self, message: str) -> tuple[list[str], TraceStep]:
        lowered = message.lower()
        routes = [ORCHESTRATOR_AGENT_ID]
        if any(term in lowered for term in ("med", "drug", "interaction", "allergy", "warfarin", "aspirin")):
            routes.append(PHARMACY_AGENT_ID)
        if any(term in lowered for term in ("schedule", "appointment", "book")):
            routes.append(SCHEDULING_AGENT_ID)
        if any(term in lowered for term in ("soap", "note", "document", "documentation")):
            routes.append(DOCUMENTATION_AGENT_ID)
        return routes, TraceStep(
            kind="route",
            title="Specialist routing",
            detail=f"Selected specialists: {', '.join(routes)}.",
            agent_id=ORCHESTRATOR_AGENT_ID,
        )

    def chat(self, *, session_id: str, nurse_id: str, patient_id: str, message: str) -> dict[str, Any]:
        trace: list[TraceStep] = []
        system_context, memory_step = self._build_context(session_id, patient_id, message, nurse_id)
        trace.append(memory_step)

        routes, route_step = self.route(message)
        trace.append(route_step)

        tool_outputs: list[ToolExecution] = []
        patient_tool = self.tools.get_patient_record(patient_id)
        tool_outputs.append(patient_tool)
        trace.append(
            TraceStep(
                kind="tool",
                title="Read tool",
                detail=f"Executed get_patient_record for {patient_id}.",
                agent_id=ORCHESTRATOR_AGENT_ID,
            )
        )

        protocols_tool = self.tools.search_protocols(message)
        tool_outputs.append(protocols_tool)
        trace.append(
            TraceStep(
                kind="tool",
                title="Knowledge retrieval",
                detail="Searched semantic memory for relevant protocols.",
                agent_id=ORCHESTRATOR_AGENT_ID,
            )
        )

        lowered = message.lower()
        if PHARMACY_AGENT_ID in routes or any(
            term in lowered for term in ("drug", "med", "interaction", "aspirin", "ibuprofen")
        ):
            pharmacy_result = self.dispatch_task(
                AgentTaskRequest(
                    requester_agent_id=ORCHESTRATOR_AGENT_ID,
                    target_agent_id=PHARMACY_AGENT_ID,
                    session_id=session_id,
                    nurse_id=nurse_id,
                    patient_id=patient_id,
                    intent="medication_review",
                    message=message,
                    context={"candidate_drug": self.tools.candidate_drug_from_message(message)},
                )
            )
            tool_outputs.append(
                ToolExecution(name="check_drug_interaction", result=pharmacy_result.data["interaction_review"])
            )
            trace.extend(pharmacy_result.trace)

        approval_messages: list[str] = []

        if SCHEDULING_AGENT_ID in routes:
            scheduling_result = self.dispatch_task(
                AgentTaskRequest(
                    requester_agent_id=ORCHESTRATOR_AGENT_ID,
                    target_agent_id=SCHEDULING_AGENT_ID,
                    session_id=session_id,
                    nurse_id=nurse_id,
                    patient_id=patient_id,
                    intent="appointment_request",
                    message=message,
                    context={"specialty": self.tools.specialty_from_message(message)},
                )
            )
            tool_outputs.append(
                ToolExecution(
                    name="schedule_appointment",
                    result=scheduling_result.data["request"],
                    approval_required=True,
                    reason=scheduling_result.summary,
                )
            )
            approval_messages.append(scheduling_result.summary)
            trace.extend(scheduling_result.trace)

        if DOCUMENTATION_AGENT_ID in routes:
            documentation_result = self.dispatch_task(
                AgentTaskRequest(
                    requester_agent_id=ORCHESTRATOR_AGENT_ID,
                    target_agent_id=DOCUMENTATION_AGENT_ID,
                    session_id=session_id,
                    nurse_id=nurse_id,
                    patient_id=patient_id,
                    intent="documentation_request",
                    message=message,
                )
            )
            tool_outputs.append(
                ToolExecution(
                    name="finalize_soap_note",
                    result=documentation_result.data["request"],
                    approval_required=True,
                    reason=documentation_result.summary,
                )
            )
            approval_messages.append(documentation_result.summary)
            trace.extend(documentation_result.trace)

        reply = self._compose_reply(
            patient_id=patient_id,
            nurse_query=message,
            tool_outputs=tool_outputs,
            approval_messages=approval_messages,
            system_context=system_context,
        )
        trace.append(
            TraceStep(
                kind="response",
                title="Final response",
                detail="Generated a clinician-facing operational summary.",
                agent_id=ORCHESTRATOR_AGENT_ID,
            )
        )
        return {"reply": reply, "trace": trace}

    def _compose_reply(
        self,
        *,
        patient_id: str,
        nurse_query: str,
        tool_outputs: list[ToolExecution],
        approval_messages: list[str],
        system_context: str,
    ) -> str:
        patient_result = next(
            (tool.result for tool in tool_outputs if tool.name == "get_patient_record"),
            {},
        )
        protocols_result = next(
            (tool.result for tool in tool_outputs if tool.name == "search_protocols"),
            {"protocols": []},
        )
        interaction_result = next(
            (tool.result for tool in tool_outputs if tool.name == "check_drug_interaction"),
            {"interactions": []},
        )

        lines = [
            f"Patient: {patient_result.get('name', patient_id)} ({patient_id})",
            f"Current meds: {', '.join(patient_result.get('medications', ['Unknown']))}",
            f"Allergies: {', '.join(patient_result.get('allergies', ['None recorded']))}",
        ]

        if interaction_result.get("interactions"):
            top = interaction_result["interactions"][0]
            lines.append(
                f"Medication review: {top['severity']} risk for {top['candidate_drug']} with {top['current_drug']}. {top['effect']}"
            )

        protocols = protocols_result.get("protocols", [])
        if protocols:
            lines.append("Relevant protocols: " + "; ".join(protocol["title"] for protocol in protocols))

        if approval_messages:
            lines.extend(approval_messages)

        if system_context:
            lines.append(f"Request summary: {nurse_query}")

        return "\n".join(lines)
