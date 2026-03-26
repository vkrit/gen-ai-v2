from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.data import DRUG_INTERACTIONS, PATIENTS
from app.memory import MemoryStore
from app.schemas import ApprovalRequest, TraceStep


HOSPITAL_SYSTEM_PROMPT = """You are a hospital operations AI assistant.
Follow the Perceive -> Reason -> Act -> Observe loop.
Use read tools freely when they help. Never execute write actions without approval.
Summaries should be operational, concise, and explicit about risk."""


@dataclass
class ToolExecution:
    name: str
    result: dict[str, Any]
    approval_required: bool = False
    reason: str = ""


class HospitalAgentRuntime:
    def __init__(self, db_path: Path) -> None:
        self.memory = MemoryStore(db_path)
        self.sessions: dict[str, list[dict[str, str]]] = {}
        self.pending_approvals: dict[str, ApprovalRequest] = {}
        self.patient_lookup = {patient["patient_id"]: patient for patient in PATIENTS}

    def bootstrap(self) -> dict[str, Any]:
        return {
            "patients": PATIENTS,
            "approvals": self.list_approvals(),
            "audit_log": self.memory.recent_audit_log(),
        }

    def list_approvals(self) -> list[ApprovalRequest]:
        approvals = list(self.pending_approvals.values())
        approvals.sort(key=lambda approval: approval.status)
        return approvals

    def _build_context(self, patient_id: str, nurse_query: str, nurse_id: str) -> tuple[str, TraceStep]:
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
        return system, TraceStep(
            kind="memory",
            title="Memory injection",
            detail="Combined procedural prompt, semantic protocols, episodic history, and nurse preference.",
        )

    def _route(self, message: str) -> tuple[list[str], TraceStep]:
        lowered = message.lower()
        routes = ["orchestrator"]
        if any(term in lowered for term in ("med", "drug", "interaction", "allergy", "warfarin", "aspirin")):
            routes.append("pharmacy")
        if any(term in lowered for term in ("schedule", "appointment", "book")):
            routes.append("intake")
        if any(term in lowered for term in ("soap", "note", "document", "documentation")):
            routes.append("documentation")
        return routes, TraceStep(
            kind="route",
            title="Specialist routing",
            detail=f"Selected specialists: {', '.join(routes)}.",
        )

    def _tool_get_patient_record(self, patient_id: str) -> ToolExecution:
        patient = self.patient_lookup.get(patient_id)
        if not patient:
            return ToolExecution(name="get_patient_record", result={"error": "Patient not found."})
        return ToolExecution(name="get_patient_record", result=patient)

    def _tool_search_protocols(self, query: str) -> ToolExecution:
        protocols = self.memory.search_protocols(query, top_k=2)
        return ToolExecution(name="search_protocols", result={"protocols": protocols})

    def _tool_check_drug_interaction(self, patient_id: str, candidate_drug: str) -> ToolExecution:
        patient = self.patient_lookup.get(patient_id)
        if not patient:
            return ToolExecution(name="check_drug_interaction", result={"error": "Patient not found."})
        findings = []
        for current_drug in patient["medications"]:
            pair = {current_drug.lower(), candidate_drug.lower()}
            for interaction in DRUG_INTERACTIONS:
                if pair == {item.lower() for item in interaction["drugs"]}:
                    findings.append(
                        {
                            "current_drug": current_drug,
                            "candidate_drug": candidate_drug,
                            "severity": interaction["severity"],
                            "effect": interaction["effect"],
                        }
                    )
        if not findings:
            findings.append(
                {
                    "current_drug": ", ".join(patient["medications"]),
                    "candidate_drug": candidate_drug,
                    "severity": "LOW",
                    "effect": "No known interaction in the local formulary reference.",
                }
            )
        return ToolExecution(name="check_drug_interaction", result={"interactions": findings})

    def _tool_schedule_appointment(
        self, patient_id: str, nurse_id: str, session_id: str, specialty: str
    ) -> ToolExecution:
        payload = {"patient_id": patient_id, "specialty": specialty}
        return ToolExecution(
            name="schedule_appointment",
            result=payload,
            approval_required=True,
            reason=f"Schedule a {specialty} appointment for {patient_id}.",
        )

    def _tool_finalize_soap_note(
        self, patient_id: str, nurse_id: str, session_id: str, note: str
    ) -> ToolExecution:
        payload = {"patient_id": patient_id, "draft_note": note}
        return ToolExecution(
            name="finalize_soap_note",
            result=payload,
            approval_required=True,
            reason=f"Finalize SOAP note for {patient_id}.",
        )

    def _candidate_drug_from_message(self, message: str) -> str:
        tokens = message.replace(",", " ").split()
        for token in tokens:
            clean = token.strip(".?!").title()
            if clean.lower() in {"aspirin", "ibuprofen", "prednisolone", "amoxicillin"}:
                return clean
        return "Aspirin"

    def _specialty_from_message(self, message: str) -> str:
        lowered = message.lower()
        if "cardio" in lowered or "heart" in lowered:
            return "Cardiology"
        if "endocr" in lowered or "diabetes" in lowered:
            return "Endocrinology"
        return "General Medicine"

    def chat(self, *, session_id: str, nurse_id: str, patient_id: str, message: str) -> dict[str, Any]:
        session = self.sessions.setdefault(session_id, [])
        session.append({"role": "user", "content": message})

        trace: list[TraceStep] = []
        system_context, memory_step = self._build_context(patient_id, message, nurse_id)
        trace.append(memory_step)

        routes, route_step = self._route(message)
        trace.append(route_step)

        tool_outputs: list[ToolExecution] = []
        patient_tool = self._tool_get_patient_record(patient_id)
        tool_outputs.append(patient_tool)
        trace.append(
            TraceStep(
                kind="tool",
                title="Read tool",
                detail=f"Executed get_patient_record for {patient_id}.",
            )
        )

        protocols_tool = self._tool_search_protocols(message)
        tool_outputs.append(protocols_tool)
        trace.append(
            TraceStep(
                kind="tool",
                title="Knowledge retrieval",
                detail="Searched semantic memory for relevant protocols.",
            )
        )

        lowered = message.lower()
        if "pharmacy" in routes or any(term in lowered for term in ("drug", "med", "interaction", "aspirin", "ibuprofen")):
            candidate_drug = self._candidate_drug_from_message(message)
            tool_outputs.append(self._tool_check_drug_interaction(patient_id, candidate_drug))
            trace.append(
                TraceStep(
                    kind="tool",
                    title="Pharmacy review",
                    detail=f"Checked interactions for candidate drug: {candidate_drug}.",
                )
            )

        approval_messages: list[str] = []
        if "intake" in routes:
            specialty = self._specialty_from_message(message)
            tool = self._tool_schedule_appointment(patient_id, nurse_id, session_id, specialty)
            tool_outputs.append(tool)
            approval = self._create_approval(
                session_id=session_id,
                nurse_id=nurse_id,
                patient_id=patient_id,
                tool_name=tool.name,
                reason=tool.reason,
                payload=tool.result,
            )
            approval_messages.append(
                f"Appointment booking is pending approval for {specialty}."
            )
            trace.append(
                TraceStep(
                    kind="approval",
                    title="Write action paused",
                    detail=f"{tool.name} was converted into approval request {approval.approval_id}.",
                )
            )

        if "documentation" in routes:
            note = self._draft_soap_note(message, patient_tool.result)
            tool = self._tool_finalize_soap_note(patient_id, nurse_id, session_id, note)
            tool_outputs.append(tool)
            approval = self._create_approval(
                session_id=session_id,
                nurse_id=nurse_id,
                patient_id=patient_id,
                tool_name=tool.name,
                reason=tool.reason,
                payload=tool.result,
            )
            approval_messages.append("SOAP note draft is ready and awaiting approval.")
            trace.append(
                TraceStep(
                    kind="approval",
                    title="Documentation checkpoint",
                    detail=f"{tool.name} was converted into approval request {approval.approval_id}.",
                )
            )

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
            )
        )

        self.memory.save_episode(
            patient_id,
            nurse_id,
            f"Query: {message[:80]} | Outcome: {reply[:120]}",
        )
        session.append({"role": "assistant", "content": reply})
        return {
            "session_id": session_id,
            "reply": reply,
            "trace": trace,
            "approvals": self.list_approvals(),
            "audit_log": self.memory.recent_audit_log(),
        }

    def _draft_soap_note(self, message: str, patient_record: dict[str, Any]) -> str:
        return (
            f"S: Patient request summarized from nurse note: {message}\n"
            f"O: Known conditions: {', '.join(patient_record.get('conditions', []))}. "
            f"Current medications: {', '.join(patient_record.get('medications', []))}.\n"
            "A: Requires clinician review before finalizing documentation.\n"
            "P: Review, confirm accuracy, and sign off if appropriate."
        )

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
            lines.append(
                "Relevant protocols: " + "; ".join(protocol["title"] for protocol in protocols)
            )

        if approval_messages:
            lines.extend(approval_messages)

        lines.append(f"Request summary: {nurse_query}")
        return "\n".join(lines)

    def _create_approval(
        self,
        *,
        session_id: str,
        nurse_id: str,
        patient_id: str,
        tool_name: str,
        reason: str,
        payload: dict[str, Any],
    ) -> ApprovalRequest:
        approval_id = f"APR-{uuid.uuid4().hex[:8].upper()}"
        approval = ApprovalRequest(
            approval_id=approval_id,
            session_id=session_id,
            nurse_id=nurse_id,
            patient_id=patient_id,
            tool_name=tool_name,
            reason=reason,
            payload=payload,
            status="pending",
        )
        self.pending_approvals[approval.approval_id] = approval
        self.memory.log_action(
            nurse_id=nurse_id,
            patient_id=patient_id,
            action=tool_name,
            detail=payload,
            approval_status="pending",
        )
        return approval

    def decide_approval(self, approval_id: str, nurse_id: str, decision: str) -> dict[str, Any]:
        approval = self.pending_approvals.get(approval_id)
        if not approval:
            raise KeyError(f"Approval {approval_id} was not found.")
        approval.status = decision
        self.memory.log_action(
            nurse_id=nurse_id,
            patient_id=approval.patient_id,
            action=approval.tool_name,
            detail=approval.payload,
            approval_status=decision,
        )
        if decision != "pending":
            del self.pending_approvals[approval_id]
        message = f"{approval.tool_name} {decision} for {approval.patient_id}."
        return {
            "message": message,
            "approvals": self.list_approvals(),
            "audit_log": self.memory.recent_audit_log(),
        }
