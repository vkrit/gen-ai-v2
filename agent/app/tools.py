from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.data import DRUG_INTERACTIONS
from app.memory import MemoryStore


@dataclass
class ToolExecution:
    name: str
    result: dict[str, Any]
    approval_required: bool = False
    reason: str = ""


class HospitalTools:
    def __init__(self, memory: MemoryStore, patient_lookup: dict[str, dict[str, Any]]) -> None:
        self.memory = memory
        self.patient_lookup = patient_lookup

    def get_patient_record(self, patient_id: str) -> ToolExecution:
        patient = self.patient_lookup.get(patient_id)
        if not patient:
            return ToolExecution(name="get_patient_record", result={"error": "Patient not found."})
        return ToolExecution(name="get_patient_record", result=patient)

    def search_protocols(self, query: str) -> ToolExecution:
        protocols = self.memory.search_protocols(query, top_k=2)
        return ToolExecution(name="search_protocols", result={"protocols": protocols})

    def check_drug_interaction(self, patient_id: str, candidate_drug: str) -> ToolExecution:
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

    def schedule_appointment(self, patient_id: str, specialty: str) -> ToolExecution:
        payload = {"patient_id": patient_id, "specialty": specialty}
        return ToolExecution(
            name="schedule_appointment",
            result=payload,
            approval_required=True,
            reason=f"Schedule a {specialty} appointment for {patient_id}.",
        )

    def finalize_soap_note(self, patient_id: str, note: str) -> ToolExecution:
        payload = {"patient_id": patient_id, "draft_note": note}
        return ToolExecution(
            name="finalize_soap_note",
            result=payload,
            approval_required=True,
            reason=f"Finalize SOAP note for {patient_id}.",
        )

    @staticmethod
    def candidate_drug_from_message(message: str) -> str:
        tokens = message.replace(",", " ").split()
        for token in tokens:
            clean = token.strip(".?!").title()
            if clean.lower() in {"aspirin", "ibuprofen", "prednisolone", "amoxicillin"}:
                return clean
        return "Aspirin"

    @staticmethod
    def specialty_from_message(message: str) -> str:
        lowered = message.lower()
        if "cardio" in lowered or "heart" in lowered:
            return "Cardiology"
        if "endocr" in lowered or "diabetes" in lowered:
            return "Endocrinology"
        return "General Medicine"

    @staticmethod
    def draft_soap_note(message: str, patient_record: dict[str, Any]) -> str:
        return (
            f"S: Patient request summarized from nurse note: {message}\n"
            f"O: Known conditions: {', '.join(patient_record.get('conditions', []))}. "
            f"Current medications: {', '.join(patient_record.get('medications', []))}.\n"
            "A: Requires clinician review before finalizing documentation.\n"
            "P: Review, confirm accuracy, and sign off if appropriate."
        )
