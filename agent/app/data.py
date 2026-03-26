from __future__ import annotations

PATIENTS = [
    {
        "patient_id": "HN-2024-00123",
        "name": "Somchai Prasert",
        "age": 67,
        "gender": "Male",
        "conditions": ["Atrial fibrillation", "Hypertension"],
        "allergies": ["Penicillin"],
        "medications": ["Warfarin", "Amlodipine"],
        "last_visit": "2026-03-10",
    },
    {
        "patient_id": "HN-2024-00456",
        "name": "Anong Srisuk",
        "age": 42,
        "gender": "Female",
        "conditions": ["Type 2 diabetes"],
        "allergies": ["Sulfa drugs"],
        "medications": ["Metformin"],
        "last_visit": "2026-03-18",
    },
    {
        "patient_id": "HN-2024-00789",
        "name": "Krit Tansakul",
        "age": 29,
        "gender": "Male",
        "conditions": ["Asthma"],
        "allergies": [],
        "medications": ["Salbutamol inhaler"],
        "last_visit": "2026-03-22",
    },
]

PROTOCOLS = [
    {
        "title": "Medication safety escalation",
        "tags": ["medication", "drug", "interaction", "allergy", "pharmacy"],
        "content": (
            "Always verify current medication list, allergy history, and high-risk "
            "anticoagulants before suggesting a new medication. Escalate severe "
            "interaction risk to the prescriber before administration."
        ),
    },
    {
        "title": "Appointment scheduling workflow",
        "tags": ["appointment", "schedule", "follow-up", "clinic"],
        "content": (
            "Confirm patient identity, specialty, urgency, and preferred date range. "
            "Any booking action must be reviewed by a nurse before final submission."
        ),
    },
    {
        "title": "SOAP note drafting guidance",
        "tags": ["soap", "documentation", "note", "ehr"],
        "content": (
            "Draft SOAP notes in a concise professional tone with clear subjective, "
            "objective, assessment, and plan sections. Final save requires human sign-off."
        ),
    },
    {
        "title": "General triage support",
        "tags": ["triage", "symptom", "assessment", "routine"],
        "content": (
            "Provide general operational support, summarize patient context, and "
            "route medication questions to pharmacy review or scheduling requests "
            "to intake workflow."
        ),
    },
]

DRUG_INTERACTIONS = [
    {
        "drugs": {"Warfarin", "Aspirin"},
        "severity": "HIGH",
        "effect": "Elevated bleeding risk. Requires physician review before co-administration.",
    },
    {
        "drugs": {"Warfarin", "Ibuprofen"},
        "severity": "HIGH",
        "effect": "Bleeding risk and INR instability. Avoid unless prescriber explicitly approves.",
    },
    {
        "drugs": {"Metformin", "Prednisolone"},
        "severity": "MEDIUM",
        "effect": "Steroids may worsen glucose control. Monitor blood sugar more closely.",
    },
]

NURSE_PREFERENCES = {
    "nurse-demo": "Prefers concise action-first summaries with bullet style recommendations.",
}
