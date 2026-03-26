from __future__ import annotations

from app.schemas import AgentCapability, AgentCard

APP_VERSION = "0.2.0"
ORCHESTRATOR_AGENT_ID = "orchestrator"
PHARMACY_AGENT_ID = "pharmacy-agent"
SCHEDULING_AGENT_ID = "scheduling-agent"
DOCUMENTATION_AGENT_ID = "documentation-agent"


def agent_mount_path(agent_id: str) -> str:
    return f"/agents/{agent_id}"


def agent_card_path(agent_id: str) -> str:
    return f"{agent_mount_path(agent_id)}/card"


def agent_task_path(agent_id: str) -> str:
    return f"{agent_mount_path(agent_id)}/tasks"


def build_agent_cards() -> dict[str, AgentCard]:
    cards = [
        AgentCard(
            agent_id=ORCHESTRATOR_AGENT_ID,
            name="Hospital Orchestrator",
            version=APP_VERSION,
            role="orchestrator",
            description=(
                "Receives nurse requests, loads memory, chooses specialists, and "
                "assembles the final operational response."
            ),
            card_path=agent_card_path(ORCHESTRATOR_AGENT_ID),
            endpoint=agent_task_path(ORCHESTRATOR_AGENT_ID),
            capabilities=[
                AgentCapability(
                    id="triage-routing",
                    name="Triage routing",
                    description="Routes work to pharmacy, scheduling, and documentation specialists.",
                ),
                AgentCapability(
                    id="response-composition",
                    name="Response composition",
                    description="Aggregates specialist outputs into one clinician-facing summary.",
                ),
            ],
            accepts=["nurse_query", "multi_domain_request"],
            returns=["final_summary", "trace", "approvals"],
            downstream_agents=[
                PHARMACY_AGENT_ID,
                SCHEDULING_AGENT_ID,
                DOCUMENTATION_AGENT_ID,
            ],
        ),
        AgentCard(
            agent_id=PHARMACY_AGENT_ID,
            name="Pharmacy Agent",
            version=APP_VERSION,
            role="specialist",
            description="Reviews medications, allergies, and local drug interaction risk.",
            card_path=agent_card_path(PHARMACY_AGENT_ID),
            endpoint=agent_task_path(PHARMACY_AGENT_ID),
            capabilities=[
                AgentCapability(
                    id="medication-review",
                    name="Medication review",
                    description="Checks candidate medications against the patient's current list.",
                )
            ],
            accepts=["medication_review"],
            returns=["interaction_summary", "interaction_data"],
        ),
        AgentCard(
            agent_id=SCHEDULING_AGENT_ID,
            name="Scheduling Agent",
            version=APP_VERSION,
            role="specialist",
            description="Prepares appointment booking requests and routes them to approval.",
            card_path=agent_card_path(SCHEDULING_AGENT_ID),
            endpoint=agent_task_path(SCHEDULING_AGENT_ID),
            capabilities=[
                AgentCapability(
                    id="appointment-planning",
                    name="Appointment planning",
                    description="Identifies specialty and prepares a scheduling request.",
                )
            ],
            accepts=["appointment_request"],
            returns=["scheduling_summary", "approval_request"],
            approval_required_actions=["schedule_appointment"],
        ),
        AgentCard(
            agent_id=DOCUMENTATION_AGENT_ID,
            name="Documentation Agent",
            version=APP_VERSION,
            role="specialist",
            description="Drafts SOAP documentation and routes finalization through approval.",
            card_path=agent_card_path(DOCUMENTATION_AGENT_ID),
            endpoint=agent_task_path(DOCUMENTATION_AGENT_ID),
            capabilities=[
                AgentCapability(
                    id="soap-drafting",
                    name="SOAP drafting",
                    description="Produces structured SOAP note drafts using patient context.",
                )
            ],
            accepts=["documentation_request"],
            returns=["soap_note_draft", "approval_request"],
            approval_required_actions=["finalize_soap_note"],
        ),
    ]
    return {card.agent_id: card for card in cards}
