from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI, HTTPException

from app.schemas import AgentCard, AgentTaskRequest, AgentTaskResult

TaskHandler = Callable[[AgentTaskRequest], AgentTaskResult]
CardGetter = Callable[[str], AgentCard]


def create_agent_service_app(
    *,
    agent_id: str,
    title: str,
    get_card: CardGetter,
    handle_task: TaskHandler,
) -> FastAPI:
    app = FastAPI(title=title, version="0.2.0")

    @app.get("/card", response_model=AgentCard)
    def card() -> AgentCard:
        return get_card(agent_id)

    @app.post("/tasks", response_model=AgentTaskResult)
    def task(payload: AgentTaskRequest) -> AgentTaskResult:
        if payload.target_agent_id != agent_id:
            raise HTTPException(
                status_code=400,
                detail=f"Task target_agent_id must be '{agent_id}' for this service.",
            )
        try:
            return handle_task(payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app
