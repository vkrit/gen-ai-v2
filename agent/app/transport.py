from __future__ import annotations

import json
import os
from collections.abc import Callable
from typing import Any
from urllib import error, request

from app.schemas import AgentCard, AgentTaskRequest, AgentTaskResult

TaskHandler = Callable[[AgentTaskRequest], AgentTaskResult]
CardResolver = Callable[[str], AgentCard]


class A2ATransport:
    def __init__(self, get_card: CardResolver) -> None:
        self.get_card = get_card
        self.local_handlers: dict[str, TaskHandler] = {}
        self.mode = os.getenv("A2A_TRANSPORT_MODE", "local").strip().lower()
        self.base_url = os.getenv("A2A_BASE_URL", "").rstrip("/")

    def register_local_handler(self, agent_id: str, handler: TaskHandler) -> None:
        self.local_handlers[agent_id] = handler

    def submit_task(self, task: AgentTaskRequest) -> AgentTaskResult:
        card = self.get_card(task.target_agent_id)
        if self.mode != "http" and task.target_agent_id in self.local_handlers:
            return self.local_handlers[task.target_agent_id](task)
        return self._submit_http(card, task)

    def _submit_http(self, card: AgentCard, task: AgentTaskRequest) -> AgentTaskResult:
        if not self.base_url:
            raise RuntimeError(
                "A2A transport is set to HTTP, but A2A_BASE_URL is not configured and no local handler is available."
            )

        target_url = f"{self.base_url}{card.endpoint}"
        body = json.dumps(task.model_dump()).encode("utf-8")
        req = request.Request(
            target_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=10) as response:
                payload: Any = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:  # pragma: no cover
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"A2A HTTP request failed for {card.agent_id}: {exc.code} {detail}") from exc
        except error.URLError as exc:  # pragma: no cover
            raise RuntimeError(f"A2A HTTP request failed for {card.agent_id}: {exc.reason}") from exc

        return AgentTaskResult.model_validate(payload)
