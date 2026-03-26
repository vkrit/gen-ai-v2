from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from app.data import NURSE_PREFERENCES, PROTOCOLS
from app.schemas import AgentTaskRequest, AgentTaskResult, ApprovalRequest, TraceStep


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_hash TEXT NOT NULL,
                    nurse_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nurse_id TEXT NOT NULL,
                    patient_hash TEXT NOT NULL,
                    action TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    approval_status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    nurse_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    patient_hash TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    status TEXT NOT NULL,
                    agent_id TEXT,
                    task_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS agent_tasks (
                    task_id TEXT PRIMARY KEY,
                    requester_agent_id TEXT NOT NULL,
                    target_agent_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    nurse_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    patient_hash TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    message TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '',
                    result_data_json TEXT NOT NULL DEFAULT '{}',
                    trace_json TEXT NOT NULL DEFAULT '[]',
                    approval_id TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    nurse_id TEXT NOT NULL,
                    patient_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    @staticmethod
    def hash_patient(patient_id: str) -> str:
        return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()

    def get_patient_context(self, patient_id: str, limit: int = 3) -> str:
        patient_hash = self.hash_patient(patient_id)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT created_at, summary
                FROM episodic_memory
                WHERE patient_hash = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (patient_hash, limit),
            ).fetchall()
        if not rows:
            return "No prior episodic memory for this patient."
        lines = [f"- {row['created_at'][:10]}: {row['summary']}" for row in rows]
        return "Recent episodic memory:\n" + "\n".join(lines)

    def get_nurse_preference(self, nurse_id: str) -> str | None:
        return NURSE_PREFERENCES.get(nurse_id)

    def save_episode(self, patient_id: str, nurse_id: str, summary: str) -> None:
        patient_hash = self.hash_patient(patient_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO episodic_memory (patient_hash, nurse_id, summary)
                VALUES (?, ?, ?)
                """,
                (patient_hash, nurse_id, summary),
            )

    def save_session_message(
        self,
        *,
        session_id: str,
        nurse_id: str,
        patient_id: str,
        role: str,
        content: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_messages (session_id, nurse_id, patient_id, role, content)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, nurse_id, patient_id, role, content),
            )

    def get_session_messages(self, session_id: str, limit: int = 6) -> list[dict[str, str]]:
        if limit <= 0:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, created_at
                FROM session_messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        messages = [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]
        return messages

    def log_action(
        self,
        *,
        nurse_id: str,
        patient_id: str,
        action: str,
        detail: dict[str, Any],
        approval_status: str,
    ) -> None:
        patient_hash = self.hash_patient(patient_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_log (nurse_id, patient_hash, action, detail, approval_status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    nurse_id,
                    patient_hash,
                    action,
                    json.dumps(detail, ensure_ascii=True),
                    approval_status,
                ),
            )

    def recent_audit_log(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT created_at, nurse_id, patient_hash, action, detail, approval_status
                FROM audit_log
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "created_at": row["created_at"],
                "nurse_id": row["nurse_id"],
                "patient_hash": row["patient_hash"][:12] + "...",
                "action": row["action"],
                "detail": json.loads(row["detail"]),
                "approval_status": row["approval_status"],
            }
            for row in rows
        ]

    def save_approval(self, approval: ApprovalRequest) -> None:
        patient_hash = self.hash_patient(approval.patient_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO approvals (
                    approval_id, session_id, nurse_id, patient_id, patient_hash,
                    tool_name, reason, payload, status, agent_id, task_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(approval_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    nurse_id = excluded.nurse_id,
                    patient_id = excluded.patient_id,
                    patient_hash = excluded.patient_hash,
                    tool_name = excluded.tool_name,
                    reason = excluded.reason,
                    payload = excluded.payload,
                    status = excluded.status,
                    agent_id = excluded.agent_id,
                    task_id = excluded.task_id,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    approval.approval_id,
                    approval.session_id,
                    approval.nurse_id,
                    approval.patient_id,
                    patient_hash,
                    approval.tool_name,
                    approval.reason,
                    json.dumps(approval.payload, ensure_ascii=True),
                    approval.status,
                    approval.agent_id,
                    approval.task_id,
                ),
            )

    def get_approval(self, approval_id: str) -> ApprovalRequest | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT approval_id, session_id, nurse_id, patient_id, tool_name, reason,
                       payload, status, agent_id, task_id
                FROM approvals
                WHERE approval_id = ?
                """,
                (approval_id,),
            ).fetchone()
        if row is None:
            return None
        return ApprovalRequest(
            approval_id=row["approval_id"],
            session_id=row["session_id"],
            nurse_id=row["nurse_id"],
            patient_id=row["patient_id"],
            tool_name=row["tool_name"],
            reason=row["reason"],
            payload=json.loads(row["payload"]),
            status=row["status"],
            agent_id=row["agent_id"],
            task_id=row["task_id"],
        )

    def list_approvals(self, status: str = "pending") -> list[ApprovalRequest]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT approval_id, session_id, nurse_id, patient_id, tool_name, reason,
                       payload, status, agent_id, task_id
                FROM approvals
                WHERE status = ?
                ORDER BY created_at DESC, approval_id DESC
                """,
                (status,),
            ).fetchall()
        return [
            ApprovalRequest(
                approval_id=row["approval_id"],
                session_id=row["session_id"],
                nurse_id=row["nurse_id"],
                patient_id=row["patient_id"],
                tool_name=row["tool_name"],
                reason=row["reason"],
                payload=json.loads(row["payload"]),
                status=row["status"],
                agent_id=row["agent_id"],
                task_id=row["task_id"],
            )
            for row in rows
        ]

    def update_approval_status(self, approval_id: str, status: str) -> ApprovalRequest | None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE approvals
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE approval_id = ?
                """,
                (status, approval_id),
            )
        return self.get_approval(approval_id)

    def record_agent_task(self, task: AgentTaskRequest) -> None:
        patient_hash = self.hash_patient(task.patient_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_tasks (
                    task_id, requester_agent_id, target_agent_id, session_id,
                    nurse_id, patient_id, patient_hash, intent, message,
                    context_json, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    requester_agent_id = excluded.requester_agent_id,
                    target_agent_id = excluded.target_agent_id,
                    session_id = excluded.session_id,
                    nurse_id = excluded.nurse_id,
                    patient_id = excluded.patient_id,
                    patient_hash = excluded.patient_hash,
                    intent = excluded.intent,
                    message = excluded.message,
                    context_json = excluded.context_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    task.task_id,
                    task.requester_agent_id,
                    task.target_agent_id,
                    task.session_id,
                    task.nurse_id,
                    task.patient_id,
                    patient_hash,
                    task.intent,
                    task.message,
                    json.dumps(task.context, ensure_ascii=True),
                    "received",
                ),
            )

    def save_agent_task_result(self, result: AgentTaskResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agent_tasks
                SET status = ?, summary = ?, result_data_json = ?, trace_json = ?,
                    approval_id = ?, updated_at = CURRENT_TIMESTAMP
                WHERE task_id = ?
                """,
                (
                    result.status,
                    result.summary,
                    json.dumps(result.data, ensure_ascii=True),
                    json.dumps([step.model_dump() for step in result.trace], ensure_ascii=True),
                    result.approval.approval_id if result.approval else None,
                    result.task_id,
                ),
            )

    def update_task_status_from_approval(self, task_id: str | None, decision: str) -> None:
        if not task_id:
            return
        task_status = "completed" if decision == "approved" else "rejected"
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agent_tasks
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE task_id = ?
                """,
                (task_status, task_id),
            )

    def list_agent_tasks(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT task_id, requester_agent_id, target_agent_id, session_id, nurse_id,
                       patient_id, intent, message, status, summary, approval_id, created_at
                FROM agent_tasks
                ORDER BY created_at DESC, task_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "task_id": row["task_id"],
                "requester_agent_id": row["requester_agent_id"],
                "target_agent_id": row["target_agent_id"],
                "session_id": row["session_id"],
                "nurse_id": row["nurse_id"],
                "patient_id": row["patient_id"],
                "intent": row["intent"],
                "message": row["message"],
                "status": row["status"],
                "summary": row["summary"],
                "approval_id": row["approval_id"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def search_protocols(self, query: str, top_k: int = 2) -> list[dict[str, str]]:
        query_terms = {term.lower() for term in query.replace("/", " ").split()}
        scored: list[tuple[int, dict[str, str]]] = []
        for protocol in PROTOCOLS:
            haystack = {term.lower() for term in protocol["tags"]} | {
                term.lower() for term in protocol["content"].split()
            }
            score = len(query_terms & haystack)
            scored.append((score, protocol))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:top_k] if item[0] > 0]
