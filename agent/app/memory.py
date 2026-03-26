from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from app.data import NURSE_PREFERENCES, PROTOCOLS


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
