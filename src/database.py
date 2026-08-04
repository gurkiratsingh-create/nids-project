"""SQLite-backed incident storage for the network intrusion detection system."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class IncidentDatabase:
    """Manage SOC incidents stored in a local SQLite database."""

    def __init__(self, database_path: Optional[str | Path] = None) -> None:
        """Initialize the database connection and create the incidents table."""
        self.project_root = Path(__file__).resolve().parents[1]
        self.database_dir = self.project_root / "database"
        self.database_dir.mkdir(parents=True, exist_ok=True)

        self.database_path = Path(database_path) if database_path is not None else self.database_dir / "incidents.db"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

        self._initialize_database()

    def _initialize_database(self) -> None:
        """Create the incidents table if it does not already exist."""
        try:
            with self._get_connection() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS incidents (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        source_ip TEXT NOT NULL,
                        destination_ip TEXT NOT NULL,
                        attack_type TEXT NOT NULL,
                        risk_score REAL NOT NULL,
                        recommendation TEXT NOT NULL
                    )
                    """
                )
                connection.commit()
        except sqlite3.Error as exc:
            logger.exception("Failed to initialize the incidents table")
            raise RuntimeError("Unable to initialize the incidents database") from exc

    def _get_connection(self) -> sqlite3.Connection:
        """Create and return a SQLite database connection."""
        return sqlite3.connect(self.database_path)

    def insert_incident(
        self,
        timestamp: str,
        source_ip: str,
        destination_ip: str,
        attack_type: str,
        risk_score: float,
        recommendation: str,
    ) -> int:
        """Insert a new incident into the database and return its generated ID."""
        query = """
            INSERT INTO incidents (timestamp, source_ip, destination_ip, attack_type, risk_score, recommendation)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as connection:
                cursor = connection.execute(
                    query,
                    (timestamp, source_ip, destination_ip, attack_type, risk_score, recommendation),
                )
                connection.commit()
                return int(cursor.lastrowid)
        except sqlite3.Error as exc:
            logger.exception("Failed to insert incident")
            raise RuntimeError("Unable to insert incident into the database") from exc

    def get_all_incidents(self) -> list[dict[str, Any]]:
        """Retrieve all incidents from the database."""
        query = "SELECT id, timestamp, source_ip, destination_ip, attack_type, risk_score, recommendation FROM incidents ORDER BY id DESC"
        try:
            with self._get_connection() as connection:
                rows = connection.execute(query).fetchall()
        except sqlite3.Error as exc:
            logger.exception("Failed to retrieve incidents")
            raise RuntimeError("Unable to retrieve incidents from the database") from exc

        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "source_ip": row[2],
                "destination_ip": row[3],
                "attack_type": row[4],
                "risk_score": row[5],
                "recommendation": row[6],
            }
            for row in rows
        ]

    def get_incident_by_id(self, incident_id: int) -> Optional[dict[str, Any]]:
        """Retrieve a single incident by its ID."""
        query = "SELECT id, timestamp, source_ip, destination_ip, attack_type, risk_score, recommendation FROM incidents WHERE id = ?"
        try:
            with self._get_connection() as connection:
                row = connection.execute(query, (incident_id,)).fetchone()
        except sqlite3.Error as exc:
            logger.exception("Failed to retrieve incident with ID %s", incident_id)
            raise RuntimeError("Unable to retrieve incident by ID") from exc

        if row is None:
            return None

        return {
            "id": row[0],
            "timestamp": row[1],
            "source_ip": row[2],
            "destination_ip": row[3],
            "attack_type": row[4],
            "risk_score": row[5],
            "recommendation": row[6],
        }

    def delete_incident(self, incident_id: int) -> bool:
        """Delete an incident by ID and return whether it was removed."""
        query = "DELETE FROM incidents WHERE id = ?"
        try:
            with self._get_connection() as connection:
                cursor = connection.execute(query, (incident_id,))
                connection.commit()
                return cursor.rowcount > 0
        except sqlite3.Error as exc:
            logger.exception("Failed to delete incident with ID %s", incident_id)
            raise RuntimeError("Unable to delete incident") from exc

    def update_incident(
        self,
        incident_id: int,
        *,
        timestamp: Optional[str] = None,
        source_ip: Optional[str] = None,
        destination_ip: Optional[str] = None,
        attack_type: Optional[str] = None,
        risk_score: Optional[float] = None,
        recommendation: Optional[str] = None,
    ) -> bool:
        """Update one or more fields of an incident identified by ID."""
        fields: list[tuple[str, Any]] = []
        if timestamp is not None:
            fields.append(("timestamp", timestamp))
        if source_ip is not None:
            fields.append(("source_ip", source_ip))
        if destination_ip is not None:
            fields.append(("destination_ip", destination_ip))
        if attack_type is not None:
            fields.append(("attack_type", attack_type))
        if risk_score is not None:
            fields.append(("risk_score", risk_score))
        if recommendation is not None:
            fields.append(("recommendation", recommendation))

        if not fields:
            raise ValueError("At least one field must be provided for update")

        assignments = ", ".join(f"{column} = ?" for column, _ in fields)
        values = [value for _, value in fields] + [incident_id]
        query = f"UPDATE incidents SET {assignments} WHERE id = ?"

        try:
            with self._get_connection() as connection:
                cursor = connection.execute(query, values)
                connection.commit()
                return cursor.rowcount > 0
        except sqlite3.Error as exc:
            logger.exception("Failed to update incident with ID %s", incident_id)
            raise RuntimeError("Unable to update incident") from exc

    def count_incidents(self) -> int:
        """Return the total number of stored incidents."""
        query = "SELECT COUNT(*) FROM incidents"
        try:
            with self._get_connection() as connection:
                row = connection.execute(query).fetchone()
        except sqlite3.Error as exc:
            logger.exception("Failed to count incidents")
            raise RuntimeError("Unable to count incidents") from exc

        return int(row[0]) if row is not None else 0

    def get_high_risk_incidents(self, threshold: float = 0.8) -> list[dict[str, Any]]:
        """Retrieve incidents whose risk score meets or exceeds the specified threshold."""
        query = "SELECT id, timestamp, source_ip, destination_ip, attack_type, risk_score, recommendation FROM incidents WHERE risk_score >= ? ORDER BY risk_score DESC"
        try:
            with self._get_connection() as connection:
                rows = connection.execute(query, (threshold,)).fetchall()
        except sqlite3.Error as exc:
            logger.exception("Failed to retrieve high-risk incidents")
            raise RuntimeError("Unable to retrieve high-risk incidents") from exc

        return [
            {
                "id": row[0],
                "timestamp": row[1],
                "source_ip": row[2],
                "destination_ip": row[3],
                "attack_type": row[4],
                "risk_score": row[5],
                "recommendation": row[6],
            }
            for row in rows
        ]
