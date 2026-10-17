"""MCP server backed by a persistent local SQLite ticket store."""

import os
import sqlite3
from datetime import date, timedelta
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

SERVER = FastMCP("Acme Support Tickets")
DATABASE = Path(os.getenv("TICKETS_DB", Path(__file__).resolve().parents[1] / "data/tickets.db"))


class TicketNote(BaseModel):
    note_id: int
    ticket_id: str
    body: str
    created_at: str


class Ticket(BaseModel):
    ticket_id: str
    subject: str
    account: str
    account_email: str
    status: str
    details: str
    invoice_id: str | None
    first_payment_date: str | None
    paid_feature_requests: int
    notes: list[TicketNote]
    today: str


def connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_store() -> None:
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    connection = connect()
    try:
        with connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id TEXT PRIMARY KEY,
                    subject TEXT NOT NULL,
                    account TEXT NOT NULL,
                    account_email TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT NOT NULL,
                    invoice_id TEXT,
                    first_payment_date TEXT,
                    paid_feature_requests INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS notes (
                    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id TEXT NOT NULL REFERENCES tickets(ticket_id),
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)
            connection.execute(
                "INSERT OR IGNORE INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "ACME-101",
                    "Annual refund and data export",
                    "Acme",
                    "admin@acme.example",
                    "open",
                    "Please refund our first annual payment and help us export our records.",
                    "INV-ACME-101",
                    (date.today() - timedelta(days=10)).isoformat(),
                    0,
                ),
            )
            connection.execute(
                "INSERT OR IGNORE INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "ACME-102",
                    "API requests return HTTP 429",
                    "Acme",
                    "developer@acme.example",
                    "open",
                    "Our integration receives HTTP 429. What should we do?",
                    None,
                    None,
                    12,
                ),
            )
    finally:
        connection.close()


@SERVER.tool()
def list_open_tickets(limit: int = 10) -> list[dict]:
    """List open local support tickets. The limit must be between 1 and 50."""
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    connection = connect()
    try:
        return [
            dict(row)
            for row in connection.execute(
                "SELECT ticket_id, subject, account FROM tickets "
                "WHERE status = 'open' ORDER BY ticket_id LIMIT ?",
                (limit,),
            )
        ]
    finally:
        connection.close()


@SERVER.tool()
def get_ticket(ticket_id: str) -> Ticket:
    """Read a local support ticket, its invoice and usage facts, and persisted notes."""
    connection = connect()
    try:
        ticket = connection.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?",
            (ticket_id,),
        ).fetchone()
        if ticket is None:
            raise ValueError(f"Ticket not found: {ticket_id}")
        result = dict(ticket)
        result["notes"] = [
            dict(row)
            for row in connection.execute(
                "SELECT note_id, ticket_id, body, created_at FROM notes "
                "WHERE ticket_id = ? ORDER BY note_id",
                (ticket_id,),
            )
        ]
        result["today"] = date.today().isoformat()
        return Ticket.model_validate(result)
    finally:
        connection.close()


@SERVER.tool()
def add_ticket_note(ticket_id: str, body: str) -> TicketNote:
    """Append an internal note to a local ticket. Does not contact anyone or approve refunds."""
    if not body.strip() or len(body) > 20_000:
        raise ValueError("Note must contain between 1 and 20000 characters")
    connection = connect()
    try:
        with connection:
            if (
                connection.execute(
                    "SELECT 1 FROM tickets WHERE ticket_id = ?",
                    (ticket_id,),
                ).fetchone()
                is None
            ):
                raise ValueError(f"Ticket not found: {ticket_id}")
            cursor = connection.execute(
                "INSERT INTO notes (ticket_id, body) VALUES (?, ?)",
                (ticket_id, body),
            )
            return TicketNote.model_validate(
                dict(
                    connection.execute(
                        "SELECT note_id, ticket_id, body, created_at FROM notes WHERE note_id = ?",
                        (cursor.lastrowid,),
                    ).fetchone()
                )
            )
    finally:
        connection.close()


if __name__ == "__main__":
    initialize_store()
    SERVER.run(transport="stdio")
