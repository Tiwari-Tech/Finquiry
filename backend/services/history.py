
import json
import sqlite3
import uuid

from datetime import datetime, timezone

from backend.services.auth_service import (
    get_db_connection,
    normalize_user_id,
)


# --------------------------------------------------
# Database Initialization
# --------------------------------------------------

def initialize_history_database():
    """Create conversation history tables."""

    with get_db_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL,
                company TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(conversation_id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user
            ON conversations(user_id)
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
            ON messages(conversation_id)
        """)

        # Enable foreign key enforcement.
        connection.execute("PRAGMA foreign_keys = ON")


initialize_history_database()


# --------------------------------------------------
# Create Conversation
# --------------------------------------------------

def create_conversation(
    user_id: str,
    question: str,
    company: str,
) -> str:

    user_id = normalize_user_id(user_id)

    conversation_id = str(uuid.uuid4())

    now = datetime.now(timezone.utc).isoformat()

    title = question.strip()[:80]

    with get_db_connection() as connection:

        connection.execute(
            """
            INSERT INTO conversations (
                conversation_id,
                user_id,
                title,
                company,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                user_id,
                title,
                company,
                now,
                now,
            ),
        )

    return conversation_id


# --------------------------------------------------
# Verify Conversation Ownership
# --------------------------------------------------

def get_conversation(
    conversation_id: str,
    user_id: str,
):

    user_id = normalize_user_id(user_id)

    with get_db_connection() as connection:

        return connection.execute(
            """
            SELECT *
            FROM conversations
            WHERE conversation_id = ?
            AND user_id = ?
            """,
            (conversation_id, user_id),
        ).fetchone()


# --------------------------------------------------
# Save Message
# --------------------------------------------------

def save_message(
    conversation_id: str,
    user_id: str,
    role: str,
    content: str,
    sources: list | None = None,
):

    user_id = normalize_user_id(user_id)

    if role not in ("user", "assistant"):
        raise ValueError("Invalid message role.")

    now = datetime.now(timezone.utc).isoformat()

    with get_db_connection() as connection:

        conversation = connection.execute(
            """
            SELECT conversation_id
            FROM conversations
            WHERE conversation_id = ?
            AND user_id = ?
            """,
            (conversation_id, user_id),
        ).fetchone()

        if conversation is None:
            raise PermissionError(
                "Conversation not found or access denied."
            )

        connection.execute(
            """
            INSERT INTO messages (
                message_id,
                conversation_id,
                role,
                content,
                sources,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                conversation_id,
                role,
                content,
                json.dumps(sources or []),
                now,
            ),
        )

        connection.execute(
            """
            UPDATE conversations
            SET updated_at = ?
            WHERE conversation_id = ?
            AND user_id = ?
            """,
            (now, conversation_id, user_id),
        )


# --------------------------------------------------
# Get Conversation Messages
# --------------------------------------------------

def get_conversation_messages(
    conversation_id: str,
    user_id: str,
):

    user_id = normalize_user_id(user_id)

    with get_db_connection() as connection:

        conversation = connection.execute(
            """
            SELECT conversation_id
            FROM conversations
            WHERE conversation_id = ?
            AND user_id = ?
            """,
            (conversation_id, user_id),
        ).fetchone()

        if conversation is None:
            raise PermissionError(
                "Conversation not found or access denied."
            )

        rows = connection.execute(
            """
            SELECT role, content, sources, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC, rowid ASC
            """,
            (conversation_id,),
        ).fetchall()

    messages = []

    for row in rows:

        messages.append({
            "role": row["role"],
            "content": row["content"],
            "sources": json.loads(row["sources"] or "[]"),
            "created_at": row["created_at"],
        })

    return messages


# --------------------------------------------------
# List User Conversations
# --------------------------------------------------

def list_conversations(user_id: str):

    user_id = normalize_user_id(user_id)

    with get_db_connection() as connection:

        rows = connection.execute(
            """
            SELECT
                conversation_id,
                title,
                company,
                created_at,
                updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()

    return [dict(row) for row in rows]


# --------------------------------------------------
# Delete Conversation
# --------------------------------------------------

def delete_conversation(
    conversation_id: str,
    user_id: str,
) -> bool:

    user_id = normalize_user_id(user_id)

    with get_db_connection() as connection:

        # Enable cascading deletion for this connection.
        connection.execute("PRAGMA foreign_keys = ON")

        cursor = connection.execute(
            """
            DELETE FROM conversations
            WHERE conversation_id = ?
            AND user_id = ?
            """,
            (conversation_id, user_id),
        )

        return cursor.rowcount > 0
