
import os
import re
import sqlite3
import hashlib
import hmac
import secrets
import uuid

from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from jose import JWTError, jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

load_dotenv(BASE_DIR / ".env")

JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")

if not JWT_SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY is missing from environment variables."
    )

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

DATABASE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "users.db"
)

security = HTTPBearer()


# --------------------------------------------------
# Database
# --------------------------------------------------

def get_db_connection():
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_auth_database():
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


initialize_auth_database()


# --------------------------------------------------
# User Validation
# --------------------------------------------------

def normalize_email(email: str) -> str:
    if not isinstance(email, str):
        raise ValueError("Email must be a string.")

    email = email.strip().lower()

    if not email or not re.fullmatch(
        r"[^@\s]+@[^@\s]+\.[^@\s]+",
        email,
    ):
        raise ValueError("Enter a valid email address.")

    return email


def normalize_user_id(user_id: str) -> str:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("Invalid user ID.")

    return user_id.strip().lower()


# --------------------------------------------------
# Password Hashing
# --------------------------------------------------

def hash_password(password: str) -> str:
    if not isinstance(password, str) or len(password) < 8:
        raise ValueError(
            "Password must contain at least 8 characters."
        )

    salt = secrets.token_bytes(16)

    password_hash = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=32,
    )

    return f"{salt.hex()}:{password_hash.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, hash_hex = stored_hash.split(":")

        salt = bytes.fromhex(salt_hex)
        expected_hash = bytes.fromhex(hash_hex)

        actual_hash = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt,
            n=2**14,
            r=8,
            p=1,
            dklen=32,
        )

        return hmac.compare_digest(
            actual_hash,
            expected_hash,
        )

    except (ValueError, TypeError, AttributeError):
        return False


# --------------------------------------------------
# Registration
# --------------------------------------------------

def register_user(email: str, password: str) -> dict:
    email = normalize_email(email)
    password_hash = hash_password(password)

    user_id = str(uuid.uuid4())

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    try:
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    user_id,
                    email,
                    password_hash,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    email,
                    password_hash,
                    created_at,
                ),
            )

    except sqlite3.IntegrityError as exc:
        raise ValueError(
            "An account with this email already exists."
        ) from exc

    return {
        "user_id": user_id,
        "email": email,
        "created_at": created_at,
    }


# --------------------------------------------------
# Login
# --------------------------------------------------

def authenticate_user(
    email: str,
    password: str,
) -> dict | None:

    email = normalize_email(email)

    with get_db_connection() as connection:
        user = connection.execute(
            """
            SELECT user_id, email, password_hash, created_at
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

    if user is None:
        return None

    if not verify_password(
        password,
        user["password_hash"],
    ):
        return None

    return {
        "user_id": user["user_id"],
        "email": user["email"],
        "created_at": user["created_at"],
    }


# --------------------------------------------------
# JWT Token Creation
# --------------------------------------------------

def create_access_token(user_id: str) -> str:
    user_id = normalize_user_id(user_id)

    now = datetime.now(timezone.utc)

    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        ),
    }

    return jwt.encode(
        payload,
        JWT_SECRET_KEY,
        algorithm=JWT_ALGORITHM,
    )


# --------------------------------------------------
# JWT Token Verification
# --------------------------------------------------

def verify_access_token(token: str) -> str:
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
        )

        user_id = payload.get("sub")

        if not user_id:
            raise ValueError("Invalid token identity.")

        return normalize_user_id(user_id)

    except (JWTError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


# --------------------------------------------------
# FastAPI Authentication Dependency
# --------------------------------------------------

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:

    token = credentials.credentials

    user_id = verify_access_token(token)

    # Confirm that the account still exists.
    with get_db_connection() as connection:
        user = connection.execute(
            """
            SELECT user_id
            FROM users
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account does not exist.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_id


# --------------------------------------------------
# Ownership Helpers
# --------------------------------------------------

def check_document_ownership(
    document_owner_id: str,
    current_user_id: str,
) -> bool:

    return (
        normalize_user_id(document_owner_id)
        == normalize_user_id(current_user_id)
    )


def authorize_document_access(
    document_owner_id: str,
    current_user_id: str,
):
    if not check_document_ownership(
        document_owner_id,
        current_user_id,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this document.",
        )


def authorize_conversation_access(
    conversation_owner_id: str,
    current_user_id: str,
):
    if not check_document_ownership(
        conversation_owner_id,
        current_user_id,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this conversation.",
        )
