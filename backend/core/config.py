
import os
from pathlib import Path

from dotenv import load_dotenv


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

# Project root: Finquiry/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"

CHROMA_DIR = DATA_DIR / "chroma_db"


# --------------------------------------------------
# ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv(BASE_DIR / ".env")


def get_env(
    key: str,
    default: str | None = None,
) -> str | None:
    """
    Read an environment variable.

    Empty strings are treated as missing values.
    """

    value = os.getenv(key)

    if value is None or not value.strip():
        return default

    return value.strip()


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

APP_NAME = "Finquiry - Multi-User Document Search"

APP_VERSION = "1.0.0"

DEBUG = get_env(
    "DEBUG",
    "False",
).lower() == "true"


# --------------------------------------------------
# AUTHENTICATION CONFIGURATION
# --------------------------------------------------

SECRET_KEY = get_env(
    "SECRET_KEY",
    "change-this-secret-key-before-production",
)

ALGORITHM = get_env(
    "ALGORITHM",
    "HS256",
)

ACCESS_TOKEN_EXPIRE_MINUTES = int(
    get_env(
        "ACCESS_TOKEN_EXPIRE_MINUTES",
        "60",
    )
)


# --------------------------------------------------
# DOCUMENT CHUNKING CONFIGURATION
# --------------------------------------------------

CHUNK_SIZE = int(
    get_env(
        "CHUNK_SIZE",
        "1000",
    )
)

CHUNK_OVERLAP = int(
    get_env(
        "CHUNK_OVERLAP",
        "150",
    )
)


# --------------------------------------------------
# UPLOAD CONFIGURATION
# --------------------------------------------------

MAX_UPLOAD_SIZE_MB = int(
    get_env(
        "MAX_UPLOAD_SIZE_MB",
        "25",
    )
)

MAX_UPLOAD_SIZE_BYTES = (
    MAX_UPLOAD_SIZE_MB * 1024 * 1024
)


# --------------------------------------------------
# CHROMA CONFIGURATION
# --------------------------------------------------

CHROMA_COLLECTION_PREFIX = get_env(
    "CHROMA_COLLECTION_PREFIX",
    "ata",
)


# --------------------------------------------------
# GROQ CONFIGURATION
# --------------------------------------------------

GROQ_API_KEY = get_env(
    "GROQ_API_KEY"
)

GROQ_MODEL = get_env(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)


# --------------------------------------------------
# EMBEDDING CONFIGURATION
# --------------------------------------------------

GOOGLE_API_KEY = get_env(
    "GOOGLE_API_KEY"
)


# --------------------------------------------------
# DIRECTORY INITIALIZATION
# --------------------------------------------------

def create_required_directories() -> None:
    """
    Create application directories if they do not exist.
    """

    directories = [
        DATA_DIR,
        CHROMA_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


create_required_directories()


# --------------------------------------------------
# CONFIGURATION VALIDATION
# --------------------------------------------------

if CHUNK_SIZE <= 0:
    raise ValueError(
        "CHUNK_SIZE must be greater than zero."
    )

if CHUNK_OVERLAP < 0:
    raise ValueError(
        "CHUNK_OVERLAP cannot be negative."
    )

if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
    )

if ACCESS_TOKEN_EXPIRE_MINUTES <= 0:
    raise ValueError(
        "ACCESS_TOKEN_EXPIRE_MINUTES must be greater than zero."
    )
