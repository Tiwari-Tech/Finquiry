
import os

from functools import lru_cache
from contextlib import nullcontext

from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from backend.core.config import BASE_DIR


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------

load_dotenv(BASE_DIR / ".env")


# --------------------------------------------------
# BACKEND EMBEDDING CONFIGURATION
# --------------------------------------------------

# Change these values directly in this file to switch
# the embedding provider and model.

EMBEDDING_PROVIDER = "ollama"

OLLAMA_MODEL = "nomic-embed-text"

GOOGLE_MODEL = "models/gemini-embedding-001"

OLLAMA_BASE_URL = "http://localhost:11434"

EMBEDDING_BATCH_SIZE = 16


# Supported Ollama models:
#
# bge-m3             -> 1024 dimensions
# nomic-embed-text   -> 768 dimensions
# all-minilm         -> 384 dimensions
#
# Google:
# models/gemini-embedding-001 -> 3072 dimensions
#
# Actual dimensions are detected dynamically below.


# --------------------------------------------------
# PROVIDER AND MODEL RESOLUTION
# --------------------------------------------------

def resolve_embedding_config(
    provider=None,
    model_name=None,
):
    """
    Resolve provider and model.

    If no arguments are supplied, use the backend
    constants defined at the top of this file.
    """

    provider = (
        provider or EMBEDDING_PROVIDER
    ).strip().lower()

    if provider not in ("ollama", "google"):
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            "Supported providers: ollama, google."
        )

    if model_name is None:

        if provider == "ollama":
            model_name = OLLAMA_MODEL
        else:
            model_name = GOOGLE_MODEL

    model_name = model_name.strip()

    if not model_name:
        raise ValueError(
            "Embedding model name cannot be empty."
        )

    return provider, model_name


# --------------------------------------------------
# EMBEDDING MODEL INITIALIZATION
# --------------------------------------------------

@lru_cache(maxsize=8)
def get_embedding_model(
    provider=None,
    model_name=None,
):
    """
    Initialize and cache the selected embedding model.

    The model is selected using backend configuration.
    """

    provider, model_name = resolve_embedding_config(
        provider=provider,
        model_name=model_name,
    )

    if provider == "ollama":

        return OllamaEmbeddings(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
        )

    if provider == "google":

        api_key = os.getenv(
            "GOOGLE_API_KEY"
        )

        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY is missing from the .env file."
            )

        return GoogleGenerativeAIEmbeddings(
            model=model_name,
            google_api_key=api_key,
        )

    raise ValueError(
        f"Unsupported embedding provider: {provider}"
    )


# --------------------------------------------------
# EMBEDDING DIMENSION DETECTION
# --------------------------------------------------

@lru_cache(maxsize=8)
def get_embedding_dimension(
    provider=None,
    model_name=None,
):
    """
    Detect the actual output dimension of an embedding model.

    The result is cached for each provider/model combination.
    """

    provider, model_name = resolve_embedding_config(
        provider=provider,
        model_name=model_name,
    )

    embedding_model = get_embedding_model(
        provider=provider,
        model_name=model_name,
    )

    test_embedding = embedding_model.embed_query(
        "embedding dimension detection"
    )

    dimension = len(test_embedding)

    if dimension <= 0:
        raise ValueError(
            "The embedding model returned an invalid dimension."
        )

    return dimension


# --------------------------------------------------
# COMPLETE EMBEDDING CONFIGURATION
# --------------------------------------------------

def get_embedding_config(
    provider=None,
    model_name=None,
):
    """
    Return the active embedding configuration.

    Dimension is detected from the actual model output.
    """

    provider, model_name = resolve_embedding_config(
        provider=provider,
        model_name=model_name,
    )

    dimension = get_embedding_dimension(
        provider=provider,
        model_name=model_name,
    )

    return {
        "provider": provider,
        "model": model_name,
        "dimension": dimension,
        "batch_size": EMBEDDING_BATCH_SIZE,
    }


# --------------------------------------------------
# DOCUMENT EMBEDDING
# --------------------------------------------------

def embed_documents(
    texts,
    tracker=None,
    batch_size=None,
    provider=None,
    model_name=None,
):
    """
    Generate embeddings for multiple document chunks.

    Documents are processed in batches to control
    memory usage and request size.
    """

    if not texts:
        return []

    provider, model_name = resolve_embedding_config(
        provider=provider,
        model_name=model_name,
    )

    embedding_model = get_embedding_model(
        provider=provider,
        model_name=model_name,
    )

    batch_size = (
        batch_size or EMBEDDING_BATCH_SIZE
    )

    if batch_size <= 0:
        raise ValueError(
            "Embedding batch_size must be greater than zero."
        )

    all_embeddings = []

    tracking_context = (
        tracker.track("embedding_generation")
        if tracker
        else nullcontext()
    )

    with tracking_context:

        for start_index in range(
            0,
            len(texts),
            batch_size,
        ):

            batch = texts[
                start_index:
                start_index + batch_size
            ]

            batch_embeddings = (
                embedding_model.embed_documents(
                    batch
                )
            )

            if len(batch_embeddings) != len(batch):
                raise ValueError(
                    "Embedding model returned an unexpected "
                    "number of document embeddings."
                )

            all_embeddings.extend(
                batch_embeddings
            )

    return all_embeddings


# --------------------------------------------------
# QUERY EMBEDDING
# --------------------------------------------------

def embed_query(
    query,
    tracker=None,
    provider=None,
    model_name=None,
):
    """
    Generate an embedding for a user query.
    """

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty."
        )

    provider, model_name = resolve_embedding_config(
        provider=provider,
        model_name=model_name,
    )

    embedding_model = get_embedding_model(
        provider=provider,
        model_name=model_name,
    )

    tracking_context = (
        tracker.track("query_embedding")
        if tracker
        else nullcontext()
    )

    with tracking_context:

        return embedding_model.embed_query(
            query.strip()
        )


# --------------------------------------------------
# CACHE MANAGEMENT
# --------------------------------------------------

def clear_embedding_cache():
    """
    Clear cached models and detected dimensions.

    Useful after changing embedding configuration
    during development.
    """

    get_embedding_model.cache_clear()

    get_embedding_dimension.cache_clear()


# --------------------------------------------------
# CONFIGURATION SUMMARY
# --------------------------------------------------

def print_embedding_config():
    """
    Print the active embedding configuration.
    """

    config = get_embedding_config()

    print("\n" + "=" * 50)
    print("ACTIVE EMBEDDING CONFIGURATION")
    print("=" * 50)

    print(
        f"Provider   : {config['provider']}"
    )

    print(
        f"Model      : {config['model']}"
    )

    print(
        f"Dimension  : {config['dimension']}"
    )

    print(
        f"Batch Size : {config['batch_size']}"
    )

    print("=" * 50 + "\n")

    return config
