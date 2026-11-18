
import hashlib
import re
import time
from collections import defaultdict

from rank_bm25 import BM25Plus

from langchain_core.documents import Document
from langchain_chroma import Chroma

from backend.core.config import CHROMA_DIR
from backend.services.embedding_service import get_embedding_model
from backend.services.auth_service import normalize_user_id
from backend.services.langfuse_service import trace_span


# --------------------------------------------------
# Configuration
# --------------------------------------------------

COLLECTION_NAME = "earnings_calls"

DEFAULT_TOP_K = 3
RRF_K = 60
BM25_CANDIDATE_LIMIT = 100
CANDIDATE_MULTIPLIER = 4
MIN_CANDIDATES = 10


# --------------------------------------------------
# Vector Store
# --------------------------------------------------

def get_vector_store(embedding_model=None):
    """Return the persistent ChromaDB vector store."""

    if embedding_model is None:
        embedding_model = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
        embedding_function=embedding_model,
    )


# --------------------------------------------------
# User ID Normalization
# --------------------------------------------------

def normalize_owner_id(email: str) -> str:

    if not email:
        raise ValueError("User identity is required.")

    return normalize_user_id(email)


# --------------------------------------------------
# Metadata Filter Validation
# --------------------------------------------------

def validate_search_filters(
    company=None,
    filename=None,
    department=None,
    document_type=None,
    fiscal_year=None,
    document_id=None,
):
    """Validate optional metadata filters."""

    if fiscal_year not in (None, ""):

        try:
            fiscal_year = int(fiscal_year)

        except (ValueError, TypeError):
            raise ValueError("Fiscal year must be an integer.")

        if not 1900 <= fiscal_year <= 2100:
            raise ValueError("Invalid fiscal year.")

    else:
        fiscal_year = None

    return {
        "company": company.strip() if company else None,
        "filename": filename.strip() if filename else None,
        "department": department.strip() if department else None,
        "document_type": (
            document_type.strip() if document_type else None
        ),
        "fiscal_year": fiscal_year,
        "document_id": (
            document_id.strip() if document_id else None
        ),
    }


# --------------------------------------------------
# Metadata Filter Builder
# --------------------------------------------------

def build_user_filter(
    owner_id: str,
    company: str = None,
    filename: str = None,
    department: str = None,
    document_type: str = None,
    fiscal_year: int = None,
    document_id: str = None,
):
    """Build a mandatory user-scoped ChromaDB filter."""

    owner_id = normalize_owner_id(owner_id)

    conditions = [{"owner_id": owner_id}]

    if company:
        conditions.append({"company": company.strip()})

    if filename:
        conditions.append({"filename": filename.strip()})

    if department:
        conditions.append({"department": department.strip()})

    if document_type:
        conditions.append({"document_type": document_type.strip()})

    if document_id:
        conditions.append({"document_id": document_id.strip()})

    if fiscal_year is not None:
        conditions.append({"fiscal_year": int(fiscal_year)})

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}


# --------------------------------------------------
# Tokenization
# --------------------------------------------------

def tokenize(text: str) -> list:

    if not text:
        return []

    return re.findall(r"[a-zA-Z0-9]+", text.lower())


# --------------------------------------------------
# Chunk Identity
# --------------------------------------------------

def get_chunk_identity(document: Document):
    """Return a stable identity for retrieval and evaluation."""

    chunk_id = document.metadata.get("_chunk_id")

    if chunk_id:
        return str(chunk_id)

    document_id = document.metadata.get("document_id", "")

    content_hash = hashlib.sha256(
        document.page_content.encode("utf-8")
    ).hexdigest()

    return f"{document_id}:{content_hash}"


def get_diagnostic_record(document: Document, rank: int):
    """Return safe retrieval metadata without document text."""

    metadata = document.metadata or {}

    return {
        "chunk_id": get_chunk_identity(document),
        "document_id": metadata.get("document_id"),
        "filename": metadata.get("filename"),
        "company": metadata.get("company"),
        "department": metadata.get("department"),
        "document_type": metadata.get("document_type"),
        "fiscal_year": metadata.get("fiscal_year"),
        "page": metadata.get("page"),
        "rank": rank,
        "bm25_score": metadata.get("_bm25_score"),
        "rrf_score": metadata.get("_rrf_score"),
    }


# --------------------------------------------------
# Semantic Search
# --------------------------------------------------

def semantic_search(
    query: str,
    owner_id: str,
    k: int = DEFAULT_TOP_K,
    company: str = None,
    filename: str = None,
    department: str = None,
    document_type: str = None,
    fiscal_year: int = None,
    document_id: str = None,
    vector_store=None,
):
    """Semantic vector search with user-scoped metadata filtering."""

    if not query or not query.strip():
        return []

    filters = validate_search_filters(
        company=company,
        filename=filename,
        department=department,
        document_type=document_type,
        fiscal_year=fiscal_year,
        document_id=document_id,
    )

    where = build_user_filter(
        owner_id=owner_id,
        **filters,
    )

    if vector_store is None:
        vector_store = get_vector_store()

    return vector_store.similarity_search(
        query=query,
        k=k,
        filter=where,
    )


# --------------------------------------------------
# Fetch User Chunks
# --------------------------------------------------

def fetch_user_chunks(
    owner_id: str,
    company: str = None,
    filename: str = None,
    department: str = None,
    document_type: str = None,
    fiscal_year: int = None,
    document_id: str = None,
    vector_store=None,
):
    """Fetch only chunks belonging to the authenticated user."""

    filters = validate_search_filters(
        company=company,
        filename=filename,
        department=department,
        document_type=document_type,
        fiscal_year=fiscal_year,
        document_id=document_id,
    )

    where = build_user_filter(
        owner_id=owner_id,
        **filters,
    )

    if vector_store is None:
        vector_store = get_vector_store()

    result = vector_store._collection.get(
        where=where,
        include=["documents", "metadatas"],
    )

    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []
    ids = result.get("ids") or []

    chunks = []

    for index, content in enumerate(documents):

        if not content:
            continue

        metadata = (
            metadatas[index]
            if index < len(metadatas)
            else {}
        )

        chunk = Document(
            page_content=content,
            metadata=dict(metadata or {}),
        )

        if index < len(ids):
            chunk.metadata["_chunk_id"] = ids[index]

        chunks.append(chunk)

    return chunks


# --------------------------------------------------
# BM25 Keyword Search
# --------------------------------------------------

def bm25_search(
    query: str,
    owner_id: str,
    k: int = DEFAULT_TOP_K,
    company: str = None,
    filename: str = None,
    department: str = None,
    document_type: str = None,
    fiscal_year: int = None,
    document_id: str = None,
    vector_store=None,
):
    """Keyword retrieval using BM25Plus."""

    if not query or not query.strip():
        return []

    documents = fetch_user_chunks(
        owner_id=owner_id,
        company=company,
        filename=filename,
        department=department,
        document_type=document_type,
        fiscal_year=fiscal_year,
        document_id=document_id,
        vector_store=vector_store,
    )

    if not documents:
        return []

    query_tokens = tokenize(query)

    if not query_tokens:
        return []

    tokenized_corpus = [
        tokenize(document.page_content)
        for document in documents
    ]

    bm25 = BM25Plus(tokenized_corpus)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []

    for index in ranked_indices[:k]:

        document = documents[index]

        document.metadata["_bm25_score"] = float(scores[index])

        results.append(document)

    return results


# --------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------

def reciprocal_rank_fusion(
    result_lists,
    k: int = RRF_K,
):
    """Merge ranked retrieval lists using Reciprocal Rank Fusion."""

    scores = defaultdict(float)
    document_map = {}

    for result_list in result_lists:

        for rank, document in enumerate(result_list, start=1):

            identity = get_chunk_identity(document)

            scores[identity] += 1.0 / (k + rank)

            existing = document_map.get(identity)

            if (
                existing is None
                or len(document.metadata or {})
                > len(existing.metadata or {})
            ):
                document_map[identity] = document

    ranked_ids = sorted(
        scores,
        key=lambda identity: scores[identity],
        reverse=True,
    )

    results = []

    for identity in ranked_ids:

        document = document_map[identity]

        document.metadata["_rrf_score"] = scores[identity]

        results.append(document)

    return results


# --------------------------------------------------
# Document Diversity
# --------------------------------------------------

def diversify_results(
    documents,
    top_k: int,
    max_per_document: int = 3,
):
    """Limit the number of chunks returned from one document."""

    selected = []
    document_counts = defaultdict(int)

    for document in documents:

        document_id = document.metadata.get(
            "document_id",
            "unknown",
        )

        if document_counts[document_id] >= max_per_document:
            continue

        selected.append(document)
        document_counts[document_id] += 1

        if len(selected) >= top_k:
            break

    return selected


# --------------------------------------------------
# Metadata-Aware Hybrid Search
# --------------------------------------------------

def hybrid_search(
    owner_id: str,
    query: str,
    company: str = None,
    filename: str = None,
    department: str = None,
    document_type: str = None,
    fiscal_year: int = None,
    document_id: str = None,
    k: int = DEFAULT_TOP_K,
    max_per_document: int = 3,
    return_diagnostics: bool = False,
):
    """
    Hybrid retrieval pipeline.

    1. Validate filters.
    2. Initialize ChromaDB.
    3. Perform semantic retrieval.
    4. Perform BM25 retrieval.
    5. Merge using RRF.
    6. Apply document diversity.

    By default, returns a list of Documents to preserve compatibility.

    When return_diagnostics=True, returns:
        {
            "documents": [...],
            "diagnostics": {...}
        }

    Diagnostics contain metadata and timings, not document contents.
    """

    if not query or not query.strip():

        if return_diagnostics:
            return {
                "documents": [],
                "diagnostics": {
                    "success": True,
                    "reason": "empty_query",
                },
            }

        return []

    if k < 1:
        raise ValueError("k must be at least 1.")

    if max_per_document < 1:
        raise ValueError(
            "max_per_document must be at least 1."
        )

    total_start = time.perf_counter()

    owner_id = normalize_owner_id(owner_id)

    filters = validate_search_filters(
        company=company,
        filename=filename,
        department=department,
        document_type=document_type,
        fiscal_year=fiscal_year,
        document_id=document_id,
    )

    candidate_k = max(
        k * CANDIDATE_MULTIPLIER,
        MIN_CANDIDATES,
    )

    timings = {}

    # 1. Initialize ChromaDB.
    stage_start = time.perf_counter()

    with trace_span(
        "chroma_vector_store_initialization",
        input={"collection": COLLECTION_NAME},
    ):
        vector_store = get_vector_store()

    timings["vector_store_initialization_ms"] = round(
        (time.perf_counter() - stage_start) * 1000,
        2,
    )

    # 2. Semantic retrieval.
    stage_start = time.perf_counter()

    with trace_span(
        "semantic_retrieval",
        input={
            "candidate_k": candidate_k,
            "filters_applied": [
                key for key, value in filters.items()
                if value is not None
            ],
        },
    ) as semantic_span:

        semantic_results = semantic_search(
            query=query,
            owner_id=owner_id,
            k=candidate_k,
            vector_store=vector_store,
            **filters,
        )

        if semantic_span:
            semantic_span.update(
                output={"result_count": len(semantic_results)}
            )

    timings["semantic_retrieval_ms"] = round(
        (time.perf_counter() - stage_start) * 1000,
        2,
    )

    # 3. BM25 retrieval.
    stage_start = time.perf_counter()

    with trace_span(
        "bm25_keyword_retrieval",
        input={
            "candidate_k": min(BM25_CANDIDATE_LIMIT, candidate_k)
        },
    ) as bm25_span:

        bm25_results = bm25_search(
            query=query,
            owner_id=owner_id,
            k=min(BM25_CANDIDATE_LIMIT, candidate_k),
            vector_store=vector_store,
            **filters,
        )

        if bm25_span:
            bm25_span.update(
                output={"result_count": len(bm25_results)}
            )

    timings["bm25_retrieval_ms"] = round(
        (time.perf_counter() - stage_start) * 1000,
        2,
    )

    # 4. Reciprocal Rank Fusion.
    stage_start = time.perf_counter()

    with trace_span(
        "reciprocal_rank_fusion",
        input={
            "semantic_count": len(semantic_results),
            "bm25_count": len(bm25_results),
            "rrf_k": RRF_K,
        },
    ) as rrf_span:

        combined_results = reciprocal_rank_fusion(
            [semantic_results, bm25_results]
        )

        if rrf_span:
            rrf_span.update(
                output={"fused_count": len(combined_results)}
            )

    timings["rrf_ms"] = round(
        (time.perf_counter() - stage_start) * 1000,
        2,
    )

    # 5. Document diversity.
    stage_start = time.perf_counter()

    with trace_span(
        "document_diversity",
        input={
            "requested_top_k": k,
            "max_per_document": max_per_document,
        },
    ) as diversity_span:

        final_results = diversify_results(
            documents=combined_results,
            top_k=k,
            max_per_document=max_per_document,
        )

        if diversity_span:
            diversity_span.update(
                output={
                    "final_result_count": len(final_results)
                }
            )

    timings["diversity_filter_ms"] = round(
        (time.perf_counter() - stage_start) * 1000,
        2,
    )

    timings["total_retrieval_ms"] = round(
        (time.perf_counter() - total_start) * 1000,
        2,
    )

    # Default behavior: return only documents.
    if not return_diagnostics:
        return final_results

    # Diagnostics mode: metadata only, no document text.
    diagnostics = {
        "success": True,
        "retrieval_method": "hybrid_semantic_bm25_rrf",
        "requested_top_k": k,
        "candidate_k": candidate_k,
        "filters_applied": {
            key: value
            for key, value in filters.items()
            if value is not None
        },
        "result_counts": {
            "semantic": len(semantic_results),
            "bm25": len(bm25_results),
            "fused": len(combined_results),
            "final": len(final_results),
        },
        "timings_ms": timings,
        "semantic_results": [
            get_diagnostic_record(doc, rank)
            for rank, doc in enumerate(semantic_results, start=1)
        ],
        "bm25_results": [
            get_diagnostic_record(doc, rank)
            for rank, doc in enumerate(bm25_results, start=1)
        ],
        "fused_results": [
            get_diagnostic_record(doc, rank)
            for rank, doc in enumerate(combined_results)
        ],
        "final_results": [
            get_diagnostic_record(doc, rank)
            for rank, doc in enumerate(final_results, start=1)
        ],
    }

    return {
        "documents": final_results,
        "diagnostics": diagnostics,
    }


# --------------------------------------------------
# User Document Search
# --------------------------------------------------

def search_user_documents(
    email: str,
    question: str,
    company: str = None,
    k: int = DEFAULT_TOP_K,
    document_id: str = None,
    filename: str = None,
    department: str = None,
    document_type: str = None,
    fiscal_year: int = None,
    return_diagnostics: bool = False,
):
    """
    Search documents belonging to one authenticated user.

    Optional metadata filters narrow the search.

    Existing callers receive a list of Documents.
    Diagnostics mode returns documents and retrieval metadata.
    """

    owner_id = normalize_owner_id(email)

    return hybrid_search(
        owner_id=owner_id,
        query=question,
        company=company,
        filename=filename,
        department=department,
        document_type=document_type,
        fiscal_year=fiscal_year,
        document_id=document_id,
        k=k,
        return_diagnostics=return_diagnostics,
    )


# --------------------------------------------------
# Search All User Documents
# --------------------------------------------------

def search_all_user_documents(
    email: str,
    question: str,
    k: int = DEFAULT_TOP_K,
):
    """Search every document belonging to one user."""

    return search_user_documents(
        email=email,
        question=question,
        k=k,
    )


# --------------------------------------------------
# List User Documents
# --------------------------------------------------

def list_user_documents(email: str):
    """Return unique documents belonging to a user."""

    owner_id = normalize_owner_id(email)

    vector_store = get_vector_store()

    result = vector_store._collection.get(
        where={"owner_id": owner_id},
        include=["metadatas"],
    )

    metadatas = result.get("metadatas") or []

    document_map = {}

    for metadata in metadatas:

        if not metadata:
            continue

        document_id = metadata.get("document_id")

        if not document_id:
            continue

        if document_id not in document_map:

            document_map[document_id] = {
                "document_id": document_id,
                "filename": metadata.get("filename"),
                "company": metadata.get("company"),
                "department": metadata.get("department"),
                "document_type": metadata.get("document_type"),
                "fiscal_year": metadata.get("fiscal_year"),
                "file_type": metadata.get("file_type"),
            }

    return list(document_map.values())


# --------------------------------------------------
# Get Specific User Document
# --------------------------------------------------

def get_user_document(
    email: str,
    document_id: str,
):
    """Return document metadata only if it belongs to the user."""

    owner_id = normalize_owner_id(email)

    where = build_user_filter(
        owner_id=owner_id,
        document_id=document_id,
    )

    vector_store = get_vector_store()

    result = vector_store._collection.get(
        where=where,
        include=["metadatas"],
        limit=1,
    )

    metadatas = result.get("metadatas") or []

    if not metadatas:
        return None

    metadata = metadatas[0]

    return {
        "document_id": metadata.get("document_id"),
        "filename": metadata.get("filename"),
        "company": metadata.get("company"),
        "department": metadata.get("department"),
        "document_type": metadata.get("document_type"),
        "fiscal_year": metadata.get("fiscal_year"),
        "file_type": metadata.get("file_type"),
    }


# --------------------------------------------------
# Delete User Document
# --------------------------------------------------

def delete_user_document(
    document_id: str,
    email: str,
):
    """Delete document chunks only after verifying ownership."""

    owner_id = normalize_owner_id(email)

    where = build_user_filter(
        owner_id=owner_id,
        document_id=document_id,
    )

    vector_store = get_vector_store()

    existing = vector_store._collection.get(
        where=where,
        include=["metadatas"],
        limit=1,
    )

    if not existing.get("ids"):
        return False

    vector_store._collection.delete(where=where)

    return True


# --------------------------------------------------
# Restricted Company Search
# --------------------------------------------------

def search_company_documents(*args, **kwargs):
    """Company-wide search without user ownership is disabled."""

    raise PermissionError(
        "Company-wide search without authenticated "
        "user-level access control is not permitted."
    )
