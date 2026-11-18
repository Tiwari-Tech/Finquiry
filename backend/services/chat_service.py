
import os
import re
import hashlib
from functools import lru_cache

from langchain_groq import ChatGroq
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from backend.services.retrieval_service import (
    search_user_documents,
    list_user_documents,
)

from backend.services.history import (
    get_conversation_messages,
    get_conversation,
    create_conversation,
    save_message,
)

from backend.services.latency import LatencyTracker

from backend.services.langfuse_service import (
    trace_span,
    flush_traces,
    get_current_trace_id,
    log_evaluation_scores,
)

from backend.services.evaluation_service import evaluate_answer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)

DEFAULT_TOP_K = 6
MAX_CONTEXT_CHARACTERS = 18000
MAX_HISTORY_MESSAGES = 10


# --------------------------------------------------
# Groq LLM
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_llm():

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured.")

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.1,
        max_tokens=2048,
    )


# --------------------------------------------------
# Text Utilities
# --------------------------------------------------

def normalize_text(value: str) -> str:

    if not value:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower(),
    )


def get_document_identity(document):

    metadata = document.metadata or {}

    return (
        metadata.get("_chunk_id")
        or (
            str(metadata.get("document_id", ""))
            + str(metadata.get("page_number", ""))
            + document.page_content[:100]
        )
    )


def merge_unique_documents(document_groups, top_k):

    merged = {}

    for documents in document_groups:

        for document in documents:

            identity = get_document_identity(document)

            merged[identity] = document

    return list(merged.values())[:top_k]


# --------------------------------------------------
# Follow-up Query Context
# --------------------------------------------------

FOLLOW_UP_PATTERNS = [
    r"\bwhat about\b",
    r"\bhow about\b",
    r"\band what\b",
    r"\bcompared to\b",
    r"\bcompare it\b",
    r"\bthat company\b",
    r"\bthis company\b",
    r"\bthat year\b",
    r"\bprevious year\b",
    r"\blast year\b",
    r"\bsame year\b",
    r"\bthere\b",
    r"\bthey\b",
    r"\btheir\b",
    r"\bit\b",
    r"\bthem\b",
    r"\bthose\b",
    r"\bthese\b",
    r"\bmore details\b",
    r"\bexplain further\b",
]


def is_follow_up_question(question: str) -> bool:

    normalized = normalize_text(question)

    return any(
        re.search(pattern, normalized)
        for pattern in FOLLOW_UP_PATTERNS
    )


def build_retrieval_query(question: str, history: list):
    """
    Add previous user questions to retrieval context
    when the current question appears to be a follow-up.
    """

    if not history or not is_follow_up_question(question):
        return question

    previous_user_questions = []

    for message in reversed(history):

        if message.get("role") == "user":

            content = message.get("content", "").strip()

            if content:
                previous_user_questions.append(content)

            if len(previous_user_questions) >= 2:
                break

    if not previous_user_questions:
        return question

    previous_context = "\n".join(
        reversed(previous_user_questions)
    )

    return (
        f"Previous conversation questions:\n"
        f"{previous_context}\n\n"
        f"Current follow-up question:\n"
        f"{question}"
    )


# --------------------------------------------------
# Metadata Matching
# --------------------------------------------------

def find_metadata_matches(question: str, documents: list):

    normalized_question = normalize_text(question)

    matched_companies = set()
    matched_filenames = set()
    matched_departments = set()
    matched_document_types = set()

    for document in documents:

        company = document.get("company")
        filename = document.get("filename")
        department = document.get("department")
        document_type = document.get("document_type")

        if company and normalize_text(company) in normalized_question:
            matched_companies.add(company)

        if filename:

            filename_normalized = normalize_text(filename)

            filename_without_extension = normalize_text(
                os.path.splitext(filename)[0]
            )

            if (
                filename_normalized in normalized_question
                or filename_without_extension in normalized_question
            ):
                matched_filenames.add(filename)

        if department and normalize_text(department) in normalized_question:
            matched_departments.add(department)

        if document_type and normalize_text(document_type) in normalized_question:
            matched_document_types.add(document_type)

    year_match = re.search(
        r"\b(?:FY\s*)?(20\d{2})\b",
        question,
        flags=re.IGNORECASE,
    )

    fiscal_year = (
        int(year_match.group(1))
        if year_match
        else None
    )

    return {
        "companies": sorted(matched_companies),
        "filenames": sorted(matched_filenames),
        "departments": sorted(matched_departments),
        "document_types": sorted(matched_document_types),
        "fiscal_year": fiscal_year,
    }


# --------------------------------------------------
# Metadata-aware Retrieval
# --------------------------------------------------

def retrieve_documents(
    user_id: str,
    question: str,
    top_k: int = DEFAULT_TOP_K,
    company: str = None,
    document_id: str = None,
    return_diagnostics: bool = False,
):
    """
    Retrieve only the authenticated user's document chunks.

    Existing behavior returns a list of Documents.

    Diagnostics mode returns:
        {
            "documents": [...],
            "diagnostics": {...}
        }
    """

    retrieval_diagnostics = {
        "success": True,
        "searches": [],
        "result_count": 0,
    }

    def run_search(**kwargs):

        result = search_user_documents(
            email=user_id,
            question=question,
            k=top_k,
            return_diagnostics=return_diagnostics,
            **kwargs,
        )

        if return_diagnostics:

            documents = result.get("documents", [])
            diagnostics = result.get("diagnostics", {})

            retrieval_diagnostics["searches"].append(
                diagnostics
            )

            return documents

        return result

    def finish(documents):

        if not return_diagnostics:
            return documents

        retrieval_diagnostics["result_count"] = len(documents)

        return {
            "documents": documents,
            "diagnostics": retrieval_diagnostics,
        }

    user_documents = list_user_documents(email=user_id)

    if not user_documents:
        return finish([])

    # Explicit document ID has the highest priority.
    if document_id:

        documents = run_search(
            document_id=document_id,
        )

        return finish(documents)

    matches = find_metadata_matches(
        question=question,
        documents=user_documents,
    )

    companies = matches["companies"]
    filenames = matches["filenames"]
    departments = matches["departments"]
    document_types = matches["document_types"]
    fiscal_year = matches["fiscal_year"]

    if company:
        companies = [company]

    filter_dimensions = {
        "company": companies,
        "filename": filenames,
        "department": departments,
        "document_type": document_types,
        "fiscal_year": (
            [fiscal_year] if fiscal_year is not None else []
        ),
    }

    active_dimensions = {
        key: values
        for key, values in filter_dimensions.items()
        if values
    }

    # No metadata constraints: search all user documents.
    if not active_dimensions:

        documents = run_search()

        return finish(documents)

    # OR within each metadata dimension.
    # AND across different metadata dimensions.
    from itertools import product

    dimension_names = list(active_dimensions.keys())

    dimension_values = [
        active_dimensions[name]
        for name in dimension_names
    ]

    document_groups = []

    for combination in product(*dimension_values):

        metadata_filter = dict(
            zip(dimension_names, combination)
        )

        results = run_search(
            company=metadata_filter.get("company"),
            filename=metadata_filter.get("filename"),
            department=metadata_filter.get("department"),
            document_type=metadata_filter.get("document_type"),
            fiscal_year=metadata_filter.get("fiscal_year"),
        )

        document_groups.append(results)

    documents = merge_unique_documents(
        document_groups,
        top_k,
    )

    return finish(documents)


# --------------------------------------------------
# Conversation History
# --------------------------------------------------

def load_conversation_history(
    conversation_id: str,
    user_id: str,
):

    if not conversation_id:
        return []

    messages = get_conversation_messages(
        conversation_id,
        user_id,
    )

    return messages[-MAX_HISTORY_MESSAGES:]


def format_history_messages(history):

    formatted_messages = []

    for message in history:

        role = message.get("role")
        content = message.get("content")

        if not content:
            continue

        if role == "user":

            formatted_messages.append(
                HumanMessage(content=content)
            )

        elif role == "assistant":

            formatted_messages.append(
                AIMessage(content=content)
            )

    return formatted_messages


# --------------------------------------------------
# Retrieved Context and Sources
# --------------------------------------------------

def format_retrieved_context(documents):

    context_parts = []
    sources = []
    total_characters = 0

    for index, document in enumerate(documents, start=1):

        metadata = document.metadata or {}

        filename = metadata.get(
            "filename",
            metadata.get("source", "Unknown document"),
        )

        company = metadata.get(
            "company",
            "Unknown company",
        )

        page_number = metadata.get("page_number")
        document_id = metadata.get("document_id")

        source_details = (
            f"Source {index}: {filename}\n"
            f"Company: {company}\n"
            f"Document ID: {document_id}\n"
            f"Page: {page_number if page_number is not None else 'Not available'}"
        )

        content = (
            f"{source_details}\n"
            f"Content:\n{document.page_content}"
        )

        remaining = MAX_CONTEXT_CHARACTERS - total_characters

        if remaining <= 0:
            break

        content = content[:remaining]

        context_parts.append(content)
        total_characters += len(content)

        source = {
            "filename": filename,
            "company": company,
            "document_id": document_id,
            "page_number": page_number,
        }

        if source not in sources:
            sources.append(source)

    return "\n\n---\n\n".join(context_parts), sources


# --------------------------------------------------
# System Prompt
# --------------------------------------------------

def build_system_prompt(context: str):

    return f"""
You are Finquiry, a multi-document question-answering assistant.

Answer questions using the retrieved document context.

RULES:

1. Use retrieved document context as the primary
   source of factual information.

2. Never invent financial figures, dates,
   statements, or company information.

3. If the answer is unavailable in the retrieved
   context, clearly state that the documents do
   not provide sufficient information.

4. When comparing companies, clearly distinguish
   their information.

5. Do not mix information from different documents.

6. For calculations, show the calculation when
   appropriate and use only supported figures.

7. Use conversation history to understand follow-up
   questions. Previous assistant answers are not
   verified document evidence.

8. Never invent page numbers.

9. Answer clearly and directly.

RETRIEVED DOCUMENT CONTEXT:

{context}
"""


# --------------------------------------------------
# Evaluation Helpers
# --------------------------------------------------

def run_answer_evaluation(
    question: str,
    answer: str,
    context: str,
):
    """
    Run automated answer evaluation.

    Evaluation errors must never fail the chat request.
    """

    try:

        evaluation_result = evaluate_answer(
            question=question,
            answer=answer,
            context=context,
        )

        if not isinstance(evaluation_result, dict):

            return {
                "success": False,
                "error": "Evaluation returned an invalid response.",
                "metrics": {},
            }

        return evaluation_result

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc),
            "metrics": {},
        }


# --------------------------------------------------
# Main Chat Function
# --------------------------------------------------

def _answer_question_impl(
    user_id: str,
    question: str,
    conversation_id: str = None,
    company: str = None,
    document_id: str = None,
):
    """Execute chat, retrieval, generation, and evaluation."""

    tracker = LatencyTracker("CHAT QUESTION ANSWERING")

    user_id = user_id.strip().lower() if user_id else ""
    question = question.strip() if question else ""

    if not user_id:
        raise ValueError("Authenticated user is required.")

    if not question:
        raise ValueError("Question cannot be empty.")

    try:

        # 1. Initialize or validate conversation.

        with tracker.track("conversation_initialization"):

            if conversation_id:

                conversation = get_conversation(
                    conversation_id,
                    user_id,
                )

                if not conversation:
                    raise ValueError(
                        "Conversation not found or access denied."
                    )

            else:

                conversation_id = create_conversation(
                    user_id=user_id,
                    question=question,
                    company=company or "",
                )

        # 2. Load previous messages.

        with tracker.track("conversation_history_loading"):

            history = load_conversation_history(
                conversation_id=conversation_id,
                user_id=user_id,
            )

        # 3. Prepare retrieval query.

        with tracker.track("retrieval_query_preparation"):

            retrieval_query = build_retrieval_query(
                question=question,
                history=history,
            )

        # 4. Retrieve documents with diagnostics.

        with tracker.track("metadata_aware_hybrid_retrieval"):

            with trace_span(
                "metadata_aware_hybrid_retrieval",
                input={
                    "top_k": DEFAULT_TOP_K,
                    "company_filter": bool(company),
                    "document_filter": bool(document_id),
                },
            ) as retrieval_span:

                retrieval_result = retrieve_documents(
                    user_id=user_id,
                    question=retrieval_query,
                    top_k=DEFAULT_TOP_K,
                    company=company,
                    document_id=document_id,
                    return_diagnostics=True,
                )

                retrieved_documents = retrieval_result["documents"]
                retrieval_diagnostics = retrieval_result["diagnostics"]

                if retrieval_span:

                    retrieval_span.update(
                        output={
                            "retrieved_chunks": len(retrieved_documents),
                            "search_count": len(
                                retrieval_diagnostics.get("searches", [])
                            ),
                        }
                    )

        # 5. Handle no relevant documents.

        if not retrieved_documents:

            answer = (
                "I could not find relevant information "
                "in your uploaded documents to answer "
                "this question."
            )

            with tracker.track("conversation_persistence"):

                save_message(
                    conversation_id,
                    user_id,
                    "user",
                    question,
                )

                save_message(
                    conversation_id,
                    user_id,
                    "assistant",
                    answer,
                    [],
                )

            return {
                "success": True,
                "conversation_id": conversation_id,
                "question": question,
                "answer": answer,
                "sources": [],
                "retrieved_chunks": 0,
                "retrieval_diagnostics": retrieval_diagnostics,
                "evaluation": {
                    "success": False,
                    "reason": "No retrieved context available.",
                    "metrics": {},
                },
                "langfuse_scores": {
                    "success": False,
                    "submitted": [],
                    "failed": [],
                    "skipped": [],
                },
            }

        # 6. Prepare context and sources.

        with tracker.track("context_preparation"):

            context, sources = format_retrieved_context(
                retrieved_documents
            )

        # 7. Initialize Groq.

        with tracker.track("llm_initialization"):

            llm = get_llm()

        # 8. Construct prompt.

        with tracker.track("prompt_construction"):

            prompt_messages = [
                SystemMessage(
                    content=build_system_prompt(context)
                )
            ]

            prompt_messages.extend(
                format_history_messages(history)
            )

            prompt_messages.append(
                HumanMessage(content=question)
            )

        # 9. Generate answer.

        with tracker.track("llm_generation"):

            with trace_span(
                "groq_generation",
                input={
                    "model": GROQ_MODEL,
                    "context_chunks": len(retrieved_documents),
                },
                metadata={"provider": "groq"},
            ) as generation_span:

                response = llm.invoke(prompt_messages)

                answer = response.content

                if isinstance(answer, list):
                    answer = "\n".join(
                        str(item) for item in answer
                    )

                elif not isinstance(answer, str):
                    answer = str(answer)

                if generation_span:

                    usage = (
                        getattr(response, "usage_metadata", None) or {}
                    )

                    generation_span.update(
                        output={
                            "answer_characters": len(answer)
                        },
                        metadata={
                            "usage": (
                                usage
                                if isinstance(usage, dict)
                                else str(usage)
                            ),
                            "model": GROQ_MODEL,
                        },
                    )

        # 10. Persist question and answer.

        with tracker.track("conversation_persistence"):

            save_message(
                conversation_id,
                user_id,
                "user",
                question,
            )

            save_message(
                conversation_id,
                user_id,
                "assistant",
                answer,
                sources,
            )

        # 11. Evaluate generated answer.

        evaluation_result = {
            "success": False,
            "reason": "Evaluation was not executed.",
            "metrics": {},
        }

        langfuse_score_report = {
            "success": False,
            "submitted": [],
            "failed": [],
            "skipped": [],
        }

        with tracker.track("answer_evaluation"):

            with trace_span(
                "answer_evaluation",
                input={
                    "context_chunks": len(retrieved_documents),
                    "evaluation_model": os.getenv(
                        "EVALUATION_MODEL",
                        GROQ_MODEL,
                    ),
                },
            ) as evaluation_span:

                evaluation_result = run_answer_evaluation(
                    question=question,
                    answer=answer,
                    context=context,
                )

                if evaluation_span:

                    evaluation_span.update(
                        output={
                            "evaluation_success": evaluation_result.get(
                                "success",
                                False,
                            ),
                            "metrics_available": list(
                                evaluation_result.get(
                                    "metrics",
                                    {},
                                ).keys()
                            ),
                        }
                    )

        # 12. Submit evaluation scores to the active Langfuse trace.

        if evaluation_result.get("success"):

            with tracker.track("langfuse_score_submission"):

                current_trace_id = get_current_trace_id()

                langfuse_score_report = log_evaluation_scores(
                    evaluation_results=evaluation_result,
                    trace_id=current_trace_id,
                )

        # 13. Return complete response.

        return {
            "success": True,
            "conversation_id": conversation_id,
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(retrieved_documents),
            "retrieval_diagnostics": retrieval_diagnostics,
            "evaluation": evaluation_result,
            "langfuse_scores": langfuse_score_report,
        }

    finally:

        tracker.report()


# --------------------------------------------------
# Langfuse-traced public entry point
# --------------------------------------------------

def answer_question(
    user_id: str,
    question: str,
    conversation_id: str = None,
    company: str = None,
    document_id: str = None,
):
    """
    Run one ATA chat request inside a Langfuse trace.

    Raw questions, document contents, prompts, generated answers,
    and email addresses are not sent to Langfuse by this integration.
    """

    normalized_user = user_id.strip().lower() if user_id else ""
    normalized_question = question.strip() if question else ""

    try:

        with trace_span(
            "ata_chat_request",
            input={
                "question_characters": len(normalized_question)
            },
            metadata={
                "user_id_hash": hashlib.sha256(
                    normalized_user.encode("utf-8")
                ).hexdigest()[:16],
                "conversation_id": conversation_id,
                "company_filter": bool(company),
                "document_filter": bool(document_id),
                "top_k": DEFAULT_TOP_K,
            },
        ) as root_span:

            result = _answer_question_impl(
                user_id=user_id,
                question=question,
                conversation_id=conversation_id,
                company=company,
                document_id=document_id,
            )

            if root_span:

                root_span.update(
                    output={
                        "success": result.get("success"),
                        "retrieved_chunks": result.get(
                            "retrieved_chunks",
                            0,
                        ),
                        "source_count": len(
                            result.get("sources", [])
                        ),
                        "answer_characters": len(
                            result.get("answer", "")
                        ),
                        "evaluation_success": result.get(
                            "evaluation",
                            {},
                        ).get("success", False),
                    },
                    metadata={
                        "conversation_id": result.get(
                            "conversation_id"
                        ),
                    },
                )

            return result

    finally:

        flush_traces()
