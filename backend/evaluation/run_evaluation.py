
"""
Finquiry Evaluation Runner

Responsibilities:
1. Load evaluation cases from evaluation_dataset.json.
2. Retrieve user-owned document chunks using hybrid retrieval.
3. Generate answers using the existing Groq configuration.
4. Evaluate answers using the evaluation service.
5. Send evaluation scores and traces to Langfuse.
6. Calculate aggregate evaluation metrics.

No local evaluation reports are generated.
No chat conversations or messages are created.
"""

import os
import json
import logging
import hashlib

from pathlib import Path

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)

from backend.services.retrieval_service import (
    search_user_documents,
)

from backend.services.chat_service import (
    get_llm,
    build_system_prompt,
    format_retrieved_context,
    GROQ_MODEL,
)

from backend.services.evaluation_service import (
    evaluate_answer,
    calculate_average_scores,
)

from backend.services.langfuse_service import (
    trace_span,
    get_current_trace_id,
    log_evaluation_scores,
    flush_traces,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = (
    PROJECT_ROOT
    / "backend"
    / "evaluation"
    / "evaluation_dataset.json"
)

DEFAULT_TOP_K = 6

TOP_K = int(
    os.getenv(
        "EVALUATION_TOP_K",
        DEFAULT_TOP_K,
    )
)


# --------------------------------------------------
# Dataset Loading
# --------------------------------------------------

def load_evaluation_dataset():
    """
    Load evaluation cases.

    Supported formats:

    1. A JSON list:
       [
           {
               "id": "case_001",
               "question": "...",
               "expected_answer": "..."
           }
       ]

    2. A JSON object:
       {
           "cases": [...]
       }

    No test cases are generated automatically.
    """

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    try:

        with DATASET_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:

            dataset = json.load(file)

    except json.JSONDecodeError as exc:

        raise ValueError(
            "evaluation_dataset.json contains invalid JSON."
        ) from exc

    if isinstance(dataset, dict):

        dataset = dataset.get("cases", [])

    if not isinstance(dataset, list):

        raise ValueError(
            "Evaluation dataset must be a JSON list "
            "or an object containing a 'cases' list."
        )

    validated_cases = []

    for index, case in enumerate(dataset, start=1):

        if not isinstance(case, dict):

            raise ValueError(
                f"Evaluation case {index} must be a JSON object."
            )

        question = case.get("question")

        if not isinstance(question, str) or not question.strip():

            raise ValueError(
                f"Evaluation case {index} has an invalid question."
            )

        expected_answer = case.get("expected_answer")

        if expected_answer is not None:

            if not isinstance(expected_answer, str):

                raise ValueError(
                    f"Evaluation case {index}: "
                    "expected_answer must be a string."
                )

            expected_answer = expected_answer.strip() or None

        validated_cases.append(
            {
                **case,
                "id": str(
                    case.get("id", f"case_{index:03d}")
                ),
                "question": question.strip(),
                "expected_answer": expected_answer,
            }
        )

    return validated_cases


# --------------------------------------------------
# Answer Generation
# --------------------------------------------------

def generate_answer(question, context):
    """
    Generate an answer using the same Groq model
    and system prompt as the existing chat service.

    This function does not save conversation history.
    """

    llm = get_llm()

    messages = [
        SystemMessage(
            content=build_system_prompt(context)
        ),
        HumanMessage(
            content=question
        ),
    ]

    response = llm.invoke(messages)

    answer = response.content

    if isinstance(answer, list):

        answer = "\n".join(
            str(item)
            for item in answer
        )

    elif not isinstance(answer, str):

        answer = str(answer)

    answer = answer.strip()

    if not answer:

        raise ValueError(
            "The LLM returned an empty answer."
        )

    return answer


# --------------------------------------------------
# Single Evaluation Case
# --------------------------------------------------

def evaluate_case(
    case,
    user_email,
    top_k=DEFAULT_TOP_K,
):
    """
    Execute retrieval, generation and evaluation
    for one dataset case.

    Every case receives its own Langfuse trace.
    """

    case_id = case["id"]
    question = case["question"]

    expected_answer = case.get(
        "expected_answer"
    )

    # Never send the actual email address to Langfuse.
    user_hash = hashlib.sha256(
        user_email.strip().lower().encode("utf-8")
    ).hexdigest()[:16]

    with trace_span(
        name="finquiry_evaluation_case",
        input={
            "question_characters": len(question),
        },
        metadata={
            "evaluation_case_id": case_id,
            "user_id_hash": user_hash,
            "top_k": top_k,
            "model": GROQ_MODEL,
            "has_expected_answer": bool(expected_answer),
        },
    ) as case_span:

        try:

            # 1. Hybrid retrieval.

            with trace_span(
                name="benchmark_retrieval",
                input={
                    "top_k": top_k,
                    "company_filter": bool(case.get("company")),
                    "document_filter": bool(case.get("document_id")),
                },
            ) as retrieval_span:

                retrieved_documents = search_user_documents(
                    email=user_email,
                    question=question,
                    company=case.get("company"),
                    document_id=case.get("document_id"),
                    k=top_k,
                )

                if retrieval_span:

                    retrieval_span.update(
                        output={
                            "retrieved_chunks": len(
                                retrieved_documents
                            ),
                        }
                    )

            # 2. Handle empty retrieval.

            if not retrieved_documents:

                if case_span:

                    case_span.update(
                        output={
                            "status": "no_retrieved_context",
                            "retrieved_chunks": 0,
                        }
                    )

                return {
                    "case_id": case_id,
                    "success": False,
                    "reason": "No relevant documents retrieved.",
                    "evaluation": None,
                }

            # 3. Prepare context.

            context, sources = format_retrieved_context(
                retrieved_documents
            )

            if not context.strip():

                raise ValueError(
                    "Retrieved documents produced empty context."
                )

            # 4. Generate answer.

            with trace_span(
                name="benchmark_generation",
                input={
                    "context_chunks": len(
                        retrieved_documents
                    ),
                    "model": GROQ_MODEL,
                },
                metadata={
                    "provider": "groq",
                },
            ) as generation_span:

                answer = generate_answer(
                    question=question,
                    context=context,
                )

                if generation_span:

                    generation_span.update(
                        output={
                            "answer_characters": len(answer),
                        }
                    )

            # 5. Evaluate generated answer.

            with trace_span(
                name="benchmark_evaluation",
                input={
                    "context_chunks": len(
                        retrieved_documents
                    ),
                    "has_expected_answer": bool(
                        expected_answer
                    ),
                },
            ) as evaluation_span:

                evaluation_result = evaluate_answer(
                    question=question,
                    answer=answer,
                    context=context,
                    expected_answer=expected_answer,
                )

                if evaluation_span:

                    evaluation_span.update(
                        output={
                            "evaluation_success": (
                                evaluation_result.get("success", False)
                            ),
                            "available_metrics": list(
                                evaluation_result.get(
                                    "metrics",
                                    {},
                                ).keys()
                            ),
                            "evaluation_type": (
                                evaluation_result.get(
                                    "evaluation_type"
                                )
                            ),
                        }
                    )

            # 6. Submit scores to Langfuse.

            langfuse_report = {
                "success": False,
                "submitted": [],
                "failed": [],
                "skipped": [],
            }

            if evaluation_result.get("success"):

                current_trace_id = get_current_trace_id()

                langfuse_report = log_evaluation_scores(
                    evaluation_results=evaluation_result,
                    trace_id=current_trace_id,
                )

            # 7. Update case trace.

            if case_span:

                case_span.update(
                    output={
                        "status": (
                            "completed"
                            if evaluation_result.get("success")
                            else "evaluation_failed"
                        ),
                        "retrieved_chunks": len(
                            retrieved_documents
                        ),
                        "source_count": len(sources),
                        "evaluation_success": (
                            evaluation_result.get("success", False)
                        ),
                        "scores_submitted": len(
                            langfuse_report.get("submitted", [])
                        ),
                    }
                )

            return {
                "case_id": case_id,
                "success": bool(
                    evaluation_result.get("success")
                ),
                "evaluation": evaluation_result,
                "langfuse": langfuse_report,
            }

        except Exception as exc:

            logger.exception(
                "Evaluation case %s failed.",
                case_id,
            )

            if case_span:

                case_span.update(
                    output={
                        "status": "failed",
                        "error_type": type(exc).__name__,
                    }
                )

            return {
                "case_id": case_id,
                "success": False,
                "reason": str(exc),
                "evaluation": None,
            }


# --------------------------------------------------
# Aggregate Evaluation Summary
# --------------------------------------------------

def print_evaluation_summary(results):
    """
    Calculate and print aggregate evaluation metrics.

    This summary is displayed in the terminal only.
    No local report files are generated.
    """

    # Extract evaluation results.
    # Cases that fail before evaluation are counted
    # as failed evaluations.

    evaluation_results = []

    for result in results:

        evaluation = result.get("evaluation")

        if isinstance(evaluation, dict):

            evaluation_results.append(evaluation)

        else:

            evaluation_results.append(
                {
                    "success": False,
                    "metrics": {},
                }
            )

    summary = calculate_average_scores(
        evaluation_results
    )

    print("\n" + "=" * 60)
    print("EVALUATION EXECUTION SUMMARY")
    print("=" * 60)

    print(
        f"Total cases: {summary['total_evaluations']}"
    )

    print(
        f"Successful evaluations: "
        f"{summary['successful_evaluations']}"
    )

    print(
        f"Failed evaluations: "
        f"{summary['failed_evaluations']}"
    )

    print("\nAVERAGE EVALUATION METRICS")
    print("-" * 60)

    averages = summary["average_scores"]
    sample_counts = summary["metric_sample_counts"]

    metric_labels = {
        "answer_relevance": "Answer Relevance",
        "faithfulness": "Faithfulness",
        "completeness": "Completeness",
        "answer_correctness": "Answer Correctness",
    }

    for metric, label in metric_labels.items():

        score = averages.get(metric)
        sample_count = sample_counts.get(metric, 0)

        if score is None:

            print(
                f"{label:<25} Not available "
                f"(0 samples)"
            )

        else:

            print(
                f"{label:<25} {score:.4f} "
                f"({score * 100:.2f}%) "
                f"[{sample_count} samples]"
            )

    print("\n" + "=" * 60)

    return summary


# --------------------------------------------------
# Benchmark Runner
# --------------------------------------------------

def run_evaluation():
    """
    Run the complete evaluation benchmark.

    All traces and scores are sent to Langfuse.
    No local JSON or CSV report is created.
    """

    print("\n" + "=" * 60)
    print("FINQUIRY EVALUATION BENCHMARK")
    print("=" * 60)

    # 1. Load dataset.

    cases = load_evaluation_dataset()

    if not cases:

        print(
            "\nNo evaluation cases found in "
            "evaluation_dataset.json."
        )

        print(
            "Add verified evaluation cases before "
            "running the benchmark."
        )

        return

    # 2. Validate evaluation user.

    user_email = (
        os.getenv("ATA_EVALUATION_USER_EMAIL")
        or os.getenv("EVALUATION_USER_EMAIL")
        or ""
    ).strip().lower()

    if not user_email:

        raise ValueError(
            "Evaluation user is not configured. "
            "Set ATA_EVALUATION_USER_EMAIL in your .env file."
        )

    if TOP_K < 1:

        raise ValueError(
            "EVALUATION_TOP_K must be at least 1."
        )

    print(f"\nDataset cases: {len(cases)}")
    print(f"Retrieval top_k: {TOP_K}")
    print("Storage: Langfuse only")

    results = []

    try:

        # 3. Execute each evaluation case.

        for index, case in enumerate(cases, start=1):

            print(
                f"\n[{index}/{len(cases)}] "
                f"Running case: {case['id']}"
            )

            result = evaluate_case(
                case=case,
                user_email=user_email,
                top_k=TOP_K,
            )

            results.append(result)

            if result["success"]:

                print(
                    f"Case {case['id']}: Evaluation completed."
                )

                report = result.get("langfuse", {})

                print(
                    "Langfuse scores submitted: "
                    f"{len(report.get('submitted', []))}"
                )

            else:

                print(
                    f"Case {case['id']}: Failed - "
                    f"{result.get('reason', 'Evaluation failed.')}"
                )

    finally:

        # 4. Flush traces and scores.

        flush_traces()

    # 5. Calculate and display aggregate metrics.

    print_evaluation_summary(results)

    print("\nCheck Langfuse for individual traces and scores.")
    print("No local evaluation report was generated.")


# --------------------------------------------------
# Entry Point
# --------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    try:

        run_evaluation()

    except Exception:

        logger.exception(
            "Finquiry evaluation benchmark could not start."
        )

        raise
