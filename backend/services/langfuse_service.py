
"""Privacy-conscious Langfuse integration for ATA.

Tracing and evaluation scoring are optional. Missing credentials, SDK
incompatibilities, or Langfuse service errors must never interrupt the
application's RAG or chat workflow.

By default, ATA records operational metadata and evaluation scores, not
retrieved document contents, full prompts, or generated answer text.
"""

import logging
import math
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

_client = None
_client_attempted = False


def get_langfuse_client():
    """Return the Langfuse client, initializing it only once."""
    global _client, _client_attempted

    if _client_attempted:
        return _client

    _client_attempted = True

    if not (
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
    ):
        logger.info(
            "Langfuse tracing disabled: API keys are not configured."
        )
        return None

    try:
        from langfuse import get_client

        _client = get_client()

    except Exception:
        logger.exception(
            "Could not initialize Langfuse. Tracing is disabled."
        )
        _client = None

    return _client


class _NoOpSpan:
    """Fallback span object used when tracing is unavailable."""

    def update(self, **kwargs):
        return None


@contextmanager
def trace_span(
    name: str,
    input: dict | None = None,
    metadata: dict | None = None,
):
    """Create a Langfuse span or yield None if tracing is unavailable."""

    client = get_langfuse_client()

    if client is None:
        yield None
        return

    observation_context = None

    try:
        observation_context = client.start_as_current_observation(
            as_type="span",
            name=name,
            input=input or {},
            metadata=metadata or {},
        )

        span = observation_context.__enter__()

    except Exception:
        logger.exception(
            "Could not start Langfuse span '%s'.",
            name,
        )
        yield None
        return

    try:
        yield span

    except BaseException as exc:
        try:
            observation_context.__exit__(
                type(exc),
                exc,
                exc.__traceback__,
            )
        except Exception:
            logger.exception(
                "Could not close Langfuse span '%s' after an error.",
                name,
            )

        raise

    else:
        try:
            observation_context.__exit__(None, None, None)

        except Exception:
            logger.exception(
                "Could not close Langfuse span '%s'.",
                name,
            )


def get_current_trace_id() -> str | None:
    """Get the active Langfuse trace ID, if one exists.

    Call this from inside an active trace/span context.
    """

    client = get_langfuse_client()

    if client is None:
        return None

    try:
        getter = getattr(client, "get_current_trace_id", None)

        if not callable(getter):
            return None

        trace_id = getter()

        return str(trace_id) if trace_id else None

    except Exception:
        logger.exception(
            "Could not retrieve the current Langfuse trace ID."
        )
        return None


def create_trace_score(
    name: str,
    value: float,
    comment: str | None = None,
    metadata: dict | None = None,
    trace_id: str | None = None,
) -> bool:
    """Attach a numeric evaluation score to a Langfuse trace.

    Preferred path:
        client.score_current_trace(...)

    Fallback:
        client.create_score(..., trace_id=...)

    Returns True when the SDK accepts the score, otherwise False.

    Scores should be numeric values between 0 and 1 for ATA's normalized
    evaluation metrics.
    """

    client = get_langfuse_client()

    if client is None:
        return False

    if not name or not isinstance(name, str):
        logger.warning("Langfuse score skipped: invalid score name.")
        return False

    try:
        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            logger.warning(
                "Langfuse score '%s' skipped: value is not finite.",
                name,
            )
            return False

        if not 0.0 <= numeric_value <= 1.0:
            logger.warning(
                "Langfuse score '%s' skipped: expected a value "
                "between 0 and 1, received %s.",
                name,
                numeric_value,
            )
            return False

    except (TypeError, ValueError):
        logger.warning(
            "Langfuse score '%s' skipped: invalid numeric value.",
            name,
        )
        return False

    score_comment = comment or ""
    score_metadata = metadata or {}

    # Preferred method: associate the score with the active trace.
    score_current_trace = getattr(
        client,
        "score_current_trace",
        None,
    )

    if callable(score_current_trace):
        try:
            score_current_trace(
                name=name,
                value=numeric_value,
                comment=score_comment,
                metadata=score_metadata,
            )

            return True

        except TypeError:
            # Some SDK versions may not accept metadata.
            try:
                score_current_trace(
                    name=name,
                    value=numeric_value,
                    comment=score_comment,
                )

                return True

            except Exception:
                logger.exception(
                    "Could not create Langfuse score '%s' "
                    "using score_current_trace.",
                    name,
                )

        except Exception:
            logger.exception(
                "Could not create Langfuse score '%s' "
                "using score_current_trace.",
                name,
            )

    # Fallback: explicitly associate the score with a trace ID.
    resolved_trace_id = trace_id or get_current_trace_id()

    create_score = getattr(client, "create_score", None)

    if not callable(create_score):
        logger.warning(
            "Langfuse scoring unavailable: SDK does not expose "
            "score_current_trace or create_score."
        )
        return False

    if not resolved_trace_id:
        logger.warning(
            "Langfuse score '%s' skipped: no active trace ID is available.",
            name,
        )
        return False

    try:
        create_score(
            name=name,
            value=numeric_value,
            trace_id=resolved_trace_id,
            comment=score_comment,
            metadata=score_metadata,
        )

        return True

    except TypeError:
        # Compatibility fallback for SDK versions without metadata support.
        try:
            create_score(
                name=name,
                value=numeric_value,
                trace_id=resolved_trace_id,
                comment=score_comment,
            )

            return True

        except Exception:
            logger.exception(
                "Could not create Langfuse score '%s'.",
                name,
            )
            return False

    except Exception:
        logger.exception(
            "Could not create Langfuse score '%s'.",
            name,
        )
        return False


def log_evaluation_scores(
    evaluation_results: dict,
    trace_id: str | None = None,
) -> dict:
    """Send available normalized evaluation metrics to Langfuse.

    Expected input example:
        {
            "success": True,
            "metrics": {
                "answer_relevance": 0.90,
                "faithfulness": 0.85
            }
        }

    Returns a report showing which scores were submitted successfully.
    """

    report = {
        "success": False,
        "submitted": [],
        "failed": [],
        "skipped": [],
    }

    if not isinstance(evaluation_results, dict):
        report["skipped"].append("Invalid evaluation result.")
        return report

    if not evaluation_results.get("success"):
        report["skipped"].append(
            "Evaluation was unsuccessful or returned no scores."
        )
        return report

    metrics = evaluation_results.get("metrics", {})

    if not isinstance(metrics, dict) or not metrics:
        report["skipped"].append("No evaluation metrics were returned.")
        return report

    for metric_name, metric_value in metrics.items():

        if metric_value is None:
            report["skipped"].append(metric_name)
            continue

        try:
            numeric_value = float(metric_value)

            if not math.isfinite(numeric_value):
                report["skipped"].append(metric_name)
                continue

            if not 0.0 <= numeric_value <= 1.0:
                report["skipped"].append(metric_name)
                continue

            submitted = create_trace_score(
                name=metric_name,
                value=numeric_value,
                comment="ATA automated evaluation",
                metadata={
                    "evaluation_type": evaluation_results.get(
                        "evaluation_type",
                        "automated",
                    ),
                    "evaluation_model": evaluation_results.get(
                        "model",
                        "unknown",
                    ),
                },
                trace_id=trace_id,
            )

            if submitted:
                report["submitted"].append(metric_name)
            else:
                report["failed"].append(metric_name)

        except (TypeError, ValueError):
            report["skipped"].append(metric_name)

        except Exception:
            logger.exception(
                "Unexpected error while logging metric '%s'.",
                metric_name,
            )
            report["failed"].append(metric_name)

    report["success"] = len(report["submitted"]) > 0

    return report


def flush_traces():
    """Flush buffered Langfuse traces and scores."""

    client = get_langfuse_client()

    if client is None:
        return

    try:
        client.flush()

    except Exception:
        logger.exception("Could not flush Langfuse events.")
