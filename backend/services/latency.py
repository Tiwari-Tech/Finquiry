
import time
import uuid
from contextlib import contextmanager


class LatencyTracker:
    """Track and report execution time for individual pipeline stages."""

    def __init__(self, pipeline_name: str = "RAG"):
        self.pipeline_name = pipeline_name
        self.request_id = str(uuid.uuid4())[:8]

        self.stages = []

        self.start_time_ns = time.perf_counter_ns()
        self.end_time_ns = None

    @contextmanager
    def track(self, stage_name: str):
        """
        Measure the execution time of an individual pipeline stage.

        Usage:
            with tracker.track("text_extraction"):
                extract_text()
        """
        stage_start_ns = time.perf_counter_ns()
        status = "SUCCESS"

        try:
            yield

        except Exception:
            status = "FAILED"
            raise

        finally:
            stage_end_ns = time.perf_counter_ns()

            duration_ms = (
                stage_end_ns - stage_start_ns
            ) / 1_000_000

            self.stages.append({
                "stage": stage_name,
                "start_ns": stage_start_ns,
                "end_ns": stage_end_ns,
                "duration_ms": duration_ms,
                "status": status
            })

    def add_stage(
        self,
        stage_name: str,
        duration_ms: float,
        status: str = "SUCCESS"
    ):
        """Add a manually measured stage."""

        self.stages.append({
            "stage": stage_name,
            "start_ns": None,
            "end_ns": None,
            "duration_ms": duration_ms,
            "status": status
        })

    def report(self):
        """Print and return the complete latency report."""

        # Capture total execution time only once.
        if self.end_time_ns is None:
            self.end_time_ns = time.perf_counter_ns()

        total_ms = (
            self.end_time_ns - self.start_time_ns
        ) / 1_000_000

        measured_stage_ms = sum(
            stage["duration_ms"]
            for stage in self.stages
        )

        untracked_ms = total_ms - measured_stage_ms

        print("\n" + "=" * 70)
        print(f"{self.pipeline_name} LATENCY REPORT")
        print("=" * 70)

        print(f"Request ID: {self.request_id}")
        print("-" * 70)

        for stage in self.stages:
            print(
                f"{stage['stage']:<42}"
                f"{stage['duration_ms']:>12.2f} ms"
                f"  [{stage['status']}]"
            )

        print("-" * 70)

        print(
            f"{'MEASURED STAGE TIME':<42}"
            f"{measured_stage_ms:>12.2f} ms"
        )

        print(
            f"{'UNTRACKED TIME':<42}"
            f"{untracked_ms:>12.2f} ms"
        )

        print("-" * 70)

        print(
            f"{'TOTAL EXECUTION TIME':<42}"
            f"{total_ms:>12.2f} ms"
        )

        print("=" * 70 + "\n")

        return {
            "request_id": self.request_id,
            "pipeline": self.pipeline_name,
            "total_ms": round(total_ms, 2),
            "measured_stage_ms": round(measured_stage_ms, 2),
            "untracked_ms": round(untracked_ms, 2),
            "stages": self.stages
        }
