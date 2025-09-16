"""GUIDE smoke test.

1. Runs ``QueryService`` once to ensure the core agentic flow can answer a
   question using the local data files.
2. Starts a temporary Uvicorn instance and checks ``/health`` and ``/docs`` to
   confirm the FastAPI application responds.

Execute from the project root after installing requirements and creating
``.env`` (needs a valid ``OPENAI_API_KEY``).
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from app.models.query import QueryRequest
from app.services.query_service import AliasSelectionRequiredError, QueryService


async def run_service_demo() -> None:
    """Invoke ``QueryService`` to verify the agentic pipeline."""

    service = QueryService()
    request = QueryRequest(
        question="What is V2G?",
        graph_path="data/output/optimized_entity_graph.json",
        subgraph_distance=2,
        use_agentic_flow=True,
    )
    response = await service.ask_question(request)
    print("\n🔍 QueryService demo result:\n", response.model_dump(), "\n", sep="")

    # Demonstrate alias disambiguation flow with an abbreviated entity.
    ambiguous_question = "What is PA?"
    try:
        await service.ask_question(
            QueryRequest(
                question=ambiguous_question,
                graph_path="data/output/optimized_entity_graph.json",
                subgraph_distance=2,
            )
        )
    except AliasSelectionRequiredError as exc:
        print("🔁 Alias confirmation required:")
        for alias, candidates in exc.aliases.items():
            print(f"  {alias} -> {candidates}")
        # Use the first suggestion automatically for the demo.
        alias, candidates = next(iter(exc.aliases.items()))
        chosen = candidates[0]
        confirmed = await service.ask_question(
            QueryRequest(
                question=ambiguous_question,
                graph_path="data/output/optimized_entity_graph.json",
                subgraph_distance=2,
                alias_overrides={alias: chosen},
            )
        )
        print("✅ Resubmitted with alias_overrides:")
        print(confirmed.model_dump())


def wait_for(url: str, proc: subprocess.Popen[str], timeout: float = 20.0) -> int:
    """Poll an HTTP URL; surface Uvicorn errors if the process stops early."""

    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(
                "Uvicorn process exited unexpectedly.\n"
                f"Return code: {proc.returncode}\n"
                f"Output:\n{output.strip()}"
            )

        try:
            with urllib.request.urlopen(url) as resp:  # noqa: S310
                return resp.status
        except urllib.error.URLError:
            time.sleep(0.5)
    raise RuntimeError(f"Timeout waiting for {url}")


def run_api_probe() -> None:
    """Start Uvicorn briefly and hit /health and /docs."""

    print("🚀 Launching temporary Uvicorn server (http://127.0.0.1:8000)...")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--log-level",
        "warning",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        wait_for("http://127.0.0.1:8000/health", proc)
        docs_status = wait_for("http://127.0.0.1:8000/docs", proc)
        print(f"✅ API health check OK (docs status: {docs_status})")
    finally:
        with contextlib.suppress(ProcessLookupError, TimeoutError):
            proc.terminate()
            proc.wait(timeout=5)
        if proc.stdout:
            with contextlib.suppress(Exception):
                proc.stdout.read()


def main() -> None:
    project_root = Path(__file__).resolve().parent
    print(f"GUIDE root: {project_root}")

    asyncio.run(run_service_demo())
    run_api_probe()


if __name__ == "__main__":
    main()
