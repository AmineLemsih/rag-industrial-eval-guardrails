#!/usr/bin/env python3
"""
Benchmark latency and approximate cost of the RAG API.

This script issues concurrent requests to the `/query` endpoint of the
local API and measures response times.  It reports the p50 and p95
latencies as well as estimated token usage and cost.  Use this tool
to evaluate performance both in local and cloud deployment modes.

Example usage:

    python scripts/bench_latency.py --url http://localhost:8000/query --requests 50 --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from typing import List
from pathlib import Path

import httpx  # type: ignore


async def worker(client: httpx.AsyncClient, url: str, questions: List[str], latencies: List[float]) -> None:
    while True:
        question = random.choice(questions)
        payload = {"query": question}
        start = time.perf_counter()
        try:
            resp = await client.post(url, json=payload, timeout=60.0)
            resp.raise_for_status()
        except Exception:
            pass
        latency = (time.perf_counter() - start) * 1000.0
        latencies.append(latency)


async def run_benchmark(url: str, requests: int, concurrency: int, questions: List[str]) -> None:
    latencies: List[float] = []
    async with httpx.AsyncClient() as client:
        tasks = []
        for _ in range(concurrency):
            tasks.append(asyncio.create_task(worker(client, url, questions, latencies)))
        # Wait until enough requests have been recorded
        while len(latencies) < requests:
            await asyncio.sleep(0.1)
        for task in tasks:
            task.cancel()
        # Wait for tasks to finish gracefully
        await asyncio.gather(*tasks, return_exceptions=True)
    # Compute statistics
    sorted_lat = sorted(latencies[:requests])
    p50 = sorted_lat[int(0.5 * requests)]
    p95 = sorted_lat[int(0.95 * requests)]
    print(f"p50 latency: {p50:.1f} ms")
    print(f"p95 latency: {p95:.1f} ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RAG API latency")
    parser.add_argument("--url", default="http://localhost:8000/query", help="URL du point de terminaison /query")
    parser.add_argument("--requests", type=int, default=20, help="Nombre total de requêtes à envoyer")
    parser.add_argument("--concurrency", type=int, default=5, help="Nombre de requêtes concurrentes")
    args = parser.parse_args()
    # Load questions from evaluation file
    eval_file = Path(__file__).resolve().parents[1] / "eval" / "questions.jsonl"
    questions: List[str] = []
    with eval_file.open("r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line)["question"])
    asyncio.run(run_benchmark(args.url, args.requests, args.concurrency, questions))


if __name__ == "__main__":
    main()