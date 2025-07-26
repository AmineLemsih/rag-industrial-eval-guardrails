## Makefile for rag-industrial-eval-guardrails
#
# This Makefile provides convenient shortcuts for setting up the
# development environment, running the stack, ingesting documents,
# evaluating the pipeline and benchmarking latency.

.PHONY: setup compose ingest query eval bench test clean

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -e .[dev]

compose:
	docker compose -f docker-compose.yml up -d --build

ingest:
	python scripts/generate_synthetic_corpus.py
	python scripts/ingest.py

query:
	@echo "Example query. Replace the question as needed."
	curl -s -X POST -H "Content-Type: application/json" -d '{"query":"Quelle est la durée de validité des mots de passe ?"}' http://localhost:8000/query | python -m json.tool

eval:
	python scripts/evaluate_ragas.py

bench:
	python scripts/bench_latency.py --url http://localhost:8000/query --requests 20 --concurrency 5

test:
	pytest -q

clean:
	rm -rf .venv
	rm -rf reports
	docker compose down -v