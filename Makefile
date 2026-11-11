.PHONY: install test lint demo api evaluate docker

install:
	uv sync --dev

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

demo:
	uv run veritas-rag index data/demo --output artifacts/demo-index.json
	uv run veritas-rag ask "When do employees need manager approval for expenses?" --index artifacts/demo-index.json

api:
	uv run veritas-rag serve --index artifacts/demo-index.json --reload

evaluate:
	uv run veritas-rag evaluate data/eval/questions.jsonl --index artifacts/demo-index.json

docker:
	docker build -t veritas-rag:local .
