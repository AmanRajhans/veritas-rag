"""Command-line interface for indexing, asking, evaluating, and serving."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from veritas_rag.cache import ResponseCache
from veritas_rag.chunking import chunk_documents, load_documents
from veritas_rag.evaluation import evaluate, load_cases
from veritas_rag.generation import OpenAICompatibleGenerator
from veritas_rag.index import CorpusIndex
from veritas_rag.pipeline import VeritasPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veritas-rag", description="Evidence-first hybrid RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="build a safe JSON index")
    index_parser.add_argument("source", type=Path)
    index_parser.add_argument("--output", type=Path, default=Path("artifacts/index.json"))
    index_parser.add_argument("--target-words", type=int, default=140)
    index_parser.add_argument("--overlap-words", type=int, default=30)

    ask_parser = subparsers.add_parser("ask", help="query an existing index")
    ask_parser.add_argument("query")
    ask_parser.add_argument("--index", type=Path, default=Path("artifacts/index.json"))
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--trace", action="store_true")
    ask_parser.add_argument("--no-cache", action="store_true")

    eval_parser = subparsers.add_parser("evaluate", help="run a JSONL evaluation set")
    eval_parser.add_argument("dataset", type=Path)
    eval_parser.add_argument("--index", type=Path, default=Path("artifacts/index.json"))
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--output", type=Path)

    serve_parser = subparsers.add_parser("serve", help="start the FastAPI service")
    serve_parser.add_argument("--index", type=Path, default=Path("artifacts/index.json"))
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    return parser


def _pipeline(index_path: Path, use_cache: bool = True) -> VeritasPipeline:
    generator = None
    if os.getenv("VERITAS_GENERATOR") == "openai-compatible":
        generator = OpenAICompatibleGenerator(
            base_url=os.getenv("VERITAS_LLM_BASE_URL", "http://localhost:11434/v1"),
            model=os.getenv("VERITAS_LLM_MODEL", "qwen2.5:7b"),
            api_key=os.getenv("VERITAS_LLM_API_KEY", ""),
        )
    cache = ResponseCache(Path("artifacts/cache.sqlite3")) if use_cache else None
    return VeritasPipeline(CorpusIndex.load(index_path), generator=generator, cache=cache)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command == "index":
        documents = load_documents(arguments.source)
        chunks = chunk_documents(documents, arguments.target_words, arguments.overlap_words)
        index = CorpusIndex(chunks)
        index.save(arguments.output)
        print(
            json.dumps(
                {
                    "documents": len(documents),
                    "chunks": len(chunks),
                    "fingerprint": index.fingerprint,
                    "path": str(arguments.output),
                },
                indent=2,
            )
        )
    elif arguments.command == "ask":
        response = _pipeline(arguments.index, use_cache=not arguments.no_cache).query(
            arguments.query, top_k=arguments.top_k, include_trace=arguments.trace
        )
        print(response.model_dump_json(indent=2))
    elif arguments.command == "evaluate":
        report = evaluate(
            _pipeline(arguments.index, use_cache=False),
            load_cases(arguments.dataset),
            arguments.top_k,
        )
        output = json.dumps(report, indent=2)
        if arguments.output:
            arguments.output.parent.mkdir(parents=True, exist_ok=True)
            arguments.output.write_text(output + "\n", encoding="utf-8")
        print(output)
    elif arguments.command == "serve":
        import uvicorn

        os.environ["VERITAS_INDEX_PATH"] = str(arguments.index)
        uvicorn.run(
            "veritas_rag.api:app",
            host=arguments.host,
            port=arguments.port,
            reload=arguments.reload,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
