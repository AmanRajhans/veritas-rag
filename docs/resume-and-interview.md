# Resume and interview notes

## Resume bullets

- Built an evaluation-first RAG service combining BM25 and local-semantic retrieval with reciprocal-rank fusion, diversity reranking, grounded citations, and calibrated abstention behind a typed FastAPI API; achieved 1.00 Recall@5/MRR/nDCG@5 on the bundled six-question synthetic evaluation.
- Designed a safe, reproducible ML delivery path with tamper-evident JSON indexes, index-aware SQLite caching, offline Recall@K/MRR/nDCG evaluation, Docker packaging, and GitHub Actions CI.

Re-run `make evaluate` before using the metrics, identify them as synthetic-demo results, and replace them with domain evaluation as the project evolves.

## Discussion prompts

**Why not use a vector database by default?** The repository's default should run offline in CI and on a recruiter's laptop. The retrieval interface and index artifact boundary make a production vector backend a contained replacement.

**How is hallucination controlled?** Generation is evidence-only, citation IDs are range-validated, and a confidence gate abstains when retrieval coverage or citations are weak. This reduces unsupported output; it does not prove factual correctness.

**How would you evaluate a learned generator?** Add claim-level citation entailment, answer correctness from expert labels, calibration error for the confidence score, slice metrics by question type, and regression gates against a versioned evaluation set.
