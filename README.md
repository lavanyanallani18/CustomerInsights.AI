# Silicon Signals

An evidence-first Streamlit dashboard for comparing NVDA, AMD, and INTC using
public SEC filings. It presents comparative metrics, input/source traceability,
RAG cited chat-style Q&A, conflict review, and transparent evaluation checks. When
prepared data or optional adapters are absent, the app stays runnable and
explains the missing preparation step instead of showing invented values.

## Screenshots

### Overview

![Overview dashboard showing latest peer metrics and trend chart](images/overview.png)

### Comparable Metrics

![Compare tab showing one source-traceable observation per company](images/compare.png)

### Source Traceability

![Traceability tab showing SEC accession and source links for each metric](images/traceability.png)

### RAG Cited Q&A

![RAG Cited Q&A tab showing chat history, computed answer, and citations](images/rag-cited-qa.png)

### Evaluation

![Evaluation tab showing coverage, source-link rate, data-quality flags, and labeled Q&A checks](images/evaluation.png)

## Quick start

```bash
cd CustomerInsights_AI
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.
The checked-in workspace already contains the prepared SEC artifacts. The
**RAG Cited Q&A** tab behaves like a small conversational filing assistant: it
keeps up to 10 questions, supports follow-ups, and shows citations under every
answer. Q&A uses a Hugging Face Llama-family model when available and falls back
to deterministic cited extraction if the model cannot load.

If `chroma_db/` is not present, rebuild the local RAG index first:

```bash
python scripts/prepare_data.py --data-dir data/sec --offline --force
```

## Prepare data

The repository's preparation script downloads/caches public SEC submissions,
companyfacts, and filing HTML, then writes a validated analytics bundle at
`data/sec/prepared_analytics.json` and rebuilds the local Chroma index at
`chroma_db`.

```bash
source venv/bin/activate
export SEC_USER_AGENT="Your Name your.email@example.com"
python scripts/prepare_data.py --data-dir data/sec
```

To validate an existing offline cache without network access:

```bash
source venv/bin/activate
python scripts/prepare_data.py --data-dir data/sec --offline --force
```

This validates the prepared analytics dataset and rebuilds `chroma_db` for
RAG Cited Q&A. To refresh from SEC EDGAR instead, use the non-offline command above.

Current prepared artifact summary:

- 3 comparable semiconductor companies: NVDA, AMD, INTC
- 15 public SEC filings total: 5 per company, 12 10-Qs and 3 10-Ks
- 417 normalized SEC companyfacts records
- 284 derived analytics calculations with formula/input provenance
- 43 anomaly records from duplicate/restated/conflicting facts
- 4,440 filing chunks indexed in local Chroma for RAG Cited Q&A

The presentation layer consumes:

- `src.analytics.dataset.load_prepared_dataset()`; or
- a zero-argument `get_metrics_data()`, `load_metrics()`, or `get_metrics()`
  from `src.analytics.metrics` (also accepts `src.metrics.metrics`); or
- Prepared JSON at `data/prepared/metrics.json`, `data/prepared_metrics.json`,
  `data/metrics.json`, `data/sec/prepared_analytics.json`,
  `prepared/metrics.json`, `outputs/metrics.json`, or `artifacts/metrics.json`;
  and
- `src.rag.qa.ask_question(...)` or `answer_question(...)` for RAG Cited Q&A.

Minimal prepared JSON:

```json
{
  "metrics": [
    {
      "ticker": "NVDA",
      "metric": "Revenue",
      "period": "2025",
      "value": 1,
      "unit": "USD",
      "form": "10-K",
      "accession": "0000000000-00-000000",
      "source_url": "https://www.sec.gov/..."
    }
  ],
  "conflicts": [],
  "evaluation": []
}
```

The normalizer also accepts common aliases such as `amount`, `fiscal_year`,
`filing_type`, and `url`, plus nested ticker/year structures.

## Run and verify

```bash
source venv/bin/activate
python -m compileall -q src scripts evaluation tests app.py
python -m pytest -q
python -m evaluation.runner --output evaluation/results.json
python -m pip check
streamlit run app.py
```

Latest evaluation (`evaluation/results.json`): 4 labeled questions, answer
correctness `1.0`, citation accuracy `1.0`, hallucination rate `0.0`.

For a headless smoke test:

```bash
source venv/bin/activate
streamlit run app.py --server.headless true --server.port 8501
```

## Q&A Model

Q&A uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` from Hugging Face when available.
No external LLM API key is required. If the Hugging Face model cannot load, the
app falls back to deterministic cited extraction.

By default, `HF_LLM_LOCAL_ONLY=true`, so the app will not pause to download a
model during a demo. To download/use TinyLlama through Transformers, set
`HF_LLM_LOCAL_ONLY=false` in `.env` and restart Streamlit.

The RAG Cited Q&A tab keeps chat history for up to 10 questions. You can ask
follow-ups like "tell me more" or "what about operating income"; the app uses
the previous turn as lightweight context and keeps citations visible under each
assistant answer.

## Public sources

- SEC company submissions JSON: `https://data.sec.gov/submissions/`
- SEC filing archives: `https://www.sec.gov/Archives/edgar/data/`
- SEC company filing browser: `https://www.sec.gov/edgar/browse/`

SEC data is public, but automated access must use a descriptive User-Agent and
respect SEC fair-access guidance. The dashboard never fetches filings on page
load; preparation is intentionally separate and reproducible.

## Architecture

```text
SEC EDGAR -> ingestion/parser -> prepared metrics + vector index
                                      |              |
                            analytics adapter     Q&A adapter
                                      |              |
                                      +---- app.py --+
                                            |
                         overview / compare / trace / quality / eval
```

`app.py` is a read-only presentation adapter. It normalizes prepared records,
derives deterministic readiness checks, and gracefully isolates optional
analytics and Q&A imports. See [WRITEUP.md](WRITEUP.md) for design tradeoffs,
evaluation strategy, diagnosis, and implementation details.

## Security

Copy `.env.example` to `.env` only when you want to override the Hugging Face
model settings. Never commit `.env` or `.streamlit/secrets.toml`. The app
renders sanitized operational errors and only turns official `sec.gov` URLs into
source links. See [SECURITY.md](SECURITY.md).

## Submission Notes

Do not include `venv/`, `__pycache__/`, `.pytest_cache/`, or `chroma_db/` in a
source submission. They are generated locally. Include the source files,
`requirements.txt`, docs, `evaluation/`, and `data/sec/` if you want reviewers
to run the offline rebuild command without re-downloading SEC filings.
