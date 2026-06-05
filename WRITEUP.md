# WRITEUP

## Architecture Approach

I built **Silicon Signals**, a Streamlit dashboard for comparing three public
semiconductor peers: **NVIDIA (NVDA), AMD, and Intel (INTC)**. The corpus uses
exactly **15 public SEC filings**, five recent 10-K/10-Q filings per company,
which matches the assignment's requested 10-15 filings across 3-5 comparable
companies.

The system separates slow preparation from the interactive dashboard:

- `src/ingestion/edgar_fetcher.py` selects filings and caches official SEC
  submissions, companyfacts, and filing HTML.
- `src/analytics/normalization.py` converts SEC companyfacts into normalized
  facts with ticker, concept, period, unit, accession, form, filed date, and SEC
  source URL.
- `src/analytics/metrics.py` computes reproducible derived metrics: YoY growth,
  gross/operating/net margin, free cash flow, and debt-to-equity.
- `src/ingestion/parser.py`, `src/rag/vectorstore.py`,
  `src/rag/retriever.py`, and `src/rag/qa.py` build the RAG layer from filing
  HTML chunks.
- `app.py` presents the dashboard: Overview, Compare, Traceability, RAG Cited
  Q&A, Data quality, and Evaluation.

I used SEC companyfacts/XBRL-style JSON for financial numbers and SEC HTML
filings for narrative evidence. The PDF allows HTML/XBRL, and this choice makes
the numeric layer more auditable than scraping every value from unstructured
PDF tables. The prepared artifact contains **417 normalized facts**, **284
derived calculations**, **43 anomaly records**, and **4,440 filing chunks** in
the local RAG index.

## Analytical Depth

The dashboard goes beyond fact retrieval. It calculates financial metrics from
source facts, compares companies over time, and keeps the formula/input trail
visible. For example, "which company has grown more?" is answered from computed
**Revenue YoY Growth**, not a loose text match. The answer explains the leader,
peer comparison, magnitude of outperformance, formula, and fiscal-period caveat.

The RAG Cited Q&A tab works as a small chat interface with up to **10 questions**
of history and follow-up support. It uses local/open components:
`all-MiniLM-L6-v2` for embeddings when available, optional
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` for generated answers, and deterministic
cited extraction as a fallback. Every returned answer must cite retrieved filing
evidence or refuse.

## Evaluation And Trust

The labeled evaluation set includes two answerable filing questions and two
refusal cases. The latest results are:

- **Answer correctness:** `1.0`
- **Citation accuracy:** `1.0`
- **Hallucination rate:** `0.0`
- **Case count:** `4`

I would trust this as a strong analyst demo and as a foundation for internal
review, but not yet as an unsupervised executive-facing system. The reason is
not that the architecture is ungrounded; it is that the labeled evaluation set
is still small. The first thing I would improve is evaluation breadth: add more
numeric tolerance checks, known restatement examples, multi-hop comparison
questions, segment-vs-consolidated traps, and harder unanswerable questions.

## Most Interesting Insight

The clearest surfaced insight is NVIDIA's recent growth advantage over AMD and
Intel. In the prepared analytics, NVIDIA leads the latest available Revenue YoY
Growth comparison, while AMD also shows strong growth and Intel is much lower
on the same metric. I am confident in the arithmetic because each result stores
the formula `(current - prior_year) / abs(prior_year)` and the SEC fact inputs.
Before presenting this to an executive, I would manually spot-check the headline
numbers against the original SEC filing accessions.

## Failure Found And Diagnosed

The most important failure I found was in the initial filing chunker. It could
enter an infinite loop because the final chunk reset `start = end - overlap`
even after reaching the end of the document. That would have made ingestion hang
on real filings. I fixed the termination condition and added tests.

I also found a retrieval risk: filtered retrieval could silently broaden a
company-specific query and return another company's evidence. For a comparative
financial system, that is worse than refusing. I changed retrieval to preserve
strict ticker/year filters. A third correction was evaluation-related: an early
generated evaluation set referenced unrelated AAPL/MSFT questions, so I replaced
it with NVDA/AMD/INTC cases from this corpus.

## Messy Financial Data

The system surfaces messy data rather than hiding it. The Data quality tab shows
duplicates, conflicts, and restatement/re-reporting signals. Each anomaly keeps
the metric, period, accession list, values, and concepts so the reviewer can
decide whether it is a true conflict, amendment/restatement, unit issue, or
period-context mismatch.

For unanswerable questions, prompt-injection attempts, and future exact
forecast questions, the system refuses rather than inventing numbers. That is a
deliberate design choice because the assignment rewards grounded evidence and
honest abstention more than fluent unsupported answers.

## Implementation And Framework Tradeoffs

Several implementation choices were corrected during review: unrelated
evaluation questions were replaced with NVDA/AMD/INTC cases, strict retrieval
filters were enforced, broad growth questions were routed to computed analytics
instead of raw RAG, and refusal behavior was added for future forecasts and
prompt-injection attempts.

For the RAG orchestration layer I chose **Chroma + sentence-transformers +
Hugging Face Transformers** as the "similar" framework the assignment permits,
rather than LangChain or LlamaIndex. The reason is deliberate: LangChain and
LlamaIndex abstract over retrieval and generation in ways that make it harder to
enforce strict financial-accuracy constraints. I needed exact control over (a)
which ticker's evidence a query can draw from, (b) whether every claim in a
generated answer is numerically supported by the retrieved passages, and (c)
how the system refuses when evidence is absent. Implementing those three
behaviors inside a high-level chain would have required overriding their defaults
extensively. Using Chroma's native filter API and writing the verification logic
directly in `src/rag/retriever.py` and `src/rag/qa.py` made those guarantees
transparent and testable. Streamlit, Chroma, and Hugging Face are real,
production-grade tools — not reinvented plumbing.

## Quant-to-Narrative Linkage

When a metric moves materially, the system can connect it to management
commentary through the RAG Cited Q&A tab. For example, asking *"What did Intel
say about revenue compared with 2024?"* retrieves the exact MD&A passage
("Intel Products revenue was roughly flat with 2024, primarily due to lower CCG
revenue that was substantially offset by higher DCAI revenue") and cites the
accession and URL. The filing chunks are segmented by section (`mda`,
`results_of_operations`, `liquidity`, `risk_factors`) so a quantitative question
can be followed up with a narrative question and the retriever will surface the
relevant section. A dedicated "metric moved → management said X" automatic
linkage view was deprioritized in favour of ensuring the RAG layer refuses
correctly and cites accurately for every answer it does give.

## How I Used AI Tools

I used an AI coding assistant throughout this exercise for scaffolding and
boilerplate — generating initial file structures, drafting docstrings, and
suggesting regex patterns. Specific places where I overrode or corrected the AI:

1. **Chunker termination bug.** The AI-generated `chunk_text` function reset
   `start = end - overlap` unconditionally, including after `end` had already
   reached `text_len`. That would have caused an infinite loop on any filing
   long enough to reach the final chunk. I diagnosed the hang in testing,
   added the `if end >= text_len: break` guard, and added a regression test.

2. **Retrieval filter broadening.** The AI initially generated retrieval logic
   that fell back to an un-filtered query when a ticker-filtered query returned
   too few results. For a comparative financial system that is worse than
   refusing, because it silently substitutes another company's evidence. I
   removed the fallback and kept strict filters, accepting that some queries
   return fewer results rather than wrong-company results.

3. **Evaluation questions.** An early AI-generated labeled evaluation set
   included AAPL and MSFT questions unrelated to this corpus. I replaced every
   case with NVDA/AMD/INTC questions drawn from the actual filed documents and
   added the two refusal cases (future forecast, prompt injection) because the
   AI omitted adversarial coverage entirely.
