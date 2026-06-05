"""Run a labeled, offline-friendly evaluation of grounded Q&A."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable, Optional

_NUMBER_RE = re.compile(r"(?<!\w)\$?\(?-?\d[\d,]*(?:\.\d+)?%?\)?")


def load_cases(path: Optional[str] = None) -> list[dict[str, Any]]:
    """Load labeled evaluation cases."""
    dataset = Path(path) if path else Path(__file__).with_name("labeled_set.json")
    value = json.loads(dataset.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("Evaluation dataset must contain a JSON list.")
    return value


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("$", "").replace(",", "")).strip()


def _numbers(text: str) -> set[str]:
    text = re.sub(r"\[\d+\]", "", text)
    return {_normalize(match.group(0)) for match in _NUMBER_RE.finditer(text)}


def _contains_terms(text: str, terms: list[str]) -> bool:
    normalized = _normalize(text)
    return all(_normalize(term) in normalized for term in terms)


def _citation_accuracy(
    answer: str,
    citations: list[dict[str, Any]],
    expected_answerable: bool,
    required_terms: list[str],
) -> float:
    if not expected_answerable:
        return float(not citations)
    if not citations:
        return 0.0

    referenced = {int(value) for value in re.findall(r"\[(\d+)\]", answer)}
    citation_ids = {citation.get("id") for citation in citations}
    quotes = " ".join(str(citation.get("quote", "")) for citation in citations)
    valid_references = referenced == citation_ids and all(
        isinstance(value, int) and value > 0 for value in citation_ids
    )
    supported_numbers = _numbers(answer).issubset(_numbers(quotes))
    required_evidence = _contains_terms(quotes, required_terms)
    return float(valid_references and supported_numbers and required_evidence)


def _is_hallucination(
    answer: str,
    answerable: bool,
    citations: list[dict[str, Any]],
    expected_answerable: bool,
) -> bool:
    if answerable and not expected_answerable:
        return True
    if not answerable:
        return False
    quotes = " ".join(str(citation.get("quote", "")) for citation in citations)
    return not citations or not _numbers(answer).issubset(_numbers(quotes))


def evaluate_cases(
    qa_fn: Callable[..., dict[str, Any]],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate a QA callable and return aggregate metrics plus case details."""
    details = []
    for case in cases:
        kwargs = {
            key: case[key]
            for key in ("ticker_filter", "year_filter")
            if case.get(key) is not None
        }
        result = qa_fn(case["question"], **kwargs)
        answer = str(result.get("answer", ""))
        citations = result.get("citations", [])
        expected_answerable = bool(case.get("expected_answerable"))
        answerable = bool(result.get("answerable"))

        if expected_answerable:
            correct = answerable and _contains_terms(
                answer, case.get("required_answer_terms", [])
            )
        else:
            correct = not answerable
        citation_accuracy = _citation_accuracy(
            answer,
            citations,
            expected_answerable,
            case.get("required_citation_terms", []),
        )
        hallucinated = _is_hallucination(
            answer, answerable, citations, expected_answerable
        )
        details.append(
            {
                "id": case.get("id", case["question"]),
                "answer_correct": bool(correct),
                "citation_accurate": bool(citation_accuracy),
                "hallucinated": hallucinated,
                "result": result,
            }
        )

    count = len(details)
    divisor = count or 1
    return {
        "case_count": count,
        "answer_correctness": sum(item["answer_correct"] for item in details) / divisor,
        "citation_accuracy": sum(item["citation_accurate"] for item in details) / divisor,
        "hallucination_rate": sum(item["hallucinated"] for item in details) / divisor,
        "cases": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Path to a labeled JSON dataset.")
    parser.add_argument("--persist-dir", default="./chroma_db")
    parser.add_argument("--output", help="Optional path for the JSON report.")
    args = parser.parse_args()

    from src.rag.qa import answer_question
    from src.rag.vectorstore import VectorStore

    vectorstore = VectorStore(persist_dir=args.persist_dir)
    report = evaluate_cases(
        lambda question, **kwargs: answer_question(question, vectorstore, **kwargs),
        load_cases(args.dataset),
    )
    rendered = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
