#!/usr/bin/env python3
"""Prepare the SEC filing/companyfacts analytics dataset."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytics.dataset import build_prepared_dataset, load_prepared_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare exactly 15 recent NVDA/AMD/INTC 10-K/10-Q filings and traceable analytics."
    )
    parser.add_argument("--data-dir", default="data/sec", help="Cache/output directory (default: data/sec)")
    parser.add_argument(
        "--user-agent",
        help="SEC-compliant identity with contact information; defaults to SEC_USER_AGENT",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use prepared JSON if present, otherwise rebuild strictly from raw caches",
    )
    parser.add_argument(
        "--skip-filings",
        action="store_true",
        help="Fetch filing metadata and companyfacts without downloading filing HTML",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip parsing filings and building the local Chroma RAG index",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild prepared JSON even when --offline and a validated prepared file exists",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.offline and not args.force:
        try:
            payload = load_prepared_dataset(args.data_dir)
        except ValueError:
            payload = build_prepared_dataset(
                args.data_dir,
                user_agent=args.user_agent,
                offline=True,
                download_filings=not args.skip_filings,
            )
    else:
        payload = build_prepared_dataset(
            args.data_dir,
            user_agent=args.user_agent,
            offline=args.offline,
            download_filings=not args.skip_filings,
        )
    if not args.skip_index:
        from src.ingestion.parser import parse_all_filings
        from src.rag.vectorstore import VectorStore

        documents = parse_all_filings(str(Path(args.data_dir) / "manifest.json"))
        store = VectorStore(persist_dir="chroma_db")
        store.clear()
        indexed = store.add_documents(documents, batch_size=100)
        payload["counts"]["rag_documents"] = len(documents)
        payload["counts"]["rag_indexed"] = indexed

    print(json.dumps(payload["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
