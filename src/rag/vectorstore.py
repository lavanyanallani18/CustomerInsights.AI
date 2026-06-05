"""
Vector store management using ChromaDB with sentence-transformers embeddings.
"""
import logging
import hashlib
import math
import re
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings
from chromadb import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "sec_filings"


class LocalEmbeddingFunction(EmbeddingFunction):
    """Offline-safe embedding function for ChromaDB.

    It prefers a locally cached sentence-transformer model, then falls back to a
    deterministic hashed bag-of-words vector so indexing never triggers a
    surprise network download.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = None
        self.dimensions = 384
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, local_files_only=True)
            logger.info("Loaded cached embedding model: %s", model_name)
        except Exception as exc:
            logger.warning(
                "Using deterministic hash embeddings because %s is not available locally: %s",
                model_name,
                exc,
            )

    def __call__(self, input: Documents) -> Embeddings:
        values = list(input)
        if self.model is not None:
            embeddings = self.model.encode(values, show_progress_bar=False)
            return embeddings.tolist()
        return [self._hash_embed(text) for text in values]

    def _hash_embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-zA-Z0-9&'-]+", text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class VectorStore:
    """ChromaDB-backed vector store for SEC filings."""

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_fn = LocalEmbeddingFunction(embedding_model)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"VectorStore ready. Documents: {self.collection.count()}")

    def add_documents(self, documents: list[dict], batch_size: int = 100) -> int:
        """Add parsed filing documents to the collection."""
        if not documents:
            return 0

        # Deduplicate by ID
        existing_ids = set(self.collection.get(include=[])["ids"])
        new_docs = [d for d in documents if d["id"] not in existing_ids]

        if not new_docs:
            logger.info("All documents already indexed.")
            return 0

        added = 0
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i : i + batch_size]
            self.collection.add(
                ids=[d["id"] for d in batch],
                documents=[d["content"] for d in batch],
                metadatas=[d["metadata"] for d in batch],
            )
            added += len(batch)
            logger.info(f"  Indexed batch {i//batch_size + 1}: {added}/{len(new_docs)} docs")

        logger.info(f"Total indexed: {self.collection.count()} documents")
        return added

    def query(
        self,
        query_text: str,
        n_results: int = 8,
        ticker_filter: Optional[str] = None,
        filing_type_filter: Optional[str] = None,
        year_filter: Optional[str] = None,
        section_filter: Optional[str] = None,
    ) -> list[dict]:
        """Query the vector store with optional metadata filters."""
        filters = []
        if ticker_filter:
            filters.append({"ticker": ticker_filter})
        if filing_type_filter:
            filters.append({"filing_type": filing_type_filter})
        if year_filter:
            filters.append({"year": year_filter})
        if section_filter:
            filters.append({"section": section_filter})

        kwargs: dict = {
            "query_texts": [query_text],
            "n_results": min(n_results, self.collection.count() or 1),
            "include": ["documents", "metadatas", "distances"],
        }
        if len(filters) == 1:
            kwargs["where"] = filters[0]
        elif filters:
            kwargs["where"] = {"$and": filters}

        results = self.collection.query(**kwargs)

        docs = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            dist = results["distances"][0][i]
            docs.append({
                "content": doc,
                "metadata": meta,
                "relevance_score": round(1 - dist, 4),
            })
        return docs

    def get_stats(self) -> dict:
        """Return collection statistics."""
        total = self.collection.count()
        if total == 0:
            return {"total": 0, "by_ticker": {}, "by_type": {}}

        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        by_ticker: dict[str, int] = {}
        by_type: dict[str, int] = {}
        by_year: dict[str, int] = {}

        for m in all_meta:
            by_ticker[m.get("ticker", "?")] = by_ticker.get(m.get("ticker", "?"), 0) + 1
            by_type[m.get("filing_type", "?")] = by_type.get(m.get("filing_type", "?"), 0) + 1
            by_year[m.get("year", "?")] = by_year.get(m.get("year", "?"), 0) + 1

        return {
            "total": total,
            "by_ticker": by_ticker,
            "by_type": by_type,
            "by_year": by_year,
        }

    def clear(self):
        """Clear all documents from the collection."""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection cleared.")
