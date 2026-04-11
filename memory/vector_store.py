import json
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config import settings
from models import Paper, SearchRecord

logger = logging.getLogger(__name__)


class ResearchVectorStore:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._search_col = self._client.get_or_create_collection(
            name="search_history",
            metadata={"hnsw:space": "cosine"},   # Cosine distance for semantic similarity
        )
        self._papers_col = self._client.get_or_create_collection(
            name="papers_seen",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Loading embedding model '%s'…", settings.embedding_model)
        self._embedder = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded.")

    def _embed(self, text: str) -> list[float]:
        """Embed a single text string. Returns a flat list (ChromaDB format)."""
        vec = self._embedder.encode(text, normalize_embeddings=True)
        return vec.tolist()

    @staticmethod
    def _days_since(iso_timestamp: str) -> float:
        """Calculate days elapsed since an ISO 8601 timestamp string."""
        try:
            past = datetime.fromisoformat(iso_timestamp)
            if past.tzinfo is None:
                past = past.replace(tzinfo=timezone.utc)
            now = datetime.now(tz=timezone.utc)
            return max(0.0, (now - past).total_seconds() / 86400)
        except ValueError:
            return 0.0

    @staticmethod
    def temporal_weight(days_ago: float) -> float:
        return math.exp(-settings.temporal_decay_lambda * days_ago)

    def save_search(self, record: SearchRecord) -> None:
        doc_id = str(uuid.uuid4())
        embedding = self._embed(record.query)
        metadata = record.to_chroma_metadata()

        self._search_col.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[record.query],
            metadatas=[metadata],
        )
        logger.debug("Saved search record id=%s query='%s'", doc_id, record.query)

    def save_papers(self, papers: list[Paper], source_query: str) -> None:
        if not papers:
            return

        ids, embeddings, documents, metadatas = [], [], [], []
        for paper in papers:
            combined_text = f"{paper.title}. {paper.abstract}"
            ids.append(paper.paper_id)
            embeddings.append(self._embed(combined_text))
            documents.append(combined_text)
            metadatas.append({
                "title": paper.title,
                "authors": ", ".join(paper.authors[:3]),
                "url": paper.url,
                "source": paper.source,
                "published_date": str(paper.published_date) or "",
                "source_query": source_query,
                "saved_at": str(datetime.utcnow().isoformat()),
            })

        self._papers_col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug("Saved %d papers to vector store.", len(papers))

    def get_similar_searches(
        self,
        query: str,
        top_k: int = None,
    ) -> list[dict]:
        k = top_k or settings.chroma_top_k
        total = self._search_col.count()
        if total == 0:
            return []

        query_embedding = self._embed(query)
        results = self._search_col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, total),
            include=["metadatas", "distances", "documents"],
        )

        enriched = []
        for meta, dist in zip(
            results["metadatas"][0],
            results["distances"][0],
        ):
            raw_sim = 1.0 - dist                      # cosine similarity
            days_ago = self._days_since(meta.get("timestamp", ""))
            tw = self.temporal_weight(days_ago)
            enriched.append({
                **meta,
                "paper_ids": meta.get("paper_ids", "").split(","),
                "expanded_topics": meta.get("expanded_topics", "").split(","),
                "raw_similarity": round(raw_sim, 4),
                "days_ago": round(days_ago, 1),
                "temporal_weight": round(tw, 4),
                "weighted_score": round(raw_sim * tw, 4),
            })

        enriched.sort(key=lambda x: x["weighted_score"], reverse=True)
        return enriched

    def get_related_papers(self, query: str, top_k: int = None) -> list[dict]:
        k = top_k or settings.chroma_top_k
        total = self._papers_col.count()
        if total == 0:
            return []

        query_embedding = self._embed(query)
        results = self._papers_col.query(
            query_embeddings=[query_embedding],
            n_results=min(k, total),
            include=["metadatas", "distances"],
        )

        related = []
        for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
            days_ago = self._days_since(meta.get("saved_at", ""))
            tw = self.temporal_weight(days_ago)
            sim = 1.0 - dist
            related.append({
                **meta,
                "similarity": round(sim, 4),
                "temporal_weight": round(tw, 4),
                "weighted_score": round(sim * tw, 4),
            })

        related.sort(key=lambda x: x["weighted_score"], reverse=True)
        return related

    def build_interest_map(self) -> dict:
        all_records = self._search_col.get(include=["metadatas", "documents"])
        if not all_records["ids"]:
            return {}

        topic_weights: dict[str, float] = {}
        for meta, doc in zip(all_records["metadatas"], all_records["documents"]):
            query = meta.get("query", doc)
            days_ago = self._days_since(meta.get("timestamp", ""))
            tw = self.temporal_weight(days_ago)
            topic_weights[query] = topic_weights.get(query, 0.0) + tw

        total_weight = sum(topic_weights.values())
        if total_weight > 0:
            topic_weights = {k: round(v / total_weight, 4) for k, v in topic_weights.items()}

        return dict(sorted(topic_weights.items(), key=lambda x: x[1], reverse=True))
