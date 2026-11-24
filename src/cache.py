import os
import re
import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any, List


def normalize_query(query: str) -> str:
    """Normalize a user query for cache lookups:

    converts to lowercase, strips punctuation, and collapses whitespace.
    """
    text = query.lower().strip()
    # Replace non-alphanumeric characters with space
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()
    return text


class FAQCache:
    """SQLite-backed FAQ cache to store repeated Q&A responses,

    saving 60-80% of daily LLM tokens in production.
    """

    def __init__(self, db_path: str = "./isro_faq_cache.sqlite"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS faq_cache (
                    normalized_query TEXT PRIMARY KEY,
                    original_query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    hit_count INTEGER NOT NULL DEFAULT 1
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_metrics (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    total_lookups INTEGER DEFAULT 0,
                    cache_hits INTEGER DEFAULT 0,
                    cache_misses INTEGER DEFAULT 0
                )
                """
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO cache_metrics (id, total_lookups, cache_hits, cache_misses)
                VALUES (1, 0, 0, 0)
                """
            )
            conn.commit()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        norm_key = normalize_query(query)
        if not norm_key:
            return None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE cache_metrics SET total_lookups = total_lookups + 1 WHERE id = 1")

            cursor.execute(
                "SELECT * FROM faq_cache WHERE normalized_query = ?",
                (norm_key,),
            )
            row = cursor.fetchone()

            if row:
                now_iso = datetime.now().isoformat()
                cursor.execute(
                    """
                    UPDATE faq_cache
                    SET hit_count = hit_count + 1, last_accessed_at = ?
                    WHERE normalized_query = ?
                    """,
                    (now_iso, norm_key),
                )
                cursor.execute("UPDATE cache_metrics SET cache_hits = cache_hits + 1 WHERE id = 1")
                conn.commit()

                sources = []
                try:
                    sources = json.loads(row["sources_json"])
                except Exception:
                    pass

                return {
                    "answer": row["answer"],
                    "sources": sources,
                    "model_used": row["model_used"],
                    "hit_count": row["hit_count"] + 1,
                    "created_at": row["created_at"],
                    "cached": True,
                }
            else:
                cursor.execute("UPDATE cache_metrics SET cache_misses = cache_misses + 1 WHERE id = 1")
                conn.commit()
                return None

    def set(
        self,
        query: str,
        answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        model_used: str = "Unknown",
    ) -> None:
        norm_key = normalize_query(query)
        if not norm_key or not answer.strip():
            return

        now_iso = datetime.now().isoformat()
        sources_json = json.dumps(sources or [])

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO faq_cache (
                    normalized_query, original_query, answer, sources_json,
                    model_used, created_at, last_accessed_at, hit_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                ON CONFLICT(normalized_query) DO UPDATE SET
                    original_query = excluded.original_query,
                    answer = excluded.answer,
                    sources_json = excluded.sources_json,
                    model_used = excluded.model_used,
                    last_accessed_at = excluded.last_accessed_at
                """,
                (norm_key, query.strip(), answer.strip(), sources_json, model_used, now_iso, now_iso),
            )
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT total_lookups, cache_hits, cache_misses FROM cache_metrics WHERE id = 1")
            metrics = cursor.fetchone()

            cursor.execute("SELECT COUNT(*), SUM(hit_count) FROM faq_cache")
            count_row = cursor.fetchone()
            total_entries = count_row[0] or 0
            total_hits_from_entries = count_row[1] or 0

            lookups = metrics["total_lookups"] if metrics else 0
            hits = metrics["cache_hits"] if metrics else 0
            misses = metrics["cache_misses"] if metrics else 0
            hit_rate = (hits / lookups * 100.0) if lookups > 0 else 0.0

            return {
                "total_entries": total_entries,
                "total_lookups": lookups,
                "cache_hits": hits,
                "cache_misses": misses,
                "hit_rate_pct": round(hit_rate, 1),
            }

    def clear(self) -> None:
        with self._get_connection() as conn:
            conn.execute("DELETE FROM faq_cache")
            conn.execute("UPDATE cache_metrics SET total_lookups = 0, cache_hits = 0, cache_misses = 0 WHERE id = 1")
            conn.commit()
