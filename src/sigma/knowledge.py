"""Knowledge Base — document ingestion, chunking, and retrieval.

Provides CrewAI-equivalent knowledge source support:
  - StringSource, TextFileSource, PDFSource, CSVSource, JSONSource
  - Chunking with configurable size and overlap
  - TF-IDF based retrieval (zero external deps beyond stdlib)
  - Pluggable embedding function for semantic search

Usage:
    kb = KnowledgeBase()
    kb.add_string("Rocket propulsion uses KNSB propellant...", source="manual")
    kb.add_file("docs/design.md")
    chunks = kb.query("What propellant should we use?", top_k=3)
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from sigma.log import get_logger

_log = get_logger("sigma.knowledge")


@dataclass
class Chunk:
    """A document chunk with metadata."""
    text: str
    source: str = ""
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A retrieved chunk with relevance score."""
    chunk: Chunk
    score: float


class KnowledgeBase:
    """Document store with chunking and retrieval.

    Supports multiple source types (string, text file, CSV, JSON, PDF placeholder).
    Retrieval uses TF-IDF by default, with optional embedding-based semantic search.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        embed_fn: Callable[[str], list[float]] | None = None,
    ):
        """
        Args:
            chunk_size: Max characters per chunk.
            chunk_overlap: Overlap characters between consecutive chunks.
            embed_fn: Optional embedding function(text) -> list[float].
                     When set, semantic search replaces TF-IDF.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embed_fn = embed_fn
        self._chunks: list[Chunk] = []
        self._idf_cache: dict[str, float] | None = None
        self._dirty = False  # Recompute IDF when chunks change

    # ── Ingestion ───────────────────────────────────────────────────

    def add_string(self, text: str, source: str = "inline",
                   metadata: dict | None = None) -> list[Chunk]:
        """Add a plain string as a knowledge source."""
        chunks = self._chunk_text(text, source, metadata or {})
        self._chunks.extend(chunks)
        self._dirty = True
        return chunks

    def add_file(self, path: str | Path, source: str = "",
                 metadata: dict | None = None) -> list[Chunk]:
        """Add a file, auto-detecting type by extension."""
        path = Path(path)
        if not path.exists():
            _log.warning("Knowledge source not found: %s", path)
            return []
        src = source or path.name
        ext = path.suffix.lower()
        if ext == ".pdf":
            return self.add_pdf(path, source=src, metadata=metadata)
        elif ext == ".csv":
            return self.add_csv(path, source=src, metadata=metadata)
        elif ext == ".json":
            return self.add_json(path, source=src, metadata=metadata)
        elif ext in (".md", ".txt", ".py", ".rst", ""):
            text = path.read_text(encoding="utf-8")
            return self.add_string(text, source=src, metadata=metadata)
        else:
            # Try as text
            try:
                text = path.read_text(encoding="utf-8")
                return self.add_string(text, source=src, metadata=metadata)
            except Exception:
                _log.warning("Cannot read file as text: %s", path)
                return []

    def add_pdf(self, path: str | Path, source: str = "",
                metadata: dict | None = None) -> list[Chunk]:
        """Add a PDF file (requires PyPDF2 or pdfplumber).

        Falls back to a placeholder note if libraries are unavailable.
        """
        path = Path(path)
        src = source or path.name
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return self.add_string(text, source=src, metadata=metadata)
        except ImportError:
            _log.info("PyPDF2 not installed; storing PDF reference: %s", path)
            return self.add_string(
                f"[PDF 文件: {path.name} — 安装 PyPDF2 以提取文本]",
                source=src, metadata=metadata,
            )
        except Exception as e:
            _log.warning("Failed to read PDF %s: %s", path, e)
            return self.add_string(
                f"[PDF 文件: {path.name} — 读取失败: {e}]",
                source=src, metadata=metadata,
            )

    def add_csv(self, path: str | Path, source: str = "",
                metadata: dict | None = None) -> list[Chunk]:
        """Add a CSV file, converting each row to text."""
        path = Path(path)
        src = source or path.name
        chunks = []
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                text = ", ".join(f"{k}: {v}" for k, v in row.items())
                chunks.extend(
                    self._chunk_text(text, source=src,
                                     metadata={"row": i, **(metadata or {})})
                )
        self._chunks.extend(chunks)
        self._dirty = True
        return chunks

    def add_json(self, path: str | Path, source: str = "",
                 metadata: dict | None = None) -> list[Chunk]:
        """Add a JSON file, converting to structured text."""
        path = Path(path)
        src = source or path.name
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        # Flatten JSON to readable text
        flat = json.dumps(data, indent=2, ensure_ascii=False)
        return self.add_string(flat, source=src, metadata=metadata)

    def add_directory(self, dir_path: str | Path, pattern: str = "*.*",
                      metadata: dict | None = None) -> int:
        """Add all files matching pattern in a directory. Returns count added."""
        dir_path = Path(dir_path)
        count = 0
        for f in sorted(dir_path.rglob(pattern)):
            if f.is_file() and not f.name.startswith("."):
                chunks = self.add_file(f, metadata=metadata)
                count += len(chunks)
        return count

    # ── Retrieval ───────────────────────────────────────────────────

    def query(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve top-k most relevant chunks for a query.

        Uses TF-IDF keyword matching by default, or semantic search
        if an embedding function was provided at init.
        """
        if not self._chunks:
            return []
        if self._embed_fn:
            return self._semantic_query(query, top_k)
        return self._tfidf_query(query, top_k)

    def query_string(self, query: str, top_k: int = 5) -> str:
        """Query and return concatenated chunk text (for LLM injection)."""
        results = self.query(query, top_k)
        if not results:
            return ""
        lines = []
        for i, r in enumerate(results):
            lines.append(f"[来源: {r.chunk.source}] {r.chunk.text}")
        return "\n\n---\n\n".join(lines)

    # ── TF-IDF Retrieval ────────────────────────────────────────────

    def _tfidf_query(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Simple TF-IDF based retrieval."""
        if self._dirty or self._idf_cache is None:
            self._compute_idf()

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for chunk in self._chunks:
            score = self._tfidf_score(query_tokens, chunk.text)
            if score > 0:
                scored.append(RetrievalResult(chunk=chunk, score=score))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def _compute_idf(self) -> None:
        """Compute IDF values across all chunks."""
        n = len(self._chunks)
        df: Counter[str, int] = Counter()
        for chunk in self._chunks:
            unique_tokens = set(self._tokenize(chunk.text))
            for token in unique_tokens:
                df[token] += 1
        self._idf_cache = {
            token: math.log((n + 1) / (count + 1)) + 1.0
            for token, count in df.items()
        }
        self._dirty = False

    def _tfidf_score(self, query_tokens: list[str], doc_text: str) -> float:
        """Compute TF-IDF cosine similarity between query and document."""
        doc_tokens = self._tokenize(doc_text)
        if not doc_tokens:
            return 0.0
        doc_tf = Counter(doc_tokens)
        doc_len = len(doc_tokens)

        query_tf = Counter(query_tokens)
        query_vec: dict[str, float] = {}
        doc_vec: dict[str, float] = {}

        for token, tf in query_tf.items():
            idf = self._idf_cache.get(token, 1.0) if self._idf_cache else 1.0
            query_vec[token] = (tf / len(query_tokens)) * idf

        for token, tf in doc_tf.items():
            idf = self._idf_cache.get(token, 1.0) if self._idf_cache else 1.0
            doc_vec[token] = (tf / doc_len) * idf

        # Cosine similarity
        dot = sum(query_vec.get(t, 0) * doc_vec.get(t, 0) for t in set(query_vec) | set(doc_vec))
        q_norm = math.sqrt(sum(v ** 2 for v in query_vec.values()))
        d_norm = math.sqrt(sum(v ** 2 for v in doc_vec.values()))
        denominator = q_norm * d_norm
        return dot / denominator if denominator > 0 else 0.0

    def _semantic_query(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Semantic search using embedding function (cosine similarity)."""
        query_vec = self._embed_fn(query)
        if not query_vec:
            return []
        scored = []
        for chunk in self._chunks:
            chunk_vec = self._embed_fn(chunk.text)
            if not chunk_vec:
                continue
            dot = sum(a * b for a, b in zip(query_vec, chunk_vec))
            q_norm = math.sqrt(sum(v ** 2 for v in query_vec))
            c_norm = math.sqrt(sum(v ** 2 for v in chunk_vec))
            score = dot / (q_norm * c_norm) if q_norm * c_norm > 0 else 0.0
            scored.append(RetrievalResult(chunk=chunk, score=score))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    # ── Chunking ────────────────────────────────────────────────────

    def _chunk_text(self, text: str, source: str, metadata: dict) -> list[Chunk]:
        """Split text into overlapping chunks by paragraph/sentence boundaries."""
        chunks = []
        if len(text) <= self.chunk_size:
            chunks.append(Chunk(text=text, source=source, chunk_index=0,
                                metadata=dict(metadata)))
            return chunks

        # Split on paragraph boundaries first
        paragraphs = text.split("\n\n")
        current = ""
        index = 0
        for para in paragraphs:
            if len(current) + len(para) <= self.chunk_size:
                current += ("\n\n" if current else "") + para
            else:
                if current:
                    chunks.append(Chunk(text=current.strip(), source=source,
                                        chunk_index=index, metadata=dict(metadata)))
                    index += 1
                    # Overlap: keep last overlap chars
                    if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                        current = current[-self.chunk_overlap:]
                    else:
                        current = ""
                # If single paragraph exceeds chunk_size, split on sentences
                if len(para) > self.chunk_size:
                    sentences = re.split(r'(?<=[。.!?！？\n])\s*', para)
                    for sent in sentences:
                        if len(current) + len(sent) <= self.chunk_size:
                            current += sent
                        else:
                            if current.strip():
                                chunks.append(Chunk(
                                    text=current.strip(), source=source,
                                    chunk_index=index, metadata=dict(metadata),
                                ))
                                index += 1
                                if self.chunk_overlap > 0 and len(current) > self.chunk_overlap:
                                    current = current[-self.chunk_overlap:]
                                else:
                                    current = ""
                            current += sent
                else:
                    current += ("\n\n" if current else "") + para
        if current.strip():
            chunks.append(Chunk(text=current.strip(), source=source,
                                chunk_index=index, metadata=dict(metadata)))
        return chunks

    # ── Helpers ─────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenizer for Chinese + English text."""
        # Split on whitespace and punctuation
        tokens = re.findall(r'[一-鿿]+|[a-zA-Z0-9]+', text.lower())
        # Filter short tokens
        return [t for t in tokens if len(t) >= 2]

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._idf_cache = None
        self._dirty = False
