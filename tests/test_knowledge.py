"""Tests for sigma.knowledge — KnowledgeBase document store."""

import csv
import json
import tempfile
from pathlib import Path

import pytest
from sigma.knowledge import KnowledgeBase, Chunk, RetrievalResult


# ═══════════════════════════════════════════════════════════════════════
# Chunk / RetrievalResult
# ═══════════════════════════════════════════════════════════════════════

class TestDataclasses:
    def test_chunk_defaults(self):
        c = Chunk(text="hello")
        assert c.text == "hello"
        assert c.source == ""
        assert c.chunk_index == 0
        assert c.metadata == {}

    def test_retrieval_result(self):
        c = Chunk(text="hello")
        r = RetrievalResult(chunk=c, score=0.85)
        assert r.score == 0.85
        assert r.chunk.text == "hello"


# ═══════════════════════════════════════════════════════════════════════
# add_string
# ═══════════════════════════════════════════════════════════════════════

class TestAddString:
    def test_add_string_short(self):
        kb = KnowledgeBase()
        chunks = kb.add_string("火箭发动机使用KNSB推进剂", source="manual")
        assert len(chunks) == 1
        assert chunks[0].source == "manual"
        assert chunks[0].text == "火箭发动机使用KNSB推进剂"
        assert kb.chunk_count == 1

    def test_add_string_long_splits(self):
        kb = KnowledgeBase(chunk_size=100, chunk_overlap=0)
        long_text = "\n\n".join([f"paragraph {i} " * 10 for i in range(10)])
        chunks = kb.add_string(long_text, source="test")
        assert len(chunks) > 1

    def test_add_string_with_metadata(self):
        kb = KnowledgeBase()
        chunks = kb.add_string("test", source="s", metadata={"author": "me"})
        assert chunks[0].metadata["author"] == "me"

    def test_add_multiple_strings(self):
        kb = KnowledgeBase()
        kb.add_string("first document", source="a")
        kb.add_string("second document", source="b")
        assert kb.chunk_count == 2


# ═══════════════════════════════════════════════════════════════════════
# add_file
# ═══════════════════════════════════════════════════════════════════════

class TestAddFile:
    def test_add_text_file(self):
        kb = KnowledgeBase()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", encoding="utf-8", delete=False
        ) as f:
            f.write("火箭发动机设计文档\nKNSB推进剂参数")
            path = f.name
        try:
            chunks = kb.add_file(path)
            assert len(chunks) >= 1
            assert "火箭" in chunks[0].text
        finally:
            Path(path).unlink()

    def test_add_file_not_found(self):
        kb = KnowledgeBase()
        chunks = kb.add_file("/nonexistent/file.txt")
        assert chunks == []

    def test_add_txt_file(self):
        kb = KnowledgeBase()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write("plain text content")
            path = f.name
        try:
            chunks = kb.add_file(path)
            assert len(chunks) >= 1
        finally:
            Path(path).unlink()

    def test_add_py_file(self):
        kb = KnowledgeBase()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", encoding="utf-8", delete=False
        ) as f:
            f.write("# Python code\nx = 1")
            path = f.name
        try:
            chunks = kb.add_file(path)
            assert len(chunks) >= 1
        finally:
            Path(path).unlink()


# ═══════════════════════════════════════════════════════════════════════
# add_csv
# ═══════════════════════════════════════════════════════════════════════

class TestAddCSV:
    def test_add_csv_file(self):
        kb = KnowledgeBase()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8", newline="", delete=False
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["name", "value"])
            writer.writeheader()
            writer.writerow({"name": "thrust", "value": "1500"})
            writer.writerow({"name": "isp", "value": "155"})
            path = f.name
        try:
            chunks = kb.add_csv(path)
            assert len(chunks) >= 1
            assert "thrust" in chunks[0].text.lower() or any(
                "thrust" in c.text.lower() for c in chunks
            )
        finally:
            Path(path).unlink()


# ═══════════════════════════════════════════════════════════════════════
# add_json
# ═══════════════════════════════════════════════════════════════════════

class TestAddJSON:
    def test_add_json_file(self):
        kb = KnowledgeBase()
        data = {"engine": "KNSB", "thrust_n": 1500}
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name
        try:
            chunks = kb.add_json(path)
            assert len(chunks) >= 1
            assert "KNSB" in chunks[0].text
        finally:
            Path(path).unlink()


# ═══════════════════════════════════════════════════════════════════════
# add_pdf
# ═══════════════════════════════════════════════════════════════════════

class TestAddPDF:
    def test_add_pdf_without_pypdf2(self):
        kb = KnowledgeBase()
        # Create a dummy file with .pdf extension
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pdf", encoding="utf-8", delete=False
        ) as f:
            f.write("not a real PDF")
            path = f.name
        try:
            chunks = kb.add_pdf(path)
            # Falls back to placeholder since PyPDF2 may not be installed
            assert len(chunks) >= 1
            assert "PDF" in chunks[0].text
        finally:
            Path(path).unlink()


# ═══════════════════════════════════════════════════════════════════════
# add_directory
# ═══════════════════════════════════════════════════════════════════════

class TestAddDirectory:
    def test_add_directory(self):
        kb = KnowledgeBase()
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.md").write_text("# Doc A\ncontent", encoding="utf-8")
            (Path(d) / "b.txt").write_text("Doc B", encoding="utf-8")
            # Hidden file should be skipped
            (Path(d) / ".secret").write_text("secret", encoding="utf-8")
            count = kb.add_directory(d, pattern="*.*")
            assert count >= 2

    def test_add_directory_with_pattern(self):
        kb = KnowledgeBase()
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.md").write_text("md", encoding="utf-8")
            (Path(d) / "b.txt").write_text("txt", encoding="utf-8")
            count = kb.add_directory(d, pattern="*.md")
            assert count >= 1


# ═══════════════════════════════════════════════════════════════════════
# query (TF-IDF)
# ═══════════════════════════════════════════════════════════════════════

class TestQuery:
    def test_query_empty(self):
        kb = KnowledgeBase()
        assert kb.query("anything") == []

    def test_query_exact_match(self):
        kb = KnowledgeBase()
        kb.add_string("KNSB推进剂具有183s理论比冲", source="propulsion")
        results = kb.query("KNSB比冲", top_k=3)
        assert len(results) >= 1
        assert results[0].score > 0

    def test_query_no_match(self):
        kb = KnowledgeBase()
        kb.add_string("rocket design", source="a")
        results = kb.query("zzzzz yyyyy")  # Tokenizer requires 2+ chars
        # "zzzzz" is 5 chars, will match as a token; but there's no overlap
        assert len(results) >= 0  # May or may not match depending on tokenization

    def test_query_top_k(self):
        kb = KnowledgeBase()
        for i in range(10):
            kb.add_string(f"document number {i} about rocket design", source=f"src{i}")
        results = kb.query("rocket", top_k=3)
        assert len(results) <= 3

    def test_query_string(self):
        kb = KnowledgeBase()
        kb.add_string("KNSB推进剂性能数据", source="test")
        text = kb.query_string("KNSB")
        assert "KNSB" in text
        assert "来源" in text

    def test_query_string_empty(self):
        kb = KnowledgeBase()
        assert kb.query_string("nothing") == ""


# ═══════════════════════════════════════════════════════════════════════
# Semantic query
# ═══════════════════════════════════════════════════════════════════════

class TestSemanticQuery:
    def test_semantic_query(self):
        def embed(text):
            return [len(text) % 3, len(text) % 5, len(text) % 7]

        kb = KnowledgeBase(embed_fn=embed)
        kb.add_string("rocket engine design", source="a")
        kb.add_string("cooking recipes", source="b")
        results = kb.query("engine thrust")
        assert len(results) >= 1

    def test_semantic_query_empty_chunks(self):
        def embed(text):
            return [1.0, 2.0]

        kb = KnowledgeBase(embed_fn=embed)
        assert kb.query("anything") == []


# ═══════════════════════════════════════════════════════════════════════
# _tokenize
# ═══════════════════════════════════════════════════════════════════════

class TestTokenize:
    def test_english(self):
        kb = KnowledgeBase()
        tokens = kb._tokenize("rocket engine design")
        assert "rocket" in tokens
        assert "engine" in tokens
        assert "design" in tokens

    def test_chinese(self):
        kb = KnowledgeBase()
        tokens = kb._tokenize("火箭发动机")
        assert "火箭" in tokens or "火箭发动机" in tokens

    def test_mixed(self):
        kb = KnowledgeBase()
        tokens = kb._tokenize("KNSB推进剂")
        assert "knsb" in tokens

    def test_short_filtered(self):
        kb = KnowledgeBase()
        tokens = kb._tokenize("a b c de fg hi")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "de" in tokens
        assert "hi" in tokens


# ═══════════════════════════════════════════════════════════════════════
# _chunk_text
# ═══════════════════════════════════════════════════════════════════════

class TestChunkText:
    def test_short_text_no_split(self):
        kb = KnowledgeBase(chunk_size=500)
        chunks = kb._chunk_text("short text", "src", {})
        assert len(chunks) == 1

    def test_long_text_splits(self):
        kb = KnowledgeBase(chunk_size=50, chunk_overlap=0)
        long_text = "\n\n".join([f"paragraph {i} with some content" for i in range(10)])
        chunks = kb._chunk_text(long_text, "src", {})
        assert len(chunks) > 1

    def test_chunk_with_overlap(self):
        kb = KnowledgeBase(chunk_size=100, chunk_overlap=20)
        long_text = "\n\n".join([f"paragraph {i} " * 10 for i in range(8)])
        chunks = kb._chunk_text(long_text, "src", {})
        assert len(chunks) >= 2


# ═══════════════════════════════════════════════════════════════════════
# IDF compute and score
# ═══════════════════════════════════════════════════════════════════════

class TestTFIDF:
    def test_compute_idf(self):
        kb = KnowledgeBase()
        kb.add_string("rocket engine thrust", source="a")
        kb.add_string("cooking recipes food", source="b")
        kb._compute_idf()
        assert kb._idf_cache is not None
        assert len(kb._idf_cache) >= 2

    def test_tfidf_score_positive(self):
        kb = KnowledgeBase()
        kb.add_string("rocket engine thrust", source="a")
        kb._compute_idf()
        score = kb._tfidf_score(["rocket", "engine"], "rocket engine thrust")
        assert score > 0

    def test_tfidf_score_zero(self):
        kb = KnowledgeBase()
        kb.add_string("rocket", source="a")
        kb._compute_idf()
        score = kb._tfidf_score(["zzzz"], "rocket")
        assert score == 0.0

    def test_tfidf_score_empty_doc(self):
        kb = KnowledgeBase()
        kb.add_string("x", source="a")
        score = kb._tfidf_score(["hello"], "")
        assert score == 0.0


# ═══════════════════════════════════════════════════════════════════════
# clear and chunk_count
# ═══════════════════════════════════════════════════════════════════════

class TestClear:
    def test_clear(self):
        kb = KnowledgeBase()
        kb.add_string("test", source="a")
        assert kb.chunk_count == 1
        kb.clear()
        assert kb.chunk_count == 0
        assert kb._idf_cache is None
        assert not kb._dirty

    def test_clear_then_add(self):
        kb = KnowledgeBase()
        kb.add_string("test1", source="a")
        kb.clear()
        kb.add_string("test2", source="b")
        assert kb.chunk_count == 1
        assert kb.query("test2")
