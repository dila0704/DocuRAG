"""pipeline.ingest_document()/search_documents() icin ilk test dosyasi
(DOC-30 B1 fazinda acildi -- ingest_document alan cikarimi kazandigi icin,
orkestrasyonun davranisini dogrudan test etmenin tam zamani).

Hicbir gercek OCR/LLM/embedding/FAISS cagrisi yapilmaz -- her bagimlilik
(ocr, classifier, field_extractor, embedder, vector_store, retrieval,
query_rewriter) monkeypatch ile sahtelestirilir."""
import classifier
import embedder
import field_extractor
import ocr
import pipeline
import query_rewriter
import retrieval
import vector_store


def _patch_ingest_chain(monkeypatch, raw_text="OCR ile cikan metin", num_chunks=2):
    monkeypatch.setattr(ocr, "extract_text_from_image", lambda image_path: raw_text)
    monkeypatch.setattr(
        classifier, "classify_chunks",
        lambda chunks: {"siniflar": ["fatura"], "guven": 0.9, "etiketler": ["ornek"], "human_review": False},
    )
    monkeypatch.setattr(
        classifier, "attach_labels_to_chunks",
        lambda chunks, classifications: [{**c, "siniflar": ["fatura"], "guven": 0.9} for c in chunks],
    )
    monkeypatch.setattr(
        field_extractor, "extract_fields",
        lambda text: {"tarih": "12.08.2026", "tutar": "500 TL", "taraflar": [], "belge_no": None, "konu": "Test"},
    )
    monkeypatch.setattr(
        field_extractor, "attach_fields_to_chunks",
        lambda chunks, fields_by_doc: [{**c, "alanlar": {"konu": "Test"}} for c in chunks],
    )
    monkeypatch.setattr(
        embedder, "embed_chunks",
        lambda chunks: [{**c, "embedding": [0.1, 0.2, 0.3]} for c in chunks],
    )


def test_ingest_document_happy_path_returns_source_doc_chunks_classification_fields(monkeypatch, tmp_path):
    _patch_ingest_chain(monkeypatch)
    monkeypatch.setattr(vector_store, "load_index_path", lambda: str(tmp_path / "idx"))
    monkeypatch.setattr(vector_store, "build_index", lambda embedded_chunks: ("fake_index", {0: embedded_chunks[0]}))
    saved = {}
    monkeypatch.setattr(vector_store, "save_index", lambda index, metadata, path: saved.update(index=index, metadata=metadata, path=path))

    result = pipeline.ingest_document("data/raw_docs/test.png")

    assert result["source_doc"] == "test.png"
    assert result["num_chunks"] > 0
    assert result["classification"]["siniflar"] == ["fatura"]
    assert result["fields"]["konu"] == "Test"
    assert saved["path"] == str(tmp_path / "idx")


def test_ingest_document_calls_on_step_in_order(monkeypatch, tmp_path):
    _patch_ingest_chain(monkeypatch)
    monkeypatch.setattr(vector_store, "load_index_path", lambda: str(tmp_path / "idx"))
    monkeypatch.setattr(vector_store, "build_index", lambda embedded_chunks: ("fake_index", {0: embedded_chunks[0]}))
    monkeypatch.setattr(vector_store, "save_index", lambda index, metadata, path: None)

    calls = []
    pipeline.ingest_document("data/raw_docs/test.png", on_step=lambda step, info: calls.append(step))

    assert calls == ["ocr", "chunking", "classification", "field_extraction", "embedding", "indexing"]


def test_ingest_document_without_on_step_does_not_raise(monkeypatch, tmp_path):
    _patch_ingest_chain(monkeypatch)
    monkeypatch.setattr(vector_store, "load_index_path", lambda: str(tmp_path / "idx"))
    monkeypatch.setattr(vector_store, "build_index", lambda embedded_chunks: ("fake_index", {0: embedded_chunks[0]}))
    monkeypatch.setattr(vector_store, "save_index", lambda index, metadata, path: None)

    result = pipeline.ingest_document("data/raw_docs/test.png")
    assert result["source_doc"] == "test.png"


def test_ingest_document_raises_value_error_when_ocr_returns_empty_text(monkeypatch):
    monkeypatch.setattr(ocr, "extract_text_from_image", lambda image_path: "   ")
    monkeypatch.setattr(vector_store, "load_index_path", lambda: "fake_path")

    import pytest
    with pytest.raises(ValueError):
        pipeline.ingest_document("data/raw_docs/bos.png")


def test_ingest_document_extends_existing_index_when_faiss_file_exists(monkeypatch, tmp_path):
    _patch_ingest_chain(monkeypatch)
    index_path = tmp_path / "idx"
    (tmp_path / "idx.faiss").write_bytes(b"fake")
    monkeypatch.setattr(vector_store, "load_index_path", lambda: str(index_path))
    monkeypatch.setattr(vector_store, "load_index", lambda path: ("existing_index", {0: {"source_doc": "onceki.png"}}))

    add_chunks_called = {}
    monkeypatch.setattr(
        vector_store, "add_chunks",
        lambda index, metadata, embedded_chunks: add_chunks_called.update(called=True) or (index, metadata),
    )
    monkeypatch.setattr(vector_store, "save_index", lambda index, metadata, path: None)

    pipeline.ingest_document("data/raw_docs/yeni.png", index_path=str(index_path))
    assert add_chunks_called.get("called") is True


def _patch_search_chain(monkeypatch):
    monkeypatch.setattr(vector_store, "load_index_path", lambda: "fake_path")
    monkeypatch.setattr(vector_store, "load_index", lambda path: ("fake_index", {0: {"text": "x", "source_doc": "a.png"}}))
    monkeypatch.setattr(embedder, "embed_chunks", lambda chunks: [{**c, "embedding": [0.1, 0.2]} for c in chunks])


def test_search_documents_uses_hybrid_by_default(monkeypatch):
    _patch_search_chain(monkeypatch)
    called = {}
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: called.update(kwargs=k) or [{"score": 0.03, "text": "x"}])

    results = pipeline.search_documents("laptop talebi", top_k=3)

    assert results == [{"score": 0.03, "text": "x"}]
    assert "metadata_filter" in called["kwargs"]


def test_search_documents_use_hybrid_false_calls_vector_store_search(monkeypatch):
    _patch_search_chain(monkeypatch)
    called = {}
    monkeypatch.setattr(vector_store, "search", lambda *a, **k: called.update(kwargs=k) or [{"score": 0.9, "text": "x"}])

    results = pipeline.search_documents("laptop talebi", top_k=3, use_hybrid=False)

    assert results == [{"score": 0.9, "text": "x"}]
    assert called["kwargs"]["top_k"] == 3


def test_search_documents_expand_query_calls_hyde_and_combines_text(monkeypatch):
    _patch_search_chain(monkeypatch)
    monkeypatch.setattr(query_rewriter, "generate_hypothetical_answer", lambda query: "varsayimsal cevap metni")
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [])

    embedded_texts = []
    monkeypatch.setattr(
        embedder, "embed_chunks",
        lambda chunks: embedded_texts.append(chunks[0]["text"]) or [{**chunks[0], "embedding": [0.1]}],
    )

    pipeline.search_documents("laptop talebi", expand_query=True)

    assert "laptop talebi" in embedded_texts[0]
    assert "varsayimsal cevap metni" in embedded_texts[0]


def test_search_documents_without_expand_query_does_not_call_hyde(monkeypatch):
    _patch_search_chain(monkeypatch)
    monkeypatch.setattr(retrieval, "hybrid_search", lambda *a, **k: [])

    def _fail(query):
        raise AssertionError("HyDE cagrilmamali")

    monkeypatch.setattr(query_rewriter, "generate_hypothetical_answer", _fail)
    pipeline.search_documents("laptop talebi", expand_query=False)


def test_search_documents_passes_metadata_filter_through(monkeypatch):
    _patch_search_chain(monkeypatch)
    called = {}
    monkeypatch.setattr(vector_store, "search", lambda *a, **k: called.update(kwargs=k) or [])

    my_filter = lambda m: True  # noqa: E731
    pipeline.search_documents("sorgu", use_hybrid=False, metadata_filter=my_filter)

    assert called["kwargs"]["metadata_filter"] is my_filter
