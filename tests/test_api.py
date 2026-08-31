import io

from fastapi.testclient import TestClient

import api
import pipeline
import vector_store

client = TestClient(api.app)


def test_ingest_document_endpoint_calls_pipeline(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "ingest_document", lambda image_path: {
        "source_doc": "test.png",
        "num_chunks": 3,
        "classification": {"siniflar": ["fatura"], "guven": 0.9, "etiketler": ["ornek"], "human_review": False},
        "fields": {"tarih": "12.08.2026", "tutar": "500 TL", "taraflar": [], "belge_no": None, "konu": "Laptop talebi"},
    })

    response = client.post(
        "/documents",
        files={"image": ("test.png", io.BytesIO(b"fake-image-bytes"), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["source_doc"] == "test.png"
    assert body["num_chunks"] == 3
    assert body["classification"]["siniflar"] == ["fatura"]
    assert body["fields"]["konu"] == "Laptop talebi"


def test_ingest_document_endpoint_returns_422_on_pipeline_error(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOAD_DIR", tmp_path)

    def _raise(image_path):
        raise ValueError("OCR sonucu bos")

    monkeypatch.setattr(pipeline, "ingest_document", _raise)

    response = client.post(
        "/documents",
        files={"image": ("bad.png", io.BytesIO(b"x"), "image/png")},
    )
    assert response.status_code == 422


def test_search_endpoint_calls_pipeline(monkeypatch):
    monkeypatch.setattr(pipeline, "search_documents", lambda query, top_k=5, use_hybrid=True, use_reranker=False, expand_query=False: [
        {"score": 0.87, "text": "laptop talebi", "source_doc": "a.png", "siniflar": ["talep formu"], "guven": 0.9, "etiketler": [], "human_review": False},
    ])

    response = client.get("/search", params={"q": "laptop", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["source_doc"] == "a.png"
    assert body[0]["score"] == 0.87


def test_search_endpoint_returns_404_when_no_index(monkeypatch):
    def _raise(query, top_k=5, use_hybrid=True, use_reranker=False, expand_query=False):
        raise FileNotFoundError()

    monkeypatch.setattr(pipeline, "search_documents", _raise)
    response = client.get("/search", params={"q": "x"})
    assert response.status_code == 404


def test_list_documents_endpoint_groups_by_source_doc(monkeypatch):
    metadata = {
        0: {"source_doc": "a.png", "chunk_id": 0, "siniflar": ["fatura"], "guven": 0.9, "human_review": False, "ingested_at": "2026-01-01T00:00:00+00:00"},
        1: {"source_doc": "a.png", "chunk_id": 1, "siniflar": ["fatura"], "guven": 0.9, "human_review": False, "ingested_at": "2026-01-01T00:00:00+00:00"},
        2: {"source_doc": "b.png", "chunk_id": 0, "siniflar": ["dilekçe"], "guven": 0.5, "human_review": True, "ingested_at": "2026-01-02T00:00:00+00:00"},
    }
    monkeypatch.setattr(vector_store, "load_index", lambda path: (None, metadata))
    monkeypatch.setattr(vector_store, "load_index_path", lambda: "fake_path")

    response = client.get("/documents")

    assert response.status_code == 200
    body = {d["source_doc"]: d for d in response.json()}
    assert body["a.png"]["chunk_count"] == 2
    assert body["b.png"]["chunk_count"] == 1
    assert body["b.png"]["human_review"] is True


def test_list_documents_endpoint_returns_empty_list_when_no_index(monkeypatch):
    def _raise(path):
        raise FileNotFoundError()

    monkeypatch.setattr(vector_store, "load_index", _raise)
    monkeypatch.setattr(vector_store, "load_index_path", lambda: "fake_path")

    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == []


def test_delete_document_endpoint_calls_vector_store(monkeypatch):
    metadata = {0: {"source_doc": "a.png"}}
    monkeypatch.setattr(vector_store, "load_index", lambda path: ("fake_index", metadata))
    monkeypatch.setattr(vector_store, "load_index_path", lambda: "fake_path")
    monkeypatch.setattr(vector_store, "delete_by_source_doc", lambda index, meta, doc: (index, {}))
    saved = {}
    monkeypatch.setattr(vector_store, "save_index", lambda index, meta, path: saved.update(index=index, meta=meta, path=path))

    response = client.delete("/documents/a.png")

    assert response.status_code == 200
    assert response.json() == {"source_doc": "a.png", "deleted": True}
    assert saved["path"] == "fake_path"


def test_delete_document_endpoint_returns_404_when_not_found(monkeypatch):
    metadata = {0: {"source_doc": "a.png"}}
    monkeypatch.setattr(vector_store, "load_index", lambda path: ("fake_index", metadata))
    monkeypatch.setattr(vector_store, "load_index_path", lambda: "fake_path")

    response = client.delete("/documents/does_not_exist.png")
    assert response.status_code == 404
