import numpy as np

import embedder


class _FakeModel:
    """sentence_transformers.SentenceTransformer'in gercek modeli indirmeden
    testte kullanilan sahte (deterministik) karsiligi."""

    def __init__(self, dim: int = 4):
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        vectors = []
        for text in texts:
            seed = (ord(text[0]) if text else 0) % 10
            vec = np.array([seed + i for i in range(self.dim)], dtype="float32")
            norm = np.linalg.norm(vec)
            vectors.append(vec / norm if norm else vec)
        return np.array(vectors)

    def get_embedding_dimension(self):
        return self.dim


def _patch_model(monkeypatch, fake_model):
    monkeypatch.setattr(embedder, "_get_model", lambda model_name=embedder.DEFAULT_MODEL_NAME: fake_model)


def test_embed_chunks_adds_embedding_field(monkeypatch):
    fake_model = _FakeModel(dim=4)
    _patch_model(monkeypatch, fake_model)

    chunks = [{"chunk_id": 0, "text": "Merhaba", "token_count": 1}]
    result = embedder.embed_chunks(chunks)

    assert len(result) == 1
    assert "embedding" in result[0]
    assert len(result[0]["embedding"]) == fake_model.dim
    assert result[0]["chunk_id"] == 0  # orijinal alanlar korunmali


def test_embed_chunks_preserves_order(monkeypatch):
    _patch_model(monkeypatch, _FakeModel(dim=4))
    chunks = [
        {"chunk_id": 0, "text": "Alfa", "token_count": 1},
        {"chunk_id": 1, "text": "Beta", "token_count": 1},
    ]
    result = embedder.embed_chunks(chunks)
    assert [c["text"] for c in result] == ["Alfa", "Beta"]


def test_embed_chunks_empty_list_returns_empty_list(monkeypatch):
    _patch_model(monkeypatch, _FakeModel())
    assert embedder.embed_chunks([]) == []


def test_embedding_dimension(monkeypatch):
    _patch_model(monkeypatch, _FakeModel(dim=8))
    assert embedder.embedding_dimension() == 8


def test_cosine_similarity_identical_vectors_is_one():
    vector = [1.0, 2.0, 3.0]
    assert abs(embedder.cosine_similarity(vector, vector) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert abs(embedder.cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9


def test_cosine_similarity_zero_vector_is_zero():
    assert embedder.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
