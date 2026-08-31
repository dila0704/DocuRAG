import numpy as np
import pytest

import retrieval
import vector_store as vs


def _embedded_chunk(text: str, source_doc: str, seed: float, dim: int = 4) -> dict:
    raw = np.array([(seed + i) % 7 for i in range(dim)], dtype="float32")
    norm = np.linalg.norm(raw)
    vec = (raw / norm if norm else raw).tolist()
    return {"chunk_id": 0, "text": text, "token_count": len(text.split()), "source_doc": source_doc, "embedding": vec}


def test_normalize_for_bm25_lowercases_and_tokenizes():
    assert retrieval._normalize_for_bm25("Laptop Talebi, Ofis!") == ["laptop", "talebi", "ofis"]


def test_normalize_for_bm25_empty_string_returns_empty_list():
    assert retrieval._normalize_for_bm25("") == []


def test_reciprocal_rank_fusion_combines_rankings_with_expected_order():
    fused = retrieval.reciprocal_rank_fusion([[10, 20, 30], [30, 10, 20]], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids == [10, 30, 20]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_reciprocal_rank_fusion_empty_rankings_returns_empty_list():
    assert retrieval.reciprocal_rank_fusion([]) == []


def test_reciprocal_rank_fusion_id_only_in_one_ranking_still_scored():
    fused = dict(retrieval.reciprocal_rank_fusion([[1], []], k=60))
    assert fused[1] == pytest.approx(1 / 61)


def test_build_bm25_index_empty_metadata_returns_none_sentinel():
    bm25, id_order = retrieval.build_bm25_index({})
    assert bm25 is None
    assert id_order == []


def test_build_bm25_index_and_search_finds_keyword_match():
    # BM25'in IDF hesabi (log((N-df+0.5)/(df+0.5))) N=2, df=1 icin tam 0
    # cikiyor (tum belgelerin yarisinda gecen bir terim bilgi vermiyor
    # sayilir) -- ayirt edici olmasi icin en az 3 belgelik bir korpus gerekli.
    metadata = {
        0: {"text": "laptop talebi ofis ekipmani", "source_doc": "a.png"},
        1: {"text": "kira sozlesmesi daire", "source_doc": "b.png"},
        2: {"text": "toplanti notlari proje", "source_doc": "c.png"},
    }
    bm25, id_order = retrieval.build_bm25_index(metadata)
    ranked = retrieval.bm25_search(bm25, id_order, "kira sozlesmesi", top_k=3)
    assert ranked[0][0] == 1  # "kira sozlesmesi" b.png ile en cok eslesir


def test_bm25_search_with_none_bm25_returns_empty_list():
    assert retrieval.bm25_search(None, [], "sorgu", top_k=5) == []


def test_hybrid_search_empty_metadata_returns_empty_list():
    assert retrieval.hybrid_search(None, {}, "sorgu", [0.1, 0.2], top_k=5) == []


def test_hybrid_search_merges_dense_and_bm25_matches():
    chunks = [
        _embedded_chunk("laptop talebi ofis ekipmani", "a.png", seed=1),
        _embedded_chunk("kira sozlesmesi daire", "b.png", seed=5),
    ]
    index, metadata = vs.build_index(chunks)

    # sorgu embedding'i a.png ile ayni yonde (dense eslesme), sorgu metni ise
    # SADECE b.png'nin kelimeleriyle ortusuyor (bm25 eslesme) -- ikisi de
    # sonuca girmeli, bu da fusion'un gercekten calistigini kanitlar.
    query_embedding = _embedded_chunk("x", "q", seed=1)["embedding"]
    results = retrieval.hybrid_search(index, metadata, "kira sozlesmesi", query_embedding, top_k=2)

    result_docs = {r["source_doc"] for r in results}
    assert result_docs == {"a.png", "b.png"}
    assert all(r["score_type"] == "rrf" for r in results)


def test_hybrid_search_respects_top_k():
    chunks = [_embedded_chunk(f"belge {i}", f"doc_{i}.png", seed=i) for i in range(5)]
    index, metadata = vs.build_index(chunks)
    query_embedding = _embedded_chunk("x", "q", seed=0)["embedding"]

    results = retrieval.hybrid_search(index, metadata, "belge", query_embedding, top_k=2)
    assert len(results) == 2


class _FakeLogits:
    """torch tensor'unun rerank()'in kullandigi .view()/.float()/.tolist()
    zincirini taklit eden minimal sahte nesne."""

    def __init__(self, values):
        self._values = values

    def view(self, *_args):
        return self

    def float(self):
        return self

    def tolist(self):
        return self._values


class _FakeCrossEncoderModel:
    def __init__(self, values):
        self._values = values

    def __call__(self, **_inputs):
        return type("Output", (), {"logits": _FakeLogits(self._values)})()


class _FakeTokenizer:
    def __call__(self, queries, texts, padding=True, truncation=True, return_tensors="pt"):
        return {}


def _patch_cross_encoder(monkeypatch, scores):
    monkeypatch.setattr(
        retrieval, "_get_cross_encoder",
        lambda model_name=retrieval.DEFAULT_RERANK_MODEL: (_FakeTokenizer(), _FakeCrossEncoderModel(scores)),
    )


def test_rerank_sorts_by_cross_encoder_score(monkeypatch):
    _patch_cross_encoder(monkeypatch, [0.1, 0.9])

    candidates = [
        {"text": "alakasiz metin", "source_doc": "a.png"},
        {"text": "cok alakali metin", "source_doc": "b.png"},
    ]
    result = retrieval.rerank("sorgu", candidates, top_k=2)
    assert [r["source_doc"] for r in result] == ["b.png", "a.png"]
    assert result[0]["rerank_score"] == 0.9


def test_rerank_empty_candidates_returns_empty_list():
    assert retrieval.rerank("sorgu", [], top_k=5) == []


def test_rerank_respects_top_k(monkeypatch):
    _patch_cross_encoder(monkeypatch, [0.5, 0.9, 0.1])
    candidates = [{"text": f"metin {i}"} for i in range(3)]
    result = retrieval.rerank("sorgu", candidates, top_k=1)
    assert len(result) == 1
    assert result[0]["rerank_score"] == 0.9


def test_hybrid_search_with_rerank_top_n_delegates_to_rerank(monkeypatch):
    chunks = [_embedded_chunk(f"belge {i}", f"doc_{i}.png", seed=i) for i in range(3)]
    index, metadata = vs.build_index(chunks)
    query_embedding = _embedded_chunk("x", "q", seed=0)["embedding"]

    called = {}

    def _fake_rerank(query, candidates, top_k=5, model_name=retrieval.DEFAULT_RERANK_MODEL):
        called["count"] = len(candidates)
        return candidates[:top_k]

    monkeypatch.setattr(retrieval, "rerank", _fake_rerank)
    results = retrieval.hybrid_search(index, metadata, "belge", query_embedding, top_k=2, rerank_top_n=3)

    assert called["count"] > 0
    assert len(results) == 2


def test_hybrid_search_without_rerank_top_n_never_calls_rerank(monkeypatch):
    chunks = [_embedded_chunk("belge", "doc.png", seed=0)]
    index, metadata = vs.build_index(chunks)
    query_embedding = _embedded_chunk("x", "q", seed=0)["embedding"]

    def _fail(*a, **k):
        raise AssertionError("rerank cagrilmamali")

    monkeypatch.setattr(retrieval, "rerank", _fail)
    results = retrieval.hybrid_search(index, metadata, "belge", query_embedding, top_k=1)
    assert len(results) == 1


def test_hybrid_search_accepts_prebuilt_bm25_index():
    chunks = [
        _embedded_chunk("laptop talebi", "a.png", seed=1),
        _embedded_chunk("kira sozlesmesi", "b.png", seed=5),
    ]
    index, metadata = vs.build_index(chunks)
    prebuilt = retrieval.build_bm25_index(metadata)
    query_embedding = _embedded_chunk("x", "q", seed=1)["embedding"]

    results = retrieval.hybrid_search(
        index, metadata, "laptop talebi", query_embedding, top_k=2, bm25_index=prebuilt,
    )
    assert len(results) == 2
