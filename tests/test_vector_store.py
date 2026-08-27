import json

import faiss
import numpy as np
import pytest

import vector_store as vs


def _chunk(text: str, source_doc: str, seed: float, dim: int = 4) -> dict:
    # embedder.embed_chunks() vektorleri normalize_embeddings=True ile uretir;
    # IndexFlatIP'in ic carpimi kosinus benzerligi olarak yorumlayabilmesi
    # icin test vektorleri de normalize edilmis olmali (aksi halde farkli
    # buyuklukteki vektorler yanlis "en yakin" sonucu verir).
    raw = np.array([(seed + i) % 7 for i in range(dim)], dtype="float32")
    norm = np.linalg.norm(raw)
    vec = (raw / norm if norm else raw).tolist()
    return {"chunk_id": 0, "text": text, "token_count": len(text.split()), "source_doc": source_doc, "embedding": vec}


def test_build_index_assigns_sequential_ids():
    chunks = [_chunk("fatura metni", "a.png", seed=1), _chunk("sozlesme metni", "b.png", seed=2)]
    index, metadata = vs.build_index(chunks)
    assert index.ntotal == 2
    assert set(metadata.keys()) == {0, 1}
    assert metadata[0]["source_doc"] == "a.png"
    assert "embedding" not in metadata[0]


def test_build_index_empty_list_raises_value_error():
    with pytest.raises(ValueError):
        vs.build_index([])


def test_build_index_and_search_roundtrip_through_disk(tmp_path):
    chunks = [_chunk("fatura metni", "a.png", seed=1), _chunk("sozlesme metni", "b.png", seed=2)]
    index, metadata = vs.build_index(chunks)

    path = str(tmp_path / "idx")
    vs.save_index(index, metadata, path)
    loaded_index, loaded_metadata = vs.load_index(path)

    assert loaded_index.ntotal == 2
    assert loaded_metadata[0]["source_doc"] == "a.png"

    results = vs.search(loaded_index, loaded_metadata, chunks[0]["embedding"], top_k=1)
    assert len(results) == 1
    assert results[0]["source_doc"] == "a.png"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-4)


def test_search_returns_results_best_first():
    chunks = [_chunk("uzak", "a.png", seed=1), _chunk("yakin", "b.png", seed=1.01)]
    index, metadata = vs.build_index(chunks)
    results = vs.search(index, metadata, chunks[1]["embedding"], top_k=2)
    assert results[0]["source_doc"] == "b.png"
    assert results[0]["score"] >= results[1]["score"]


def test_search_on_empty_index_returns_empty_list():
    chunks = [_chunk("x", "a.png", seed=1)]
    index, metadata = vs.build_index(chunks)
    index, metadata = vs.delete_by_source_doc(index, metadata, "a.png")
    assert index.ntotal == 0
    assert vs.search(index, metadata, chunks[0]["embedding"], top_k=5) == []


def test_search_top_k_zero_returns_empty_list():
    chunks = [_chunk("x", "a.png", seed=1)]
    index, metadata = vs.build_index(chunks)
    assert vs.search(index, metadata, chunks[0]["embedding"], top_k=0) == []


def test_add_chunks_extends_index_without_rebuilding():
    index, metadata = vs.build_index([_chunk("fatura", "a.png", seed=1)])
    index, metadata = vs.add_chunks(index, metadata, [_chunk("sozlesme", "b.png", seed=5)])

    assert index.ntotal == 2
    assert len(metadata) == 2
    assert set(metadata.keys()) == {0, 1}


def test_add_chunks_with_empty_list_is_noop():
    index, metadata = vs.build_index([_chunk("fatura", "a.png", seed=1)])
    index2, metadata2 = vs.add_chunks(index, metadata, [])
    assert index2 is index
    assert metadata2 == metadata


def test_add_chunks_rejects_dimension_mismatch():
    index, metadata = vs.build_index([_chunk("fatura", "a.png", seed=1, dim=4)])
    with pytest.raises(ValueError):
        vs.add_chunks(index, metadata, [_chunk("sozlesme", "b.png", seed=2, dim=8)])


def test_delete_by_source_doc_removes_only_matching_chunks():
    chunks = [
        _chunk("a1", "a.png", seed=1),
        _chunk("a2", "a.png", seed=2),
        _chunk("b1", "b.png", seed=3),
    ]
    index, metadata = vs.build_index(chunks)
    index, metadata = vs.delete_by_source_doc(index, metadata, "a.png")

    assert index.ntotal == 1
    assert all(m["source_doc"] == "b.png" for m in metadata.values())


def test_delete_by_source_doc_with_no_match_is_noop():
    index, metadata = vs.build_index([_chunk("a1", "a.png", seed=1)])
    index2, metadata2 = vs.delete_by_source_doc(index, metadata, "does-not-exist.png")
    assert index2.ntotal == 1
    assert metadata2 == metadata


def test_update_metadata_by_source_doc_updates_matching_only():
    chunks = [_chunk("a1", "a.png", seed=1), _chunk("b1", "b.png", seed=2)]
    _, metadata = vs.build_index(chunks)

    updated = vs.update_metadata_by_source_doc(metadata, "a.png", {"human_review": False, "siniflar": ["fatura"]})

    assert updated[0]["human_review"] is False
    assert updated[0]["siniflar"] == ["fatura"]
    assert updated[1] == metadata[1]  # eslesmeyen kayit degismemis olmali


def test_load_index_migrates_legacy_flat_index_and_list_metadata(tmp_path):
    chunks = [_chunk("x", "a.png", seed=1), _chunk("y", "b.png", seed=2)]
    dim = len(chunks[0]["embedding"])

    legacy_index = faiss.IndexFlatIP(dim)
    legacy_index.add(np.array([c["embedding"] for c in chunks], dtype="float32"))
    legacy_metadata = [{k: v for k, v in c.items() if k != "embedding"} for c in chunks]

    path = str(tmp_path / "legacy")
    with open(path + ".faiss", "wb") as f:
        f.write(faiss.serialize_index(legacy_index).tobytes())
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(legacy_metadata, f, ensure_ascii=False)

    index, metadata = vs.load_index(path)

    assert isinstance(index, faiss.IndexIDMap)
    assert index.ntotal == 2
    assert metadata[0]["source_doc"] == "a.png"

    # Migrasyon sonrasi CRUD calisir olmali.
    index, metadata = vs.delete_by_source_doc(index, metadata, "a.png")
    assert index.ntotal == 1
