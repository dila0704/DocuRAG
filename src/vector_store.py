"""
Embedding'leri FAISS vektor veritabanina kaydeden/okuyan ve uzerinde
benzerlik aramasi yapan modul.

FAISS sadece vektorleri (sayilari) saklar, chunk metnini/meta verisini
saklamaz. Bu yuzden index dosyasinin yaninda ayni isimle bir ".meta.json"
dosyasi tutuyoruz: index'teki i. vektorun karsiligi metadata[i] olacak
sekilde sirali.

embedder.embed_chunks() vektorleri normalize_embeddings=True ile urettigi
icin, kosinus benzerligi = ic carpim (inner product). Bu yuzden
faiss.IndexFlatIP kullaniliyor.
"""
from __future__ import annotations

import json
import os

import faiss
import numpy as np
import yaml

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")


def load_index_path(config_path: str = DEFAULT_CONFIG_PATH) -> str:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["vector_db"]["path"]


def build_index(embedded_chunks: list[dict]) -> tuple[faiss.Index, list[dict]]:
    """embedder.embed_chunks() ciktisindan bir FAISS index ve metadata listesi kurar.

    Returns:
        (index, metadata) - metadata[i], index'teki i. vektorun
        (embedding harici) chunk bilgilerini icerir.
    """
    if not embedded_chunks:
        raise ValueError("embedded_chunks bos olamaz.")

    dim = len(embedded_chunks[0]["embedding"])
    vectors = np.array([c["embedding"] for c in embedded_chunks], dtype="float32")

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    metadata = [{k: v for k, v in c.items() if k != "embedding"} for c in embedded_chunks]
    return index, metadata


def save_index(index: faiss.Index, metadata: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    faiss.write_index(index, path + ".faiss")
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_index(path: str) -> tuple[faiss.Index, list[dict]]:
    index = faiss.read_index(path + ".faiss")
    with open(path + ".meta.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def search(
    index: faiss.Index,
    metadata: list[dict],
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    """En yakin top_k chunk'i, en benzerden en az benzere dogru dondurur.

    Returns:
        [{"score": float, **chunk_metadata}, ...]
    """
    query_vector = np.array([query_embedding], dtype="float32")
    scores, indices = index.search(query_vector, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({"score": float(score), **metadata[idx]})
    return results
