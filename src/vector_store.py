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


def save_metadata(metadata: list[dict], path: str) -> None:
    """Sadece metadata (".meta.json") dosyasini yazar, FAISS index'ine dokunmaz.

    Insan incelemesi sonrasi bir belgenin sinif/etiket/human_review bilgisi
    duzeltildiginde vektorler degismedigi icin index'i (ve dolayisiyla
    ".faiss" dosyasini) yeniden kurmaya gerek yoktur; sadece bu fonksiyonla
    metadata guncellenir. update_metadata_by_source_doc() ile birlikte
    kullanilir.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def update_metadata_by_source_doc(metadata: list[dict], source_doc: str, updates: dict) -> list[dict]:
    """Belirli bir source_doc'a ait tum chunk'larin metadata'sini gunceller.

    Insan incelemesi sonrasi bir belgenin kategorisi/etiketleri/human_review
    durumu duzeltildiginde kullanilir: belge OCR/embedding tamamlaninca
    zaten indekslenip aranabilir durumdadir (bkz. classifier.attach_labels_to_chunks),
    inceleme sonrasi bu fonksiyonla sadece ilgili alanlar guncellenir;
    vektorler ve index yeniden kurulmaz.

    Args:
        metadata: load_index() veya build_index() ciktisindaki metadata listesi.
        source_doc: guncellenecek belgenin dosya adi (chunk'lardaki "source_doc" alani).
        updates: metadata'ya uygulanacak alan guncellemeleri (orn. {"siniflar": [...], "human_review": False}).

    Returns:
        Guncellenmis yeni bir metadata listesi (girdi degistirilmez).
    """
    return [
        {**m, **updates} if m.get("source_doc") == source_doc else m
        for m in metadata
    ]


def save_index(index: faiss.Index, metadata: list[dict], path: str) -> None:
    """Index'i ve metadata'yi diske yazar.

    faiss.write_index() Windows'ta yol ASCII olmayan karakterler icerdiginde
    (ornegin kullanici adinda Turkce harf) native fopen cagrisinda basarisiz
    olabiliyor. Bunun onune gecmek icin index once bellekte serialize edilip
    (faiss.serialize_index), diske Python'un kendi (Unicode-guvenli) dosya
    I/O'su ile yaziliyor.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    index_bytes = faiss.serialize_index(index)
    with open(path + ".faiss", "wb") as f:
        f.write(index_bytes.tobytes())
    save_metadata(metadata, path)


def load_index(path: str) -> tuple[faiss.Index, list[dict]]:
    with open(path + ".faiss", "rb") as f:
        index_bytes = np.frombuffer(f.read(), dtype="uint8")
    index = faiss.deserialize_index(index_bytes)
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
