"""
Metin bloklarini vektor uzayina ceviren embedding modulu.

Yerel (offline) calisan cok dilli bir sentence-transformers modeli kullanir,
boylece API anahtari veya internet baglantisi gerektirmeden (ilk model
indirmesi haric) embedding uretilebilir. Turkce metinler icin
"paraphrase-multilingual-MiniLM-L12-v2" secildi.
"""
from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_model_cache: dict[str, SentenceTransformer] = {}


def _get_model(model_name: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    if model_name not in _model_cache:
        logger.info("embedder: model yukleniyor (ilk kullanim): %s", model_name)
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_chunks(
    chunks: list[dict],
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[dict]:
    """text_splitter.split_text() ciktisindaki her chunk'a "embedding" alani ekler.

    Args:
        chunks: [{"chunk_id": int, "text": str, "token_count": int}, ...]
        model_name: kullanilacak sentence-transformers model adi.

    Returns:
        Her ogeye "embedding" (float listesi) eklenmis, girdiyle ayni sirada
        yeni bir liste.
    """
    if not chunks:
        return []

    model = _get_model(model_name)
    texts = [c["text"] for c in chunks]
    logger.info("embed_chunks: %d chunk vektorlestiriliyor (model=%s).", len(chunks), model_name)
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    return [
        {**chunk, "embedding": vector.tolist()}
        for chunk, vector in zip(chunks, vectors)
    ]


def embedding_dimension(model_name: str = DEFAULT_MODEL_NAME) -> int:
    # mypy: sentence-transformers stub'i get_embedding_dimension()'i
    # `int | None` olarak isaretliyor; yuklenmis bir modelde bu deger
    # pratikte hep dolu gelir (None sadece hic model yuklenmemisken teorik
    # olarak mumkun, ki _get_model() zaten modeli yukler).
    return _get_model(model_name).get_embedding_dimension()  # type: ignore[return-value]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """embed_chunks normalize_embeddings=True ile ciktigi icin dot product yeterli,
    ama fonksiyon vektorler normalize olmasa da dogru sonuc versin diye guvenli
    hesaplaniyor."""
    a_arr, b_arr = np.array(a), np.array(b)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if denom == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / denom)
