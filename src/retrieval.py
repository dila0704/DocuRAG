"""
Hybrid (BM25 + dense) semantik arama modulu (DOC-30, "wow" ozellik seti A1).

vector_store.search() (FAISS cosine similarity, IndexFlatIP) sadece anlam
bazli benzerlik buluyor -- tam kelime/anahtar kelime eslesmesi gereken
sorgularda (orn. bir belge numarasi, bir isim) zayif kalabiliyor. Bu modul
dense (FAISS) ve BM25 (anahtar kelime) siralamalarini Reciprocal Rank Fusion
(RRF) ile birlestirir.

Bilinen sinir: `_normalize_for_bm25` Turkce'nin sondan eklemeli
(agglutinative) yapisina karsi bir kok/lemma normalizasyonu YAPMAZ (rank_bm25
saf kelime-bazli calisir); sadece kucuk harfe cevirme + noktalama temizligi
uygulanir. Agir bir Turkce NLP kutuphanesi eklemek yerine bu sinir bilinclidir
(bkz. README "Bilinen Sinirlar").
"""
from __future__ import annotations

import logging
import re
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

DEFAULT_RRF_K = 60

# mMARCO (multilingual MS MARCO, Turkce dahil) uzerinde fine-tune edilmis bir
# cross-encoder -- embedder.py'nin zaten kullandigi
# paraphrase-multilingual-MiniLM-L12-v2 ile ayni "MiniLM-L12" agirlik
# sinifinda. NOT: sentence-transformers.CrossEncoder SARMALAYICISI bu repo
# icin transformers'in yeni "processor" auto-detection mantigiyla uyumsuz
# cikti (gercek modelle test edilirken bulundu); bu yuzden llm_factory.
# LocalHFClient'daki ayni desenle, dogrudan transformers Auto siniflari
# kullanilir -- model, sentence-transformers'in higher-level wrapper'indan
# BAGIMSIZ, guvenilir sekilde calisir.
DEFAULT_RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

_cross_encoder_cache: dict[str, tuple] = {}


def _get_cross_encoder(model_name: str = DEFAULT_RERANK_MODEL):
    """embedder._get_model()/llm_factory._get_local_model()'daki module-seviyesi
    onbellekleme deseniyle tutarli: ayni model_name ile tekrar cagrildiginda
    yeniden yuklenmez. (tokenizer, model) cifti dondurur."""
    if model_name not in _cross_encoder_cache:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("retrieval: cross-encoder yukleniyor (ilk kullanim): %s", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        _cross_encoder_cache[model_name] = (tokenizer, model)
    return _cross_encoder_cache[model_name]


def rerank(query: str, candidates: list[dict], top_k: int = 5, model_name: str = DEFAULT_RERANK_MODEL) -> list[dict]:
    """Bir aday listesini (dense/BM25/hybrid_search ciktisi gibi, her ogede
    "text" alani olmali) cross-encoder ile yeniden siralar.

    Mevcut "score" alanina DOKUNULMAZ (hangi ilk-asama siralamadan geldigi
    kaybolmaz); yeni bir "rerank_score" alani eklenir ve donen liste bu alana
    gore en yuksekten en dusuge siralanip top_k'ya indirgenir."""
    if not candidates or top_k <= 0:
        return []

    import torch

    tokenizer, model = _get_cross_encoder(model_name)
    texts = [c.get("text", "") for c in candidates]
    inputs = tokenizer([query] * len(texts), texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**inputs).logits.view(-1).float().tolist()

    reranked = [
        {**candidate, "rerank_score": float(score)}
        for candidate, score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]


def _normalize_for_bm25(text: str) -> list[str]:
    """Kucuk harf + \\w+ tokenizasyon. Turkce morfolojik kok bulma YAPMAZ
    (bilinen sinir, bkz. modul docstring'i)."""
    return _TOKEN_PATTERN.findall(text.lower())


def build_bm25_index(metadata: dict[int, dict]) -> tuple[BM25Okapi | None, list[int]]:
    """metadata'daki tum chunk metinlerinden bir BM25 indeksi kurar.

    Returns:
        (bm25, id_order) - id_order, BM25Okapi'nin pozisyonel sonuclarini
        gercek chunk id'lerine geri eslemek icin gerekli (rank_bm25 pozisyonel
        calisir, vector_store ise IndexIDMap ile gercek id'ler kullanir).
        metadata bossa (None, []) doner.
    """
    id_order = sorted(metadata.keys())
    if not id_order:
        return None, []
    corpus = [_normalize_for_bm25(metadata[doc_id].get("text", "")) for doc_id in id_order]
    return BM25Okapi(corpus), id_order


def bm25_search(bm25: BM25Okapi | None, id_order: list[int], query: str, top_k: int) -> list[tuple[int, float]]:
    """(chunk_id, bm25_score) ciftlerini en yuksek skordan en dusuge dondurur."""
    if bm25 is None or not id_order or top_k <= 0:
        return []
    scores = bm25.get_scores(_normalize_for_bm25(query))
    ranked = sorted(zip(id_order, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = DEFAULT_RRF_K) -> list[tuple[int, float]]:
    """Birden fazla siralanmis id listesini (dense, bm25, ...) standart RRF
    formuluyle (score = sum(1/(k+rank+1))) birlestirir. Herhangi sayida
    siralama listesi kabul eder."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def _dense_search_with_ids(index, query_embedding: list[float], top_k: int) -> list[tuple[int, float]]:
    """vector_store.search()'un ic mantiginin id-donduren kopyasi.

    vector_store.search() kasitli olarak sadece metadata dondurur (id'yi
    disariya sizdirmaz) -- o fonksiyonun sozlesmesini/imzasini degistirmemek
    icin RRF'nin ihtiyac duydugu id'ler burada, kucuk bir kopya mantikla,
    ayrica hesaplanir."""
    if index.ntotal == 0 or top_k <= 0:
        return []
    query_vector = np.array([query_embedding], dtype="float32")
    scores, indices = index.search(query_vector, min(top_k, index.ntotal))
    return [(int(idx), float(score)) for score, idx in zip(scores[0], indices[0]) if idx != -1]


def hybrid_search(
    index,
    metadata: dict[int, dict],
    query: str,
    query_embedding: list[float],
    top_k: int = 5,
    dense_top_k: int | None = None,
    bm25_top_k: int | None = None,
    bm25_index: tuple[BM25Okapi | None, list[int]] | None = None,
    rerank_top_n: int | None = None,
    rerank_model: str = DEFAULT_RERANK_MODEL,
    metadata_filter: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """Dense (FAISS cosine) ve BM25 siralamalarini RRF ile birlestirir.

    Donen "score" alani ARTIK COSINE BENZERLIGI DEGIL, RRF skorudur (kucuk
    bir sayi, mutlak deger olarak degil SADECE SIRALAMA icin anlamlidir).
    Bunu ayirt edebilmek icin her sonuca "score_type": "rrf" eklenir
    (vector_store.search()'un duz cosine skorlarindan bilerek ayristirilir).

    dense_top_k/bm25_top_k verilmezse top_k ile en az 20 arasindaki buyuk
    deger kullanilir -- boylece hem kucuk top_k'lerde (orn. 5) yeterince genis
    bir aday havuzundan fusion yapilir, hem de "TUM chunk'lari getir" gibi
    buyuk top_k istekleri (bkz. app/views/search.py ALL_CHUNKS_TOP_K) dogru
    calisir.

    bm25_index: onceden kurulmus (BM25Okapi, id_order) cifti verilirse (orn.
    app/data_access.get_bm25_index()'in mtime-tabanli onbellegi) BM25 indeksi
    sifirdan kurulmaz.
    rerank_top_n: verilirse, RRF sonrasi ilk rerank_top_n aday cross-encoder
    ile yeniden siralanip top_k'ya indirgenir (bkz. rerank()). None ise
    (varsayilan) reranking YAPILMAZ -- yeni model indirme/inference gecikmesi
    varsayilan arama akisini yavaslatmasin diye.
    metadata_filter: verilirse (orn. field_extractor.build_amount_range_filter()),
    SADECE bu fonksiyonun True dondugu chunk'lar sonuca girer -- vector_store.
    search()'un ayni parametresiyle tutarli. Filtre yeterince aday bulabilsin
    diye dense_top_k/bm25_top_k verilmemisse otomatik genisletilir (overfetch).
    """
    if not metadata or top_k <= 0:
        return []

    default_pool = max(top_k * 20, 20) if metadata_filter is not None else max(top_k, 20)
    dense_top_k = dense_top_k if dense_top_k is not None else default_pool
    bm25_top_k = bm25_top_k if bm25_top_k is not None else default_pool

    dense_ranking = [doc_id for doc_id, _ in _dense_search_with_ids(index, query_embedding, dense_top_k)]

    bm25, id_order = bm25_index if bm25_index is not None else build_bm25_index(metadata)
    bm25_ranking = [doc_id for doc_id, _ in bm25_search(bm25, id_order, query, bm25_top_k)]

    fused = reciprocal_rank_fusion([dense_ranking, bm25_ranking])
    logger.info(
        "hybrid_search: sorgu=%r dense_aday=%d bm25_aday=%d fused=%d top_k=%d",
        query, len(dense_ranking), len(bm25_ranking), len(fused), top_k,
    )

    all_candidates = [
        {"score": rrf_score, "score_type": "rrf", **metadata[doc_id]}
        for doc_id, rrf_score in fused
        if doc_id in metadata
    ]
    if metadata_filter is not None:
        all_candidates = [c for c in all_candidates if metadata_filter(c)]

    fetch_n = rerank_top_n if rerank_top_n is not None else top_k
    candidates = all_candidates[:fetch_n]

    if rerank_top_n is None:
        return candidates[:top_k]
    return rerank(query, candidates, top_k=top_k, model_name=rerank_model)
