"""
Embedding'leri FAISS vektor veritabanina kaydeden/okuyan ve uzerinde
benzerlik aramasi yapan modul.

FAISS sadece vektorleri (sayilari) saklar, chunk metnini/meta verisini
saklamaz. Bu yuzden index dosyasinin yaninda ayni isimle bir ".meta.json"
dosyasi tutuyoruz.

embedder.embed_chunks() vektorleri normalize_embeddings=True ile urettigi
icin, kosinus benzerligi = ic carpim (inner product). Bu yuzden
faiss.IndexFlatIP kullaniliyor.

CRUD (DOC-31): Index, faiss.IndexIDMap ile sarmalanir; her chunk'a acik bir
tam sayi id atanir ve metadata bu id'ye gore bir dict (id -> alanlar) olarak
tutulur (pozisyonel bir liste degil). Boylece:
  - add_chunks(): var olan index'i sifirdan kurmadan yeni chunk ekleyebilir.
  - delete_by_source_doc(): index.remove_ids() ile secili vektorleri
    (ve karsilik gelen metadata girdilerini) index'i yeniden kurmadan siler.
Eski (duz IndexFlatIP + liste metadata) formatinda kaydedilmis index'ler
load_index() tarafindan otomatik olarak bu id-tabanli formata gocurulur.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Callable

import faiss
import numpy as np
import yaml

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")


def load_index_path(config_path: str = DEFAULT_CONFIG_PATH) -> str:
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["vector_db"]["path"]


def _validate_dimension(index: faiss.Index, embedded_chunks: list[dict]) -> np.ndarray:
    vectors = np.array([c["embedding"] for c in embedded_chunks], dtype="float32")
    if vectors.shape[1] != index.d:
        raise ValueError(
            f"Embedding boyutu ({vectors.shape[1]}) index boyutuyla ({index.d}) uyusmuyor. "
            "Farkli bir embedding modeliyle uretilmis chunk'lar ayni index'e eklenemez."
        )
    return vectors


def build_index(embedded_chunks: list[dict]) -> tuple[faiss.IndexIDMap, dict[int, dict]]:
    """embedder.embed_chunks() ciktisindan bir FAISS index ve id->metadata sozlugu kurar.

    Returns:
        (index, metadata) - metadata[i], id'si i olan vektorun (embedding
        harici) chunk bilgilerini icerir. id'ler 0'dan baslar.
    """
    if not embedded_chunks:
        raise ValueError("embedded_chunks bos olamaz.")

    dim = len(embedded_chunks[0]["embedding"])
    vectors = np.array([c["embedding"] for c in embedded_chunks], dtype="float32")

    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    ids = np.arange(len(embedded_chunks), dtype="int64")
    index.add_with_ids(vectors, ids)

    metadata = {
        int(i): {k: v for k, v in c.items() if k != "embedding"}
        for i, c in zip(ids, embedded_chunks)
    }
    logger.info("build_index: %d chunk ile yeni index kuruldu (dim=%d).", len(embedded_chunks), dim)
    return index, metadata


def add_chunks(
    index: faiss.IndexIDMap, metadata: dict[int, dict], embedded_chunks: list[dict]
) -> tuple[faiss.IndexIDMap, dict[int, dict]]:
    """Var olan index'i sifirdan kurmadan yeni chunk'lar ekler.

    Yeni id'ler, mevcut en buyuk id'nin devami olarak atanir.

    Args:
        index: build_index()/load_index() ciktisi bir IndexIDMap.
        metadata: build_index()/load_index() ciktisi id->alanlar sozlugu.
        embedded_chunks: embedder.embed_chunks() ciktisi, eklenecek yeni chunk'lar.

    Returns:
        (index, metadata) - index yerinde (in-place) guncellenir, metadata
        icin yeni bir sozluk dondurulur.
    """
    if not embedded_chunks:
        return index, metadata

    vectors = _validate_dimension(index, embedded_chunks)
    next_id = (max(metadata) + 1) if metadata else 0
    ids = np.arange(next_id, next_id + len(embedded_chunks), dtype="int64")
    index.add_with_ids(vectors, ids)

    new_metadata = dict(metadata)
    for i, c in zip(ids, embedded_chunks):
        new_metadata[int(i)] = {k: v for k, v in c.items() if k != "embedding"}

    logger.info("add_chunks: %d yeni chunk eklendi (yeni toplam=%d).", len(embedded_chunks), index.ntotal)
    return index, new_metadata


def delete_by_source_doc(
    index: faiss.IndexIDMap, metadata: dict[int, dict], source_doc: str
) -> tuple[faiss.IndexIDMap, dict[int, dict]]:
    """Belirli bir source_doc'a ait tum vektorleri/metadata'yi index'i
    yeniden kurmadan siler (faiss.IndexIDMap.remove_ids).

    Args:
        index: build_index()/load_index() ciktisi bir IndexIDMap.
        metadata: id->alanlar sozlugu.
        source_doc: silinecek belgenin dosya adi (chunk'lardaki "source_doc" alani).

    Returns:
        (index, metadata) - index yerinde guncellenir, metadata icin yeni
        bir sozluk dondurulur.
    """
    ids_to_remove = [doc_id for doc_id, m in metadata.items() if m.get("source_doc") == source_doc]
    if not ids_to_remove:
        logger.info("delete_by_source_doc: '%s' icin eslesen chunk bulunamadi.", source_doc)
        return index, metadata

    index.remove_ids(np.array(ids_to_remove, dtype="int64"))
    new_metadata = {doc_id: m for doc_id, m in metadata.items() if doc_id not in ids_to_remove}
    logger.info(
        "delete_by_source_doc: '%s' icin %d chunk silindi (yeni toplam=%d).",
        source_doc, len(ids_to_remove), index.ntotal,
    )
    return index, new_metadata


def save_metadata(metadata: dict[int, dict], path: str) -> None:
    """Sadece metadata (".meta.json") dosyasini yazar, FAISS index'ine dokunmaz.

    Insan incelemesi sonrasi bir belgenin sinif/etiket/human_review bilgisi
    duzeltildiginde vektorler degismedigi icin index'i (ve dolayisiyla
    ".faiss" dosyasini) yeniden kurmaya gerek yoktur; sadece bu fonksiyonla
    metadata guncellenir. update_metadata_by_source_doc() ile birlikte
    kullanilir.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    records = [{"id": doc_id, **fields} for doc_id, fields in sorted(metadata.items())]
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info("save_metadata: %d kayit '%s.meta.json' dosyasina yazildi.", len(records), path)


def _metadata_from_records(records: list[dict]) -> dict[int, dict]:
    """load_index()/load_metadata() icin ortak donusum.

    Eski formatta (id alani olmayan duz liste) kayitlarin id'si, listedeki
    pozisyonlariyla ayni kabul edilir -- bu, eski duz IndexFlatIP'in de
    vektorleri 0..n-1 sirasiyla sakladigi varsayimiyla tutarlidir.
    """
    metadata: dict[int, dict] = {}
    for position, record in enumerate(records):
        record = dict(record)
        doc_id = record.pop("id", position)
        metadata[int(doc_id)] = record
    return metadata


def update_metadata_by_source_doc(metadata: dict[int, dict], source_doc: str, updates: dict) -> dict[int, dict]:
    """Belirli bir source_doc'a ait tum chunk'larin metadata'sini gunceller.

    Insan incelemesi sonrasi bir belgenin kategorisi/etiketleri/human_review
    durumu duzeltildiginde kullanilir; vektorler ve index yeniden kurulmaz.

    Args:
        metadata: build_index()/load_index() ciktisi id->alanlar sozlugu.
        source_doc: guncellenecek belgenin dosya adi (chunk'lardaki "source_doc" alani).
        updates: metadata'ya uygulanacak alan guncellemeleri (orn. {"siniflar": [...], "human_review": False}).

    Returns:
        Guncellenmis yeni bir metadata sozlugu (girdi degistirilmez).
    """
    updated_count = sum(1 for m in metadata.values() if m.get("source_doc") == source_doc)
    logger.info("update_metadata_by_source_doc: '%s' icin %d kayit guncellenecek.", source_doc, updated_count)
    return {
        doc_id: ({**m, **updates} if m.get("source_doc") == source_doc else m)
        for doc_id, m in metadata.items()
    }


def save_index(index: faiss.Index, metadata: dict[int, dict], path: str) -> None:
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
    logger.info("save_index: index '%s.faiss' dosyasina yazildi (ntotal=%d).", path, index.ntotal)


def _migrate_to_id_map(index: faiss.Index) -> faiss.IndexIDMap:
    """Eski (IndexIDMap olmayan) bir index'i, mevcut vektorlerini 0..n-1
    id'leriyle yeniden ekleyerek IndexIDMap'e gocurur.

    Onceden `faiss.IndexIDMap(index)` ile sarmalamak yeterli DEGILDIR: bu,
    ic index'te zaten var olan vektorlere id atamaz (id_map bos kalir),
    remove_ids/search sonuclarini bozar. Bunun yerine vektorler
    reconstruct_n ile geri okunup id'leriyle birlikte yeni bir IndexIDMap'e
    eklenir.
    """
    n = index.ntotal
    vectors = index.reconstruct_n(0, n) if n > 0 else np.empty((0, index.d), dtype="float32")
    migrated = faiss.IndexIDMap(faiss.IndexFlatIP(index.d))
    if n > 0:
        migrated.add_with_ids(vectors, np.arange(n, dtype="int64"))
    return migrated


def load_index(path: str) -> tuple[faiss.IndexIDMap, dict[int, dict]]:
    with open(path + ".faiss", "rb") as f:
        index_bytes = np.frombuffer(f.read(), dtype="uint8")
    index = faiss.deserialize_index(index_bytes)
    with open(path + ".meta.json", encoding="utf-8") as f:
        records = json.load(f)
    metadata = _metadata_from_records(records)

    if not isinstance(index, faiss.IndexIDMap):
        logger.info("load_index: eski format (IndexIDMap degil) tespit edildi, id-tabanli formata gocuruluyor.")
        index = _migrate_to_id_map(index)

    logger.info("load_index: '%s' yuklendi (ntotal=%d).", path, index.ntotal)
    return index, metadata


def group_latest_by_source_doc(metadata: dict[int, dict]) -> dict[str, dict]:
    """metadata'daki chunk'lari source_doc'a gore gruplayip her belge icin en
    guncel (en buyuk "ingested_at") kaydi dondurur.

    app/views/dashboard.py, app/views/inventory.py ve app/api.py'de tekrarlanan
    ayni gruplama dongusunden cikarildi (DRY) -- hem Streamlit hem FastAPI
    tarafinin ayni "belge envanteri" mantigini kullanmasi icin src/ katmaninda.
    "ingested_at" alani olmayan (DOC-30 oncesi indekslenmis) kayitlar bos
    string ile karsilastirilir, boylece her zaman bir kayit secilir.
    """
    documents: dict[str, dict] = {}
    for m in metadata.values():
        doc = m.get("source_doc", "bilinmiyor")
        if doc not in documents or m.get("ingested_at", "") > documents[doc].get("ingested_at", ""):
            documents[doc] = m
    return documents


# metadata_filter verildiginde, top_k'nin bu kati kadar aday cekilip filtre
# uygulanir (search.py'deki eski ALL_CHUNKS_TOP_K=1000 "hepsini getir filtrele"
# hack'inin resmilesmis hali) -- amac, filtreden gecen yeterince aday bulmak.
_METADATA_FILTER_OVERFETCH_FACTOR = 20


def search(
    index: faiss.Index,
    metadata: dict[int, dict],
    query_embedding: list[float],
    top_k: int = 5,
    metadata_filter: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """En yakin top_k chunk'i, en benzerden en az benzere dogru dondurur.

    Args:
        metadata_filter: verilirse, SADECE bu fonksiyonun True dondugu
            chunk'lar sonuca girer (orn. field_extractor.build_amount_range_filter()).
            None (varsayilan) ise davranis TAMAMEN eskisiyle aynidir -- bu
            parametre geriye donuk uyumluluk icin opsiyonel tutuldu.

    Returns:
        [{"score": float, **chunk_metadata}, ...]
    """
    if index.ntotal == 0 or top_k <= 0:
        return []

    fetch_k = top_k
    if metadata_filter is not None:
        fetch_k = min(index.ntotal, max(top_k * _METADATA_FILTER_OVERFETCH_FACTOR, top_k))

    query_vector = np.array([query_embedding], dtype="float32")
    scores, indices = index.search(query_vector, min(fetch_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk_metadata = metadata[int(idx)]
        if metadata_filter is not None and not metadata_filter(chunk_metadata):
            continue
        results.append({"score": float(score), **chunk_metadata})
        if len(results) >= top_k:
            break
    logger.info("search: top_k=%d istendi, %d sonuc donduruldu.", top_k, len(results))
    return results
