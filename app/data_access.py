"""
FAISS index/metadata erisimini Streamlit rerun'lari arasinda onbellekleyen
katman.

Streamlit, bir widget'la her etkilesimde (slider, secim kutusu, metin
girisi...) TUM script'i yeniden calistirir. vector_store.load_index()
her seferinde diskten okuyup deserialize ediyor -- bu, sayfa gercekte
degismemisken bile gozle gorulur bir yavasliga yol aciyordu. Bu modul,
index/metadata ciftini dosyalarin degisim zamanina (mtime) gore
onbellekler: dosyalar degismedigi surece ayni nesne tekrar kullanilir,
degistiginde (yeni ingest, inceleme duzeltmesi) otomatik yeniden yuklenir.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path

import pandas as pd
import streamlit as st

import llm_factory
import retrieval
import vector_store

# Bir source_doc'un orijinal gorseli hem data/raw_docs/uploads/ (yeni
# yuklenenler) hem de data/raw_docs/ (baslangic seed belgeleri) altinda
# olabilir -- ikisi de kontrol edilir. search.py/review.py/inventory.py
# arasinda paylasilan tek kaynak (eskiden search.py'de kopya olarak vardi).
_DOC_IMAGE_DIRS = (Path("data/raw_docs/uploads"), Path("data/raw_docs"))


def document_image_path(source_doc: str) -> Path | None:
    """document_preview_data_uri ile ayni arama mantigi, ama base64 yerine
    dosya YOLUNU dondurur -- ocr.extract_word_boxes()/render_highlighted_image()
    gibi dosya yolu bekleyen fonksiyonlar icin (bkz. DOC-30 B4)."""
    for directory in _DOC_IMAGE_DIRS:
        candidate = directory / source_doc
        if candidate.is_file():
            return candidate
    return None


@st.cache_data(show_spinner=False)
def _read_image_data_uri_cached(path_str: str, mtime: float) -> str:
    path = Path(path_str)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def document_preview_data_uri(source_doc: str) -> str | None:
    """Streamlit HER widget etkilesiminde TUM script'i yeniden calistirir
    -- bu fonksiyon onbelleksiz haldeyken, orn. Tum Belgeler sayfasindaki
    her belge kucuk resmi HER rerun'da diskten okunup yeniden base64'e
    cevriliyordu (gozle gorulur yavaslik kaynagiydi, DOC-30 performans
    duzeltmesi). mtime-tabanli onbellek get_index()/get_bm25_index() ile
    ayni desen: dosya degismedigi surece yeniden okunmaz."""
    path = document_image_path(source_doc)
    if path is None:
        return None
    return _read_image_data_uri_cached(str(path), path.stat().st_mtime)


@st.cache_resource(show_spinner="İndeks yükleniyor...")
def _load_index_cached(index_path: str, faiss_mtime: float, meta_mtime: float):
    return vector_store.load_index(index_path)


def get_index(index_path: str | None = None):
    """vector_store.load_index() ile ayni sozlesme (ayni donus degeri,
    ayni FileNotFoundError), ama dosyalar degismedigi surece onbellekten
    doner.

    Raises:
        FileNotFoundError: index dosyalari yoksa.
    """
    index_path = index_path or vector_store.load_index_path()
    faiss_mtime = os.path.getmtime(index_path + ".faiss")
    meta_mtime = os.path.getmtime(index_path + ".meta.json")
    return _load_index_cached(index_path, faiss_mtime, meta_mtime)


@st.cache_resource(show_spinner="Anahtar kelime indeksi kuruluyor...")
def _build_bm25_index_cached(index_path: str, meta_mtime: float):
    _, metadata = get_index(index_path)
    return retrieval.build_bm25_index(metadata)


def get_bm25_index(index_path: str | None = None):
    """retrieval.build_bm25_index()'in mtime-tabanli onbellekli hali --
    get_index() ile ayni desen (metadata dosyasi degismedigi surece BM25
    indeksi yeniden kurulmaz)."""
    index_path = index_path or vector_store.load_index_path()
    meta_mtime = os.path.getmtime(index_path + ".meta.json")
    return _build_bm25_index_cached(index_path, meta_mtime)


@st.cache_data(show_spinner=False)
def _read_usage_log_cached(path: str, mtime: float) -> pd.DataFrame:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def load_usage_log(path: str | None = None) -> pd.DataFrame:
    """llm_factory._append_usage_log()'un yazdigi data/processed/usage_log.jsonl'i
    okur (bkz. Dashboard'daki maliyet/gecikme paneli).

    Dosya henuz yoksa (hic LLM cagrisi yapilmamis) BOS bir DataFrame doner --
    bu bir hata degil, "henuz veri yok" durumu oldugu icin get_index()'in
    aksine FileNotFoundError firlatilmaz. Dosya buyudukce (her LLM cagrisinda
    bir satir) her rerun'da tamamini yeniden okumamak icin mtime-tabanli
    onbellek kullanilir (get_index() ile ayni desen).
    """
    path = path or str(llm_factory.DEFAULT_USAGE_LOG_PATH)
    if not os.path.exists(path):
        return pd.DataFrame(columns=["timestamp", "provider", "model_name", "input_tokens", "output_tokens", "cost_usd", "duration_s"])
    return _read_usage_log_cached(path, os.path.getmtime(path))
