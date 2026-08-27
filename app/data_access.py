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

import os

import streamlit as st

import vector_store


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
