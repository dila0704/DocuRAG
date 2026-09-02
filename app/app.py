"""
DocuRAG Streamlit uygulamasi -- giris noktasi.

Calistirma: streamlit run app/app.py

src/ altindaki moduller (ocr, text_splitter, embedder, classifier,
vector_store, llm_factory, pipeline, answer) notebook'larla ayni "duz"
(flat) import konvansiyonuyla yazildi; bu yuzden src/ sys.path'e elle
eklenir (bkz. tests/conftest.py'deki ayni yaklasim).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from styles import inject_global_styles  # noqa: E402

st.set_page_config(
    page_title="DocuRAG",
    page_icon=":material/description:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_styles()

# st.logo, st.navigation'in kendi nav listesinin HER ZAMAN ustune (sidebar'in
# en tepesine) yerlestirdigi tek resmi mekanizma -- elle "with st.sidebar"
# ile eklenen icerik nav listesinin ALTINA duser (Streamlit'in sabit
# yerlesim davranisi), bu yuzden marka logosu icin bu API kullanilir.
st.logo(str(Path(__file__).resolve().parent / "assets" / "logo.svg"), size="large")

pages = [
    st.Page("views/search.py", title="Arama", icon=":material/search:", default=True),
    st.Page("views/upload.py", title="Belge Yükleme", icon=":material/upload_file:"),
    st.Page("views/inventory.py", title="Tüm Belgeler", icon=":material/folder_open:"),
    st.Page("views/review.py", title="İnceleme Kuyruğu", icon=":material/fact_check:"),
    st.Page("views/anomalies.py", title="Anomaliler", icon=":material/warning:"),
    st.Page("views/relationships.py", title="Belge İlişkileri", icon=":material/hub:"),
    st.Page("views/dashboard.py", title="Dashboard", icon=":material/dashboard:"),
    st.Page("views/model_compare.py", title="Model Karşılaştırma", icon=":material/compare_arrows:"),
    st.Page("views/settings.py", title="Ayarlar", icon=":material/tune:"),
]
navigation = st.navigation(pages)

# NOT: sidebar'daki "AKTIF MOD" rozeti kullanici istegiyle kaldirildi --
# aktif mod/model hala Ayarlar sayfasinda goruntulenip degistirilebiliyor.

navigation.run()
