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
import llm_factory  # noqa: E402

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
    st.Page("views/review.py", title="İnceleme Kuyruğu", icon=":material/fact_check:"),
    st.Page("views/dashboard.py", title="Dashboard", icon=":material/dashboard:"),
]
navigation = st.navigation(pages)

try:
    _llm_settings = llm_factory.load_llm_config()
    _active_mode = _llm_settings.get("active_mode", "cloud")
    _model_config = _llm_settings.get("cloud_model" if _active_mode == "cloud" else "local_model", {})
    _mode_label = f"{_active_mode.capitalize()} · {_model_config.get('model_name', '?')}"
except Exception:
    _mode_label = "Bilinmiyor"

with st.sidebar:
    st.markdown('<div style="flex-grow:1;min-height:8px;"></div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:8px;padding:8px 10px;background:var(--surface);
                    border:1px solid var(--border);border-radius:8px;margin-top:16px;">
          <span style="width:6px;height:6px;border-radius:50%;background:var(--accent);display:inline-block;"></span>
          <div style="display:flex;flex-direction:column;line-height:1.3;">
            <span class="mono" style="font-size:9px;color:var(--text-tertiary);font-weight:600;letter-spacing:0.05em;">AKTİF MOD</span>
            <span style="font-size:12px;color:var(--text-primary);font-weight:600;">{_mode_label}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

navigation.run()
