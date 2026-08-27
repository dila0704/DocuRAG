"""Dashboard sayfasi: sadece GERCEK verilerle doldurulan KPI'lar ve
sinif dagilimi. Mockup'taki "Haftalik Islem Hacmi" grafigi ve "Model
Performansi" paneli (gecikme/maliyet) bilerek eklenmedi -- bunlar
zaman damgasi/maliyet izleme altyapisi (DOC-30 Oncelik 2/3: audit_log,
feedback_dataset) gerektiriyor, henuz yok. bkz. plan dosyasi."""
from __future__ import annotations

from collections import Counter

import components
import data_access
import llm_factory
import streamlit as st
import vector_store
from styles import inject_global_styles

inject_global_styles()

st.markdown('<div class="disp" style="font-size:28px;">Dashboard</div>', unsafe_allow_html=True)
st.caption("Sistem durumu ve belge işleme özeti")

index_path = vector_store.load_index_path()
try:
    _, metadata = data_access.get_index(index_path)
except FileNotFoundError:
    st.info("Henüz indekslenmiş belge yok.")
    st.stop()

documents: dict[str, dict] = {}
for m in metadata.values():
    doc = m.get("source_doc", "bilinmiyor")
    if doc not in documents or m.get("ingested_at", "") > documents[doc].get("ingested_at", ""):
        documents[doc] = m

total_docs = len(documents)
total_chunks = len(metadata)
pending_count = sum(1 for m in documents.values() if m.get("human_review"))

llm_settings = llm_factory.load_llm_config()
active_mode = llm_settings.get("active_mode", "cloud")
model_config = llm_settings.get("cloud_model" if active_mode == "cloud" else "local_model", {})

col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam Belge", total_docs)
col2.metric("Toplam Chunk", total_chunks, help=f"Belge başına ortalama {total_chunks / total_docs:.1f}" if total_docs else None)
col3.metric("İnceleme Bekleyen", pending_count)
col4.metric("Aktif LLM Modu", active_mode.capitalize(), model_config.get("model_name", "—"))

st.divider()

left, right = st.columns([1, 1.4])

with left:
    st.markdown('<div class="disp" style="font-size:16px;font-style:italic;">Sınıf Dağılımı</div>', unsafe_allow_html=True)
    counter = Counter()
    for m in documents.values():
        for category in m.get("siniflar", []):
            counter[category] += 1
    st.markdown(components.render_class_distribution_donut(counter), unsafe_allow_html=True)

with right:
    st.markdown('<div class="disp" style="font-size:16px;font-style:italic;">Son İşlemler</div>', unsafe_allow_html=True)
    timestamped = [(doc, m.get("ingested_at")) for doc, m in documents.items() if m.get("ingested_at")]
    timestamped.sort(key=lambda item: item[1], reverse=True)

    if not timestamped:
        st.caption("Henüz zaman damgalı işlem yok (bu belgeler DOC-30 öncesi indekslendi).")
    else:
        for doc, ts in timestamped[:8]:
            info = documents[doc]
            st.markdown(
                f"""
                <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;
                            border-bottom:1px solid var(--border);">
                  <span style="font-size:12.5px;font-weight:600;">{doc}</span>
                  <span class="mono" style="font-size:10.5px;color:var(--text-tertiary);">{components.format_relative_time(ts)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
