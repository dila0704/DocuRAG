"""Anomaliler sayfasi: field_extractor'in cikardigi yapilandirilmis alanlara
(DOC-30 B1) dayanan, tamamen kod-tabanli (LLM'siz) anomali tespiti (DOC-30 B2)
-- BENTO GRID olarak, "bento-tile-warn-outline" (kirmizi kenarlikli) uyari
kutulariyla. Envanter sayfasindan (pasif katalog) bilerek AYRI: bu sayfa
aksiyoner bir liste ("bunlara bak") sunar."""
from __future__ import annotations

import html

import anomaly
import data_access
import streamlit as st
import vector_store
from styles import inject_global_styles

inject_global_styles()

st.markdown('<div class="disp" style="font-size:28px;">Anomaliler</div>', unsafe_allow_html=True)
st.caption("Tekrarlanan belge numaraları ve tutar aykırı değerleri (kural tabanlı, LLM kullanmaz)")

index_path = vector_store.load_index_path()
try:
    _, metadata = data_access.get_index(index_path)
except FileNotFoundError:
    st.info("Henüz indekslenmiş belge yok.")
    st.stop()

documents = vector_store.group_latest_by_source_doc(metadata)

duplicates = anomaly.find_duplicate_document_numbers(documents)
outliers = anomaly.find_amount_outliers(documents)

if not duplicates and not outliers:
    st.success("Herhangi bir anomali tespit edilmedi.")
    if sum(1 for d in documents.values() if (d.get("alanlar") or {}).get("tutar")) < anomaly.MIN_SAMPLE_SIZE:
        st.caption(f"Not: tutar aykırı değeri tespiti için en az {anomaly.MIN_SAMPLE_SIZE} sayısallaştırılabilir tutar gerekir.")
    st.stop()

st.markdown(
    f"""
    <div class="bento-grid">
      <div class="bento-tile bento-tile-warn-outline bento-span-2">
        <span class="bento-label">Tekrarlanan Belge Numarası</span>
        <span class="bento-number">{len(duplicates)}</span>
        <div class="bento-sub">Aynı belge_no birden fazla belgede geçiyor</div>
      </div>
      <div class="bento-tile bento-tile-warn-outline bento-span-2">
        <span class="bento-label">Tutar Aykırı Değeri</span>
        <span class="bento-number">{len(outliers)}</span>
        <div class="bento-sub">z-skoru ≥ {anomaly.DEFAULT_Z_THRESHOLD}</div>
      </div>
    </div>
    <div style="height:18px;"></div>
    """,
    unsafe_allow_html=True,
)

if duplicates:
    st.markdown('<div class="disp" style="font-size:16px;font-style:italic;margin-bottom:8px;">Tekrarlanan Belge Numaraları</div>', unsafe_allow_html=True)
    for row_start in range(0, len(duplicates), 2):
        cols = st.columns(2)
        for col, dup in zip(cols, duplicates[row_start:row_start + 2]):
            with col:
                doc_links = "".join(
                    f'<span class="mono" style="font-size:11px;">{html.escape(doc)}</span><br/>' for doc in dup["documents"]
                )
                st.markdown(
                    f"""
                    <div class="bento-tile bento-tile-warn-outline">
                      <span class="spotlight-tag">{len(dup['documents'])} belge</span>
                      <span class="bento-label">Belge No</span>
                      <div class="disp" style="font-size:18px;margin-bottom:10px;">{html.escape(dup['belge_no'])}</div>
                      <div>{doc_links}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for doc in dup["documents"]:
                    st.page_link("views/search.py", label=f"{doc} — incele", icon=":material/search:", query_params={"doc": doc})
    st.divider()

if outliers:
    st.markdown('<div class="disp" style="font-size:16px;font-style:italic;margin-bottom:8px;">Tutar Aykırı Değerleri</div>', unsafe_allow_html=True)
    for row_start in range(0, len(outliers), 2):
        cols = st.columns(2)
        for col, outlier in zip(cols, outliers[row_start:row_start + 2]):
            with col:
                st.markdown(
                    f"""
                    <div class="bento-tile bento-tile-warn-outline">
                      <span class="spotlight-tag">z={outlier['z_score']:.2f}</span>
                      <span class="bento-label">{html.escape(outlier['source_doc'])}</span>
                      <span class="bento-number" style="font-size:30px;">{outlier['tutar']:.2f} TL</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.page_link("views/search.py", label="İncele", icon=":material/search:", query_params={"doc": outlier["source_doc"]})
