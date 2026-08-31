"""Belge İlişkileri sayfasi: field_extractor'in cikardigi "taraflar" alanina
gore ortak kisi/kurum paylasan belgeleri bir grafikte gosterir (DOC-30 B3).
LLM kullanmaz -- graph_builder.build_document_graph() tamamen kod-tabanli."""
from __future__ import annotations

import components
import data_access
import graph_builder
import streamlit as st
import vector_store
from styles import inject_global_styles

inject_global_styles()

st.markdown('<div class="disp" style="font-size:28px;">Belge İlişkileri</div>', unsafe_allow_html=True)
st.caption("Ortak taraf (kişi/kurum) paylaşan belgeler arasındaki ilişki grafiği")

index_path = vector_store.load_index_path()
try:
    _, metadata = data_access.get_index(index_path)
except FileNotFoundError:
    st.info("Henüz indekslenmiş belge yok.")
    st.stop()

documents = vector_store.group_latest_by_source_doc(metadata)
graph = graph_builder.build_document_graph(documents)

st.markdown(components.render_document_graph_svg(graph), unsafe_allow_html=True)

if graph.number_of_edges() > 0:
    st.caption(f"{graph.number_of_nodes()} belge, {graph.number_of_edges()} ilişki. Bir çizginin kalınlığı ortak taraf sayısını yansıtır (üzerine gelerek görebilirsiniz).")
