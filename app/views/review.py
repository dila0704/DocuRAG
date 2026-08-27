"""Inceleme Kuyrugu sayfasi: human_review=True isaretli belgeleri
listeler; duzeltme index'i yeniden kurmadan sadece metadata dosyasini
gunceller (vector_store.update_metadata_by_source_doc + save_metadata)."""
from __future__ import annotations

import classifier
import components
import data_access
import streamlit as st
import vector_store
from styles import inject_global_styles

inject_global_styles()

st.markdown('<div class="disp" style="font-size:28px;">İnceleme Kuyruğu</div>', unsafe_allow_html=True)
st.caption("Güven skoru eşiğin altında kalan belgeleri gözden geçirin ve onaylayın")

index_path = vector_store.load_index_path()
try:
    _, metadata = data_access.get_index(index_path)
except FileNotFoundError:
    st.info("Henüz indekslenmiş belge yok.")
    st.stop()

pending: dict[str, dict] = {}
for m in metadata.values():
    if not m.get("human_review"):
        continue
    doc = m.get("source_doc", "bilinmiyor")
    pending.setdefault(doc, {"siniflar": m.get("siniflar", []), "guven": m.get("guven"), "etiketler": m.get("etiketler", [])})

if not pending:
    st.success("İnceleme bekleyen belge yok.")
    st.stop()

st.caption(f"{len(pending)} belge onay bekliyor")

for doc, info in sorted(pending.items(), key=lambda kv: kv[1]["guven"] if isinstance(kv[1]["guven"], (int, float)) else 0):
    with st.container(border=True):
        cols = st.columns([3, 2, 1, 2])
        cols[0].markdown(f"**{doc}**")
        cols[1].markdown("".join(components.render_category_badge(c) for c in info["siniflar"]) or "—", unsafe_allow_html=True)
        cols[2].markdown(components.render_confidence_stamp(info["guven"]), unsafe_allow_html=True)

        with cols[3].popover("Düzelt", use_container_width=True):
            selected = st.multiselect(
                "Doğru kategori(ler)",
                classifier.DEFAULT_CATEGORIES,
                default=[c for c in info["siniflar"] if c in classifier.DEFAULT_CATEGORIES],
                key=f"cats_{doc}",
            )
            tags_raw = st.text_input(
                "Etiketler (virgülle ayırın)",
                value=", ".join(info["etiketler"]),
                key=f"tags_{doc}",
            )
            if st.button("Onayla", key=f"approve_{doc}", type="primary"):
                updates = {
                    "siniflar": selected or [classifier.FALLBACK_CATEGORY],
                    "etiketler": [t.strip() for t in tags_raw.split(",") if t.strip()],
                    "human_review": False,
                }
                _, current_metadata = vector_store.load_index(index_path)
                updated_metadata = vector_store.update_metadata_by_source_doc(current_metadata, doc, updates)
                vector_store.save_metadata(updated_metadata, index_path)
                st.success(f"{doc} güncellendi.")
                st.rerun()
