"""Tum Belgeler sayfasi: indekslenen her belgenin (thumbnail, kategori,
guven, tarih, chunk sayisi) tam envanterini bir BENTO GRID olarak gosterir --
en son islenen belge tam genislikte bir "hero" kutusu, geri kalani 3 sutunluk
duzenli bir izgara.

Her kutu (delete popover'i, "Sor" page_link'i gibi GERCEK Streamlit
widget'lari icerdigi icin) `st.container(border=True)` ile kuruluyor --
Dashboard'daki gibi tek bir buyuk HTML blob'una donusturulmedi (native
widget'lar oyle bir blob'a guvenilir sekilde gomulemez). Bento hissi,
container'in ICINDEKI renk-bloklu "banner" div'i (negatif margin ile
container kenarlarina tasan, bkz. app/styles.py .doc-bento-banner) ile
veriliyor -- Streamlit'in container/widget DOM modeliyle celismeyen,
saglam bir yaklasim."""
from __future__ import annotations

import html

import components
import data_access
import streamlit as st
import vector_store
from styles import inject_global_styles

inject_global_styles()

st.markdown('<div class="disp" style="font-size:28px;">Tüm Belgeler</div>', unsafe_allow_html=True)
st.caption("İndekslenen tüm belgelerin envanteri")

index_path = vector_store.load_index_path()
try:
    _, metadata = data_access.get_index(index_path)
except FileNotFoundError:
    st.info("Henüz indekslenmiş belge yok.")
    st.stop()

documents = vector_store.group_latest_by_source_doc(metadata)
chunk_counts: dict[str, int] = {}
for m in metadata.values():
    doc = m.get("source_doc", "bilinmiyor")
    chunk_counts[doc] = chunk_counts.get(doc, 0) + 1

if not documents:
    st.info("Henüz indekslenmiş belge yok.")
    st.stop()

ordered = sorted(documents.items(), key=lambda kv: kv[1].get("ingested_at", ""), reverse=True)
st.caption(f"{len(documents)} belge · {len(metadata)} toplam parça")


def _render_tile(doc: str, info: dict, is_hero: bool) -> None:
    with st.container(border=True):
        banner_class = "is-hero" if is_hero else "is-plain"
        st.markdown(
            f"""
            <div class="doc-bento-banner {banner_class}">
              <span class="title">{html.escape(doc)}</span>
              <span class="mono" style="font-size:10px;opacity:0.85;">{components.format_relative_time(info.get('ingested_at'))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        preview_uri = data_access.document_preview_data_uri(doc)
        if is_hero:
            img_col, info_col = st.columns([1, 2])
            if preview_uri:
                img_col.image(preview_uri, width=220)
            with info_col:
                st.markdown("".join(components.render_category_badge(c) for c in info.get("siniflar", [])) or "—", unsafe_allow_html=True)
                st.markdown(components.render_confidence_stamp(info.get("guven")), unsafe_allow_html=True)
                st.caption(f"{chunk_counts.get(doc, 0)} parça")
                action_cols = st.columns([1, 1])
                action_cols[0].page_link("views/search.py", label="Sor", icon=":material/search:", use_container_width=True, query_params={"doc": doc})
                with action_cols[1].popover("Sil", use_container_width=True):
                    confirm_delete = st.checkbox("Eminim, sil.", key=f"inv_confirm_delete_{doc}")
                    if st.button("Sil", key=f"inv_delete_{doc}", disabled=not confirm_delete, type="primary"):
                        _delete(doc, index_path)
        else:
            if preview_uri:
                st.image(preview_uri, width=200)
            st.markdown("".join(components.render_category_badge(c) for c in info.get("siniflar", [])) or "—", unsafe_allow_html=True)
            stamp_col, count_col = st.columns([1, 1])
            stamp_col.markdown(components.render_confidence_stamp(info.get("guven")), unsafe_allow_html=True)
            count_col.caption(f"{chunk_counts.get(doc, 0)} parça")
            action_cols = st.columns([1, 1])
            action_cols[0].page_link("views/search.py", label="Sor", icon=":material/search:", use_container_width=True, query_params={"doc": doc})
            with action_cols[1].popover("Sil", use_container_width=True):
                confirm_delete = st.checkbox("Eminim, sil.", key=f"inv_confirm_delete_{doc}")
                if st.button("Sil", key=f"inv_delete_{doc}", disabled=not confirm_delete, type="primary"):
                    _delete(doc, index_path)


def _delete(doc: str, index_path: str) -> None:
    current_index, current_metadata = vector_store.load_index(index_path)
    updated_index, updated_metadata = vector_store.delete_by_source_doc(current_index, current_metadata, doc)
    vector_store.save_index(updated_index, updated_metadata, index_path)
    st.success(f"{doc} silindi.")
    st.rerun()


hero_doc, hero_info = ordered[0]
_render_tile(hero_doc, hero_info, is_hero=True)

rest = ordered[1:]
if rest:
    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
    for row_start in range(0, len(rest), 3):
        cols = st.columns(3)
        for col, (doc, info) in zip(cols, rest[row_start:row_start + 3]):
            with col:
                _render_tile(doc, info, is_hero=False)
