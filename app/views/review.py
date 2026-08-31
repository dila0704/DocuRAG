"""Inceleme Kuyrugu sayfasi: human_review=True isaretli belgeleri BENTO GRID
olarak listeler (en dusuk guvenli/en acil belge tam genislikte bir "hero"
kutusu, geri kalani duzenli bir izgara -- app/views/inventory.py'deki ayni
desen); duzeltme index'i yeniden kurmadan sadece metadata dosyasini gunceller
(vector_store.update_metadata_by_source_doc + save_metadata).

Reviewer'in belgenin GORSELINI ve OCR HAM METNINI gormeden kategori
onaylaması, insan-dongude-inceleme adiminin amacini zayiflatiyordu -- ikisi
de artik "Belge" popover'inde gosteriliyor (gorsel: data_access.
document_preview_data_uri, ayni kaynak search.py ile paylasilir; OCR metni:
bu belgeye ait chunk'lar chunk_id sirasina gore birlestirilir, ayri bir sema
degisikligi gerekmez)."""
from __future__ import annotations

import html

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
    entry = pending.setdefault(doc, {
        "siniflar": m.get("siniflar", []),
        "guven": m.get("guven"),
        "etiketler": m.get("etiketler", []),
        "chunks": [],
    })
    entry["chunks"].append(m)

if not pending:
    st.success("İnceleme bekleyen belge yok.")
    st.stop()

st.caption(f"{len(pending)} belge onay bekliyor")

ordered = sorted(pending.items(), key=lambda kv: kv[1]["guven"] if isinstance(kv[1]["guven"], (int, float)) else 0)


def _apply_update(doc: str, updates: dict) -> None:
    _, current_metadata = vector_store.load_index(index_path)
    updated_metadata = vector_store.update_metadata_by_source_doc(current_metadata, doc, updates)
    vector_store.save_metadata(updated_metadata, index_path)
    st.success(f"{doc} güncellendi.")
    st.rerun()


def _delete(doc: str) -> None:
    current_index, current_metadata = vector_store.load_index(index_path)
    updated_index, updated_metadata = vector_store.delete_by_source_doc(current_index, current_metadata, doc)
    vector_store.save_index(updated_index, updated_metadata, index_path)
    st.success(f"{doc} silindi.")
    st.rerun()


def _render_tile(doc: str, info: dict, is_hero: bool) -> None:
    with st.container(border=True):
        banner_class = "is-hero" if is_hero else "is-plain"
        st.markdown(
            f"""
            <div class="doc-bento-banner {banner_class}">
              <span class="title">{html.escape(doc)}</span>
              <span class="mono" style="font-size:10px;opacity:0.85;">{'⚠ İNCELEME BEKLİYOR' if is_hero else ''}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        top_cols = st.columns([2, 1])
        top_cols[0].markdown("".join(components.render_category_badge(c) for c in info["siniflar"]) or "—", unsafe_allow_html=True)
        top_cols[1].markdown(components.render_confidence_stamp(info["guven"]), unsafe_allow_html=True)

        action_cols = st.columns(2)
        with action_cols[0].popover("Belge", use_container_width=True):
            preview_uri = data_access.document_preview_data_uri(doc)
            if preview_uri:
                st.image(preview_uri, width=260)
            else:
                st.caption("Görsel bulunamadı.")

            ordered_chunks = sorted(info["chunks"], key=lambda c: c.get("chunk_id", 0))
            raw_text = "\n".join(c.get("text", "") for c in ordered_chunks)
            st.text_area("OCR metni", raw_text, height=180, disabled=True, key=f"ocr_{doc}")

        with action_cols[1].popover("Düzelt", use_container_width=True):
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
                _apply_update(doc, {
                    "siniflar": selected or [classifier.FALLBACK_CATEGORY],
                    "etiketler": [t.strip() for t in tags_raw.split(",") if t.strip()],
                    "human_review": False,
                })

            st.divider()
            st.caption("Belgeyi tamamen sil")
            confirm_delete = st.checkbox("Eminim, bu belgeyi ve tüm parçalarını sil.", key=f"confirm_delete_{doc}")
            if st.button("Sil", key=f"delete_{doc}", disabled=not confirm_delete):
                _delete(doc)


hero_doc, hero_info = ordered[0]
_render_tile(hero_doc, hero_info, is_hero=True)

rest = ordered[1:]
if rest:
    st.markdown('<div style="height:4px;"></div>', unsafe_allow_html=True)
    for row_start in range(0, len(rest), 2):
        cols = st.columns(2)
        for col, (doc, info) in zip(cols, rest[row_start:row_start + 2]):
            with col:
                _render_tile(doc, info, is_hero=False)
