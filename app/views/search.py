"""Arama sayfasi: semantik arama + kaynak gosterimli (grounded) AI ozeti.

DOC-30 Oncelik 1: AI ozetindeki her cumle, hangi arama sonucundan
(kaynaktan) geldigini gosteren tiklanabilir [n] linkleriyle isaretlenir;
tiklandiginda ilgili sonuc karti vurgulanir (bkz. .result-card.highlighted,
app/styles.py). Grounding (kaynaklanma) zorunlulugu backend'de
(src/answer.py -> generate_grounded_answer) saglanir.
"""
from __future__ import annotations

import html
from urllib.parse import quote

import streamlit as st

import answer as answer_module
import components
import data_access
import pipeline
import vector_store
from styles import inject_global_styles

# Belirli bir belgeye sinirlanan sorularda, o belgenin TUM chunk'lari
# (sadece FAISS'in genel top-k'si degil) cevaba dahil edilir -- "yuzeysel"
# degil, o belge hakkindaki her detayi kapsayan bir cevap hedeflenir. Bu
# ust sinir, pratikte hicbir test belgesinin asamayacagi kadar buyuk.
ALL_CHUNKS_TOP_K = 1000

inject_global_styles()

st.markdown('<div class="disp" style="font-size:28px;">Arama</div>', unsafe_allow_html=True)
st.caption("İndekslenen belgelerde doğal dil ile semantik arama yapın")

if "highlighted_source" not in st.session_state:
    st.session_state["highlighted_source"] = None
if "last_query" not in st.session_state:
    st.session_state["last_query"] = None

# Sorgu, kaynak linki tiklamasinin (asagida) tetikledigi tam sayfa
# navigasyonundan sonra da kaybolmasin diye st.query_params'taki "q"
# ile senkron tutulur -- text_input'un varsayilan degeri URL'den okunur.
query = st.text_input(
    "Sorgu",
    value=st.query_params.get("q", ""),
    placeholder="laptop talebi ile ilgili belgeleri bul",
    label_visibility="collapsed",
)

index_path = vector_store.load_index_path()
try:
    _, all_metadata = data_access.get_index(index_path)
    doc_options = ["Tüm belgeler"] + sorted({m.get("source_doc", "bilinmiyor") for m in all_metadata.values()})
except FileNotFoundError:
    doc_options = ["Tüm belgeler"]

filter_col, slider_col = st.columns([1.4, 1])
with filter_col:
    doc_filter = st.selectbox(
        "Belgeye göre sınırla",
        doc_options,
        index=doc_options.index(st.query_params["doc"]) if st.query_params.get("doc") in doc_options else 0,
        help="Belirli bir belge seçilirse cevap SADECE o belgenin tüm parçaları kullanılarak (genel en-alakalı-5 değil, o belgenin tamamı) oluşturulur.",
    )
scoped_to_doc = doc_filter if doc_filter != "Tüm belgeler" else None
with slider_col:
    top_k = st.slider("Sonuç sayısı", min_value=1, max_value=10, value=5, disabled=scoped_to_doc is not None)

if scoped_to_doc:
    st.query_params["doc"] = scoped_to_doc
elif "doc" in st.query_params:
    del st.query_params["doc"]

if query != st.session_state["last_query"]:
    st.session_state["highlighted_source"] = None
    st.session_state["last_query"] = query

if query:
    st.query_params["q"] = query
elif "q" in st.query_params:
    del st.query_params["q"]

clicked = st.query_params.get("cite")
if clicked is not None:
    try:
        st.session_state["highlighted_source"] = int(clicked)
    except ValueError:
        pass

if not query:
    st.info("Aramaya başlamak için yukarıya bir sorgu yazın.")
    st.stop()

try:
    data_access.get_index(index_path)
except FileNotFoundError:
    st.warning("Henüz hiçbir belge indekslenmemiş. Önce **Belge Yükleme** sayfasından bir belge ekleyin.")
    st.stop()

with st.spinner("Aranıyor..."):
    if scoped_to_doc:
        # Belgeye sinirlandiginda FAISS'in genel top-k'sina degil, o
        # belgenin TUM parcalarina ihtiyacimiz var -- once genis bir
        # top-k ile tum siralamayi al, sonra sadece secili belgeye ait
        # olanlari (gercek relevance skorlariyla) filtrele.
        all_ranked = pipeline.search_documents(query, top_k=ALL_CHUNKS_TOP_K)
        results = [c for c in all_ranked if c.get("source_doc") == scoped_to_doc]
    else:
        results = pipeline.search_documents(query, top_k=top_k)

if not results:
    st.info("Sonuç bulunamadı.")
    st.stop()

for i, chunk in enumerate(results, start=1):
    chunk["_citation_index"] = i

if scoped_to_doc:
    st.caption(f"📄 **{scoped_to_doc}** belgesinin tamamı ({len(results)} parça) kullanılarak yanıtlanıyor.")

with st.spinner("Özet oluşturuluyor..."):
    grounded = answer_module.generate_grounded_answer(query, results)

if grounded["grounded"]:
    query_param = quote(query)
    sentence_parts = []
    for sentence in grounded["sentences"]:
        text = html.escape(sentence["text"])
        links = " ".join(
            f'<a class="citation-link" href="?q={query_param}&cite={s}" target="_self">[{s}]</a>'
            for s in sentence["sources"]
        )
        sentence_parts.append(f"{text} {links}")
    summary_html = " ".join(sentence_parts)

    st.markdown(
        f"""
        <div class="answer-card">
          <div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="#8B6F47"><path d="M12 2l1.8 6.2L20 10l-6.2 1.8L12 18l-1.8-6.2L4 10l6.2-1.8L12 2z"/></svg>
            <span class="mono" style="font-size:9.5px;font-weight:700;letter-spacing:0.09em;color:var(--bronze);text-transform:uppercase;">Özet</span>
          </div>
          <div class="disp" style="font-size:16px;font-style:italic;line-height:1.65;">{summary_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.caption("Kaynaklardan güvenle özetlenebilecek bir cevap oluşturulamadı; aşağıdaki sonuçları inceleyin.")

groups = components.group_results_by_document(results)
st.caption(f"{len(groups)} belge · {len(results)} eşleşen parça")

for group in groups:
    citation_indices = {c["_citation_index"] for c in group["chunks"]}
    is_highlighted = st.session_state["highlighted_source"] in citation_indices
    card_class = "result-card highlighted" if is_highlighted else "result-card"

    badges = "".join(components.render_category_badge(c) for c in group["siniflar"])
    stamp = components.render_confidence_stamp(group["guven"])
    best_chunk = max(group["chunks"], key=lambda c: c.get("score", 0.0))
    preview = components.highlight_terms(best_chunk["text"][:280], query)

    chunk_scores = " · ".join(
        f'[{c["_citation_index"]}] %{round(c.get("score", 0.0) * 100)}' for c in sorted(group["chunks"], key=lambda c: c["_citation_index"])
    )

    st.markdown(
        f"""
        <div class="{card_class}">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:16px;">
            <div>
              <div style="font-weight:600;font-size:14px;">{html.escape(group['source_doc'])}</div>
              <div class="mono" style="font-size:10.5px;color:var(--text-tertiary);margin-top:1px;">{html.escape(chunk_scores)}</div>
            </div>
            {stamp}
            <div>{badges}</div>
          </div>
          <div style="font-size:12.5px;color:var(--text-secondary);line-height:1.65;margin-top:10px;">"{preview}"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
