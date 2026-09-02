"""Arama sayfasi: semantik arama + kaynak gosterimli (grounded) AI ozeti.

DOC-30 Oncelik 1: AI ozetindeki her cumle, hangi arama sonucundan
(kaynaktan) geldigini gosteren tiklanabilir [n] linkleriyle isaretlenir;
tiklandiginda ilgili sonuc karti vurgulanir (bkz. .result-card.highlighted,
app/styles.py). Grounding (kaynaklanma) zorunlulugu backend'de
(src/answer.py -> generate_grounded_answer) saglanir.
"""
from __future__ import annotations

import html
import logging
import time
from collections import Counter
from urllib.parse import quote

import streamlit as st

import answer as answer_module
import components
import data_access
import field_extractor
import ocr
import pipeline
import query_rewriter
import vector_store
from styles import inject_global_styles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_document_preview_data_uri = data_access.document_preview_data_uri

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
if "word_boxes_cache" not in st.session_state:
    # source_doc -> ocr.extract_word_boxes() ciktisi (list|None). Tesseract
    # pahali/yavas olabilir; ayni belge icin oturum boyunca tekrar
    # calistirilmaz (bkz. DOC-30 B4, ocr.extract_word_boxes).
    st.session_state["word_boxes_cache"] = {}
if "conversation_history" not in st.session_state:
    # [{"query": str, "answer_summary": str}, ...] -- takip sorularini
    # baglama gore yeniden yazmak (bkz. query_rewriter.condense_conversation)
    # icin kullanilir. Streamlit oturumuna ozeldir (bilinen sinir: farkli bir
    # tarayici sekmesi/kullanici ayri bir gecmis gorur).
    st.session_state["conversation_history"] = []

# Sorgu, kaynak linki tiklamasinin (asagida) tetikledigi tam sayfa
# navigasyonundan sonra da kaybolmasin diye st.query_params'taki "q"
# ile senkron tutulur -- text_input'un varsayilan degeri URL'den okunur.
# key="search-query" -> ".st-key-search-query" CSS class'i (bkz. styles.py):
# resmi Streamlit mekanizmasiyla native input'u ham HTML'e sarmadan restyle
# eder. icon parametresi native bir arama ikonu ekler (ekstra CSS gerekmez).
query = st.text_input(
    "Sorgu",
    value=st.query_params.get("q", ""),
    placeholder="laptop talebi ile ilgili belgeleri bul",
    label_visibility="collapsed",
    icon=":material/search:",
    key="search-query",
)

index_path = vector_store.load_index_path()
try:
    _, all_metadata = data_access.get_index(index_path)
    doc_options = ["Tüm belgeler"] + sorted({m.get("source_doc", "bilinmiyor") for m in all_metadata.values()})
except FileNotFoundError:
    all_metadata = {}
    doc_options = ["Tüm belgeler"]

# key="search-toolbar" -> tum filtre/secenek satirlarini tek bir bento-tile
# gorunumlu "kontrol paneli" kutusuna alir (bkz. styles.py .st-key-search-toolbar).
with st.container(key="search-toolbar"):
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

    option_col, expand_col, clear_col = st.columns([1.4, 1, 1])
    with option_col:
        use_reranker = st.checkbox(
            "Yeniden sırala (cross-encoder)",
            value=False,
            help="İlk hibrit (BM25+dense) sonuçları çok dilli bir cross-encoder modeliyle yeniden sıralar. Daha isabetli ama daha yavaş (model ilk kullanımda indirilir).",
        )
    with expand_col:
        expand_query = st.checkbox(
            "Sorgu genişletme (HyDE)",
            value=False,
            help="Sorguya bir LLM ile varsayımsal bir cevap ürettirip anlam bazlı aramayı bununla zenginleştirir. Belirsiz/kısa sorgularda işe yarar, her aramaya bir LLM çağrısı ekler.",
        )
    with clear_col:
        if st.session_state["conversation_history"] and st.button("Sohbeti temizle", use_container_width=True):
            st.session_state["conversation_history"] = []
            st.rerun()

    compare_mode = st.toggle(
        "Karşılaştırma modu (Temel Hibrit vs Rerank + HyDE)",
        value=False,
        key="compare-mode",
        help=(
            "Aynı sorguyu iki ayrı yapılandırmayla (rerank/HyDE kapalı vs açık) çalıştırıp "
            "sonuçları ve süreleri yan yana gösterir — 'rerank/HyDE gerçekten fark yaratıyor mu' "
            "sorusuna canlı bir cevap. Normal aramadan daha yavaştır (arama iki kez çalışır) ve "
            "AI özeti üretmez."
        ),
    )

if st.session_state["conversation_history"]:
    st.caption(f"💬 {len(st.session_state['conversation_history'])} önceki soru bağlam olarak kullanılıyor (takip sorularını buna göre yorumlar).")

with st.expander("Gelişmiş filtre (tutar / tarih)"):
    afc1, afc2 = st.columns(2)
    with afc1:
        use_amount_filter = st.checkbox("Tutar aralığı", key="use_amount_filter")
        min_amount = st.number_input("Min (TL)", min_value=0.0, value=0.0, step=100.0, disabled=not use_amount_filter)
        max_amount = st.number_input("Max (TL)", min_value=0.0, value=100000.0, step=100.0, disabled=not use_amount_filter)
    with afc2:
        use_date_filter = st.checkbox("Tarih aralığı", key="use_date_filter")
        date_range = st.date_input("Aralık", value=(), disabled=not use_date_filter)
    st.caption("Bu filtreler, belge yüklenirken LLM ile çıkarılan yapılandırılmış alanlara (bkz. Belge Yükleme sayfası) dayanır; alan çıkarılamayan belgeler filtre dışı kalır.")

_extra_filters = []
if use_amount_filter:
    _extra_filters.append(field_extractor.build_amount_range_filter(min_amount, max_amount))
if use_date_filter and isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    _extra_filters.append(field_extractor.build_date_range_filter(date_range[0].isoformat(), date_range[1].isoformat()))

if _extra_filters:
    field_filter = lambda m: all(f(m) for f in _extra_filters)  # noqa: E731
else:
    field_filter = None

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

# Sonuc alani st.empty() ile tek bir placeholder'a alinir: her yeni
# sorguda bu satira ulasilir ulasilmaz onceki calistirmadan kalan icerik
# (eski ozet/sonuc kartlari) aninda temizlenir. Aksi halde arama/LLM
# cagrilari suren birkac saniye boyunca ekranda BIR ONCEKI sorgunun
# cevabi gorunmeye devam eder (Streamlit, script bu satira yeniden
# ulasana kadar eski elemanlari degistirmez).
results_area = st.empty()
with results_area.container():
    if not query:
        # Bos durum: soyut "sorgu yaz" mesaji yerine, indekslenen
        # belgelerden uretilen GERCEK istatistikler (kac belge, kac
        # kategori) ve GERCEK belgelere ait tiklanabilir ornek sorgular
        # gosterilir -- boylece kullanici sayfaya girer girmez pipeline'in
        # (OCR -> siniflandirma -> kaynakli arama) ne urettigini gorur.
        documents = vector_store.group_latest_by_source_doc(all_metadata)

        if not documents:
            st.info("Henüz hiçbir belge indekslenmemiş. Önce **Belge Yükleme** sayfasından bir belge ekleyin.")
            st.stop()

        category_counter = Counter()
        for m in documents.values():
            for category in m.get("siniflar", []):
                category_counter[category] += 1

        example_docs = sorted(documents.values(), key=lambda m: m.get("ingested_at", ""), reverse=True)[:3]
        chip_cards = []
        for m in example_docs:
            label = html.escape(m.get("konu") or m["source_doc"])
            query_value = quote(m.get("konu") or m["source_doc"])
            preview_uri = _document_preview_data_uri(m["source_doc"])
            img_tag = f'<img src="{preview_uri}" alt="">' if preview_uri else ""
            chip_cards.append(
                f'<a class="doc-preview-chip" href="?q={query_value}" target="_self">'
                f'{img_tag}<span class="scrim"></span><span class="lens">🔍</span>'
                f'<span class="label">{label}</span></a>'
            )
        chips_html = "".join(chip_cards)

        # NOT: components.render_class_distribution_donut() kendi coklu
        # satirli HTML'ini (bos satirlarla baslayip biten bir f-string
        # olarak) dondurur. Bunu BASKA bir cok-satirli f-string'in
        # ORTASINA gomersek, aradaki bos satir Markdown'un HTML block
        # algisini erken kapatir ve geri kalan icerik duz metin/kod
        # bloğu olarak sizar. Bu yuzden donut, dashboard.py'deki gibi
        # AYRI ve kendi basina bir st.markdown() cagrisiyla basiliyor.
        left, right = st.columns([2, 1], vertical_alignment="center")
        with left:
            st.markdown(
                f"""
                <div class="bento-tile bento-tile-hero">
                  <span class="bento-label">Sistem hazır</span>
                  <div class="disp" style="font-size:20px;font-style:italic;margin-bottom:8px;color:#FBF9F4;">
                    {len(documents)} belge işlendi, {len(category_counter)} kategoriye ayrıldı
                  </div>
                  <div style="color:rgba(247,244,238,0.82);font-size:13px;line-height:1.65;margin-bottom:18px;">
                    Her belge OCR ile okunuyor, LLM ile sınıflandırılıyor ve anlam bazlı (semantik) aranabilir
                    hale getiriliyor. Gerçek bir belgeyle başlamak için aşağıdakilerden birine tıkla:
                  </div>
                  <div style="display:flex;gap:8px;flex-wrap:wrap;">{chips_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f'<div class="bento-tile"><span class="bento-label">Sınıf Dağılımı</span>'
                f'{components.render_class_distribution_donut(category_counter).strip()}</div>',
                unsafe_allow_html=True,
            )
        st.stop()

    try:
        data_access.get_index(index_path)
    except FileNotFoundError:
        st.warning("Henüz hiçbir belge indekslenmemiş. Önce **Belge Yükleme** sayfasından bir belge ekleyin.")
        st.stop()

    # DOC-35: "en yuksek/en dusuk tutarli fatura" gibi sorgular KARSILASTIRMA
    # gerektirir -- dogru cevap icin TUM belgelerin taranmasi lazim, ama
    # asagidaki normal akis sadece "en alakali top-k parca"yi getirir. LLM bu
    # sinirli baglamla, gormedigi belgeleri hic yokmus gibi "en yuksek" diye
    # sunabilir (yanlis oldugunu SOYLEMEDEN yanlis olabilir). Bunun onune
    # gecmek icin boyle sorgular tespit edilip cevap LLM'e SORULMADAN, TUM
    # metadata uzerinde deterministik hesaplanir (anomaly.py'nin "tam corpus,
    # LLM'siz, kod-dogrulanmis" ilkesiyle AYNI yaklasim) ve normal arama
    # sonuclarinin USTUNE, ayri/acik sekilde etiketlenmis bir banner olarak
    # eklenir -- normal akisi degistirmez/engellemez, sadece tamamlar.
    amount_direction = field_extractor.detect_amount_superlative_query(query)
    if amount_direction is not None:
        category_hint = field_extractor.detect_category_hint(query)
        winner = field_extractor.find_amount_superlative_document(all_metadata, amount_direction, category=category_hint)
        if winner is not None:
            label = "en yüksek" if amount_direction == "max" else "en düşük"
            scope = f"tüm {html.escape(category_hint)} belgeleri" if category_hint else "tüm belgeler"
            taraflar_str = ", ".join(winner["taraflar"]) if winner["taraflar"] else "bilinmiyor"
            konu_suffix = f" ({html.escape(winner['konu'])})" if winner.get("konu") else ""
            belge_no_suffix = f" · Belge No: {html.escape(winner['belge_no'])}" if winner.get("belge_no") else ""
            st.markdown(
                f"""
                <div class="bento-tile bento-tile-hero" style="margin-bottom:16px;">
                  <span class="bento-label">Kesin Sonuç · kod ile hesaplandı, LLM'e sorulmadı</span>
                  <div class="disp" style="font-size:16px;font-style:italic;margin-top:6px;color:#FBF9F4;">
                    {scope.capitalize()} arasında {label} tutar <b>{html.escape(winner['tutar_raw'] or '')}</b> ile
                    <b>{html.escape(winner['source_doc'])}</b>{konu_suffix} belgesinde.
                  </div>
                  <div style="font-size:12px;color:rgba(247,244,238,0.75);margin-top:8px;">
                    Taraflar: {html.escape(taraflar_str)}{belge_no_suffix}
                  </div>
                  <div style="font-size:11px;color:rgba(247,244,238,0.55);margin-top:10px;">
                    Bu sonuç, arama sonucu getirilen az sayıda parça yerine indekslenen TÜM belgeler taranarak
                    hesaplandı — "en yüksek/en düşük" gibi karşılaştırma sorularında LLM'in gördüğü kısıtlı
                    bağlamla yanlış genelleme yapmasını önler.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Tutar karşılaştırması yapmaya çalıştım ama indekslenen belgelerde ayrıştırılabilir bir tutar bulamadım.")

    if compare_mode:
        # DOC-30 sonrasi eklenen "wow" ozelligi: rerank/HyDE'nin GERCEKTEN bir
        # fark yaratip yaratmadigini canli olcup gostermek icin ayni sorgu
        # TAM OLARAK IKI KEZ calistirilir (temel hibrit vs rerank+HyDE).
        # Kaynak gosterimli AI ozeti (ayri bir LLM cagrisi) bilerek burada
        # calistirilmaz -- bu mod SADECE retrieval katmanini karsilastirir,
        # normal arama akisiyla ayni sorguda ic ice calismaz (st.stop() ile
        # asagidaki normal akis atlanir).
        st.markdown("#### Karşılaştırma: Temel Hibrit vs Rerank + HyDE")
        bm25_index = data_access.get_bm25_index(index_path)
        compare_cols = st.columns(2)
        compare_configs = [
            ("Temel Hibrit", {"use_reranker": False, "expand_query": False}),
            ("Rerank + HyDE", {"use_reranker": True, "expand_query": True}),
        ]
        for col, (label, extra_kwargs) in zip(compare_cols, compare_configs):
            with col:
                st.markdown(f"**{label}**")
                t0 = time.perf_counter()
                try:
                    with st.spinner(f"{label} çalışıyor..."):
                        cmp_results = pipeline.search_documents(
                            query, top_k=5, bm25_index=bm25_index, metadata_filter=field_filter, **extra_kwargs,
                        )
                except Exception:
                    logger.exception("Karşılaştırma modu başarısız: %s", label)
                    st.error("Bu yapılandırmada arama başarısız oldu.")
                    continue
                elapsed = time.perf_counter() - t0
                st.caption(f"⏱ {elapsed:.2f} sn · {len(cmp_results)} sonuç")
                if not cmp_results:
                    st.caption("Sonuç bulunamadı.")
                for rank, r in enumerate(cmp_results[:5], start=1):
                    if r.get("score_type") == "rrf":
                        score_label = f"RRF {r['score']:.3f}"
                    else:
                        score_label = f"%{round(r.get('score', 0.0) * 100)}"
                    if r.get("rerank_score") is not None:
                        score_label += f" · rerank {r['rerank_score']:.3f}"
                    st.markdown(f"{rank}. **{html.escape(r.get('source_doc', 'bilinmiyor'))}** · `{score_label}`")
                    st.caption((r.get("text", "") or "")[:140])
        st.divider()
        st.caption("Karşılaştırma modunu kapatarak normal aramaya (AI özeti dahil) dönebilirsiniz.")
        st.stop()

    try:
        with st.spinner("Aranıyor..."):
            bm25_index = data_access.get_bm25_index(index_path)

            # Cok-turlu arama: onceki sorular varsa, takip sorusunu
            # (orn. "peki tarihi neydi?") bagimsiz/tek basina anlasilir bir
            # sorguya yogunlastirir. Kullaniciya gosterilen metin (query)
            # DEGISMEZ, sadece arka plandaki arama/cevap uretimi bu
            # yogunlastirilmis sorguyu kullanir.
            search_query = query
            if st.session_state["conversation_history"]:
                try:
                    search_query = query_rewriter.condense_conversation(
                        st.session_state["conversation_history"], query,
                    )
                except Exception:
                    logger.exception("condense_conversation basarisiz, orijinal sorgu kullanilacak: %r", query)

            if scoped_to_doc:
                # Belgeye sinirlandiginda FAISS'in genel top-k'sina degil, o
                # belgenin TUM parcalarina ihtiyacimiz var -- vector_store.
                # search()'un metadata_filter'i (DOC-30 B1) ile resmilesmis
                # "genis top-k + filtrele" deseni (eskiden burada elle
                # yazilmis bir liste comprehension'du).
                doc_filter = lambda m: m.get("source_doc") == scoped_to_doc  # noqa: E731
                combined_filter = (lambda m: doc_filter(m) and field_filter(m)) if field_filter else doc_filter
                results = pipeline.search_documents(
                    search_query, top_k=ALL_CHUNKS_TOP_K, bm25_index=bm25_index, metadata_filter=combined_filter,
                )
            else:
                results = pipeline.search_documents(
                    search_query, top_k=top_k, bm25_index=bm25_index,
                    use_reranker=use_reranker, expand_query=expand_query, metadata_filter=field_filter,
                )
    except Exception:
        logger.exception("search_documents basarisiz: sorgu=%r", query)
        st.error("Arama sırasında bir sorun oluştu. Lütfen daha sonra tekrar deneyin.")
        st.stop()

    if not results:
        st.info("Sonuç bulunamadı.")
        st.stop()

    for i, chunk in enumerate(results, start=1):
        chunk["_citation_index"] = i

    if scoped_to_doc:
        st.caption(f"📄 **{scoped_to_doc}** belgesinin tamamı ({len(results)} parça) kullanılarak yanıtlanıyor.")

    answer_error = False
    try:
        with st.spinner("Özet oluşturuluyor..."):
            grounded = answer_module.generate_grounded_answer(search_query, results)
    except Exception:
        logger.exception("generate_grounded_answer basarisiz: sorgu=%r", query)
        grounded = {"grounded": False, "sentences": []}
        answer_error = True
        st.warning("Özet oluşturulurken bir sorun oluştu; aşağıdaki sonuçları inceleyebilirsiniz.")

    if grounded["grounded"]:
        # Bir kaynak [n] linkine tiklamak da bu sayfayi (ayni sorguyla) yeniden
        # calistirir -- ayni soru ust uste tekrar eklenmesin diye sadece
        # gecmisteki SON kayittan farkliysa eklenir (aynı soruyu farkli bir
        # zamanda TEKRAR sormak hala mumkun, sadece ardisik tekrar engellenir).
        history = st.session_state["conversation_history"]
        if not history or history[-1]["query"] != query:
            answer_summary = " ".join(s["text"] for s in grounded["sentences"])[:200]
            history.append({"query": query, "answer_summary": answer_summary})

        query_param = quote(query)
        sentence_parts = []
        for sentence in grounded["sentences"]:
            text = html.escape(sentence["text"])
            links = " ".join(
                f'<a class="citation-link" href="?q={query_param}&cite={s}" target="_self">[{s}]</a>'
                for s in sentence["sources"]
            )
            sentence_parts.append(f"{text} {links}")
        # DOC-30 C2: gercek token-stream DEGIL -- backend zaten TAM JSON'i
        # uretip _enforce_grounding ile dogruladiktan SONRA burasi calisir
        # (grounding garantisi risk edilmiyor). Sadece okuma deneyimini
        # kademeli/"canli yaziliyor" gibi hissettirir, gercek gecikme
        # kazanci YOKTUR (bkz. components.stream_sentences docstring'i).
        def _wrap_answer_card(body_html: str) -> str:
            return f"""
            <div class="bento-tile bento-tile-bronze">
              <div style="display:flex;align-items:center;gap:7px;margin-bottom:8px;">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="#FBF9F4"><path d="M12 2l1.8 6.2L20 10l-6.2 1.8L12 18l-1.8-6.2L4 10l6.2-1.8L12 2z"/></svg>
                <span class="mono" style="font-size:9.5px;font-weight:700;letter-spacing:0.09em;color:#FBF9F4;text-transform:uppercase;">Özet</span>
              </div>
              <div class="disp" style="font-size:16px;font-style:italic;line-height:1.65;color:#FBF9F4;">{body_html}</div>
            </div>
            """

        answer_placeholder = st.empty()
        components.stream_sentences(sentence_parts, answer_placeholder, wrap=_wrap_answer_card)
    elif not answer_error:
        st.caption("Kaynaklardan güvenle özetlenebilecek bir cevap oluşturulamadı; aşağıdaki sonuçları inceleyin.")

    groups = components.group_results_by_document(results)
    cap_col, dl_col = st.columns([4, 1])
    cap_col.caption(f"{len(groups)} belge · {len(results)} eşleşen parça")
    csv_bytes = components.search_results_to_dataframe(results).to_csv(index=False).encode("utf-8-sig")
    dl_col.download_button("CSV indir", csv_bytes, "arama_sonuclari.csv", "text/csv", use_container_width=True)

    # Bento grid: en alakali sonuc (groups zaten best_score'a gore azalan
    # siralı, bkz. components.group_results_by_document) tam genislikte bir
    # "spotlight" kutusu, geri kalani 2 sutunlu duzenli bir izgara.
    def _render_result_card(group: dict, hero: bool) -> None:
        citation_indices = {c["_citation_index"] for c in group["chunks"]}
        is_highlighted = st.session_state["highlighted_source"] in citation_indices
        classes = ["bento-tile", "result-card"]
        if is_highlighted:
            classes.append("highlighted")
        if hero:
            classes.append("bento-tile-spotlight")
        card_class = " ".join(classes)

        badges = "".join(components.render_category_badge(c) for c in group["siniflar"])
        stamp = components.render_confidence_stamp(group["guven"])
        best_chunk = max(group["chunks"], key=lambda c: c.get("score", 0.0))
        preview = components.highlight_terms(best_chunk["text"][:420 if hero else 200], query)

        chunk_scores = " · ".join(
            components.format_chunk_score(c) for c in sorted(group["chunks"], key=lambda c: c["_citation_index"])
        )
        spotlight_tag = '<span class="spotlight-tag">En iyi eşleşme</span>' if hero else ""

        # NOT: {spotlight_tag} kasitli olarak {card_class} ile AYNI satirda --
        # hero=False iken bos string oldugu icin kendi satirinda birakilirsa
        # TAMAMEN BOS bir satir olusur, bu da Markdown'in HTML blok algisini
        # erken kapatir ve geri kalan tum <div>'ler duz metin/kod blogu olarak
        # sizar (render_class_distribution_donut()'taki .strip() ile AYNI
        # kok neden, bkz. app/views/dashboard.py). Gercek uygulamada (grid
        # kartlarinda) gozlenip duzeltildi.
        st.markdown(
            f"""
            <div class="{card_class}">{spotlight_tag}
              <div style="display:flex;align-items:center;justify-content:space-between;gap:16px;">
                <div>
                  <div style="font-weight:600;font-size:{'16px' if hero else '14px'};">{html.escape(group['source_doc'])}</div>
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

        # "Nasil bulundu?" seffaflik paneli: retrieval'in ic skorlarini
        # (RRF fusion oncesi ham dense/BM25 skorlari + varsa rerank skoru)
        # kullaniciya gosterir -- hibrit aramanin "black box" hissini kirmak
        # icin (bkz. retrieval.hybrid_search dense_score/bm25_score alanlari).
        score_rows = []
        for chunk in sorted(group["chunks"], key=lambda c: c["_citation_index"]):
            parts = [f"**[{chunk['_citation_index']}]**"]
            if chunk.get("dense_score") is not None:
                parts.append(f"dense (cosine) {chunk['dense_score']:.3f}")
            if chunk.get("bm25_score") is not None:
                parts.append(f"BM25 {chunk['bm25_score']:.3f}")
            score = chunk.get("score")
            if score is not None:
                label = "RRF" if chunk.get("score_type") == "rrf" else "cosine"
                parts.append(f"{label} {score:.3f}")
            if chunk.get("rerank_score") is not None:
                parts.append(f"rerank {chunk['rerank_score']:.3f}")
            score_rows.append(" · ".join(parts))

        if score_rows:
            with st.expander("Nasıl bulundu?"):
                for row in score_rows:
                    st.caption(row)

        if is_highlighted:
            image_path = data_access.document_image_path(group["source_doc"])
            if image_path is not None:
                cache = st.session_state["word_boxes_cache"]
                if group["source_doc"] not in cache:
                    with st.spinner("Kaynağın belge üzerindeki konumu tespit ediliyor..."):
                        cache[group["source_doc"]] = ocr.extract_word_boxes(str(image_path))
                word_boxes = cache[group["source_doc"]]
                bbox = ocr.locate_chunk_bbox(best_chunk["text"], word_boxes) if word_boxes else None
                image_bytes = ocr.render_highlighted_image(str(image_path), bbox)
                caption = (
                    "Kaynağın belge üzerindeki yaklaşık konumu"
                    if bbox is not None
                    else "Bu kaynağın belge üzerindeki konumu otomatik tespit edilemedi."
                )
                st.image(image_bytes, caption=caption, width=320)

    _render_result_card(groups[0], hero=True)
    rest_groups = groups[1:]
    for row_start in range(0, len(rest_groups), 2):
        cols = st.columns(2)
        for col, group in zip(cols, rest_groups[row_start:row_start + 2]):
            with col:
                _render_result_card(group, hero=False)
