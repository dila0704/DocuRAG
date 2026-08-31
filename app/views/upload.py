"""Belge Yukleme sayfasi: coklu dosya secip her birini gercek OCR ->
siniflandirma -> alan cikarimi -> embedding -> indeksleme zincirinden
gecirir, st.status() ile canli ilerleme gosterir.

Onceden bu adimlar burada ELLE (pipeline.ingest_document()'tan bagimsiz,
kopya) yazilmisti -- DOC-30 C1'de coklu dosya destegi eklenirken bu kod
tekrari da kapatildi: artik pipeline.ingest_document(on_step=...) callback'i
ile ayni adimlar tek bir yerden (src/pipeline.py) yonetiliyor."""
from __future__ import annotations

import logging
import os

import components
import pipeline
import streamlit as st
from styles import inject_global_styles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

inject_global_styles()

st.markdown('<div class="disp" style="font-size:28px;">Belge Yükleme</div>', unsafe_allow_html=True)
st.caption("OCR, sınıflandırma, alan çıkarımı ve indeksleme otomatik çalışır — birden fazla dosya seçebilirsiniz")

UPLOAD_DIR = os.path.join("data", "raw_docs", "uploads")

uploaded_files = st.file_uploader(
    "Belge görselleri", type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True, label_visibility="collapsed",
)

if uploaded_files:
    st.caption(f"{len(uploaded_files)} dosya seçildi")
    for f in uploaded_files:
        st.image(f, width=140)

    if st.button("Tümünü İşle", type="primary"):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        overall_success = 0

        for uploaded_file in uploaded_files:
            source_doc = uploaded_file.name
            image_path = os.path.join(UPLOAD_DIR, source_doc)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                with st.status(f"{source_doc} işleniyor...", expanded=True) as status:
                    step_labels = {
                        "ocr": lambda info: f"✓ OCR tamamlandı ({info['char_count']} karakter)",
                        "chunking": lambda info: f"✓ {info['chunk_count']} chunk oluşturuldu",
                        "classification": lambda info: f"✓ Sınıflar: {', '.join(info['classification']['siniflar'])} (güven: {info['classification'].get('guven')})",
                        "field_extraction": lambda info: f"✓ Konu: {info['fields'].get('konu') or '—'} · Tutar: {info['fields'].get('tutar') or '—'} · Tarih: {info['fields'].get('tarih') or '—'}",
                        "embedding": lambda info: f"✓ {info['chunk_count']} chunk vektörleştirildi",
                        "indexing": lambda info: "✓ Index güncellendi",
                    }

                    last_info: dict = {}

                    def _on_step(step: str, info: dict, _last_info=last_info) -> None:
                        _last_info[step] = info
                        st.write(step_labels[step](info))
                        if step == "ocr":
                            with st.expander("OCR ham metnini görüntüle"):
                                st.text_area("OCR metni", info["raw_text"], height=150, disabled=True, label_visibility="collapsed", key=f"ocr_{source_doc}")

                    result = pipeline.ingest_document(image_path, on_step=_on_step)
                    status.update(label=f"{source_doc}: Tamamlandı", state="complete", expanded=False)
            except Exception:
                logger.exception("Belge isleme sirasinda hata olustu: %s", source_doc)
                st.error(f"{source_doc} işlenirken bir sorun oluştu, atlanıyor.")
                continue

            overall_success += 1
            classification = result["classification"]
            fields = result["fields"]
            st.markdown(
                f"""
                <div style="background:var(--surface);border-radius:12px;padding:20px;box-shadow:var(--shadow);
                            display:flex;align-items:center;gap:18px;margin-bottom:8px;">
                  {components.render_confidence_stamp(classification.get('guven'))}
                  <div>
                    <div class="mono" style="font-size:9.5px;font-weight:600;color:var(--text-tertiary);
                                letter-spacing:0.07em;text-transform:uppercase;margin-bottom:7px;">{source_doc}</div>
                    <div>{''.join(components.render_category_badge(c) for c in classification['siniflar'])}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"{source_doc} — çıkarılan alanlar"):
                st.json(fields)
            st.page_link(
                "views/search.py",
                label=f"{source_doc} hakkında soru sor",
                icon=":material/search:",
                query_params={"doc": source_doc},
            )

        if overall_success:
            st.success(f"{overall_success}/{len(uploaded_files)} belge başarıyla işlendi.")
