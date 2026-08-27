"""Belge Yukleme sayfasi: gercek OCR -> siniflandirma -> embedding ->
indeksleme zincirini adim adim calistirip st.status() ile canli ilerleme
gosterir (mockup'taki stepper'in idiomatic Streamlit karsiligi)."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import streamlit as st

import classifier
import components
import embedder
import ocr
import text_splitter
import vector_store
from styles import inject_global_styles

inject_global_styles()

st.markdown('<div class="disp" style="font-size:28px;">Belge Yükleme</div>', unsafe_allow_html=True)
st.caption("OCR, sınıflandırma ve indeksleme otomatik çalışır")

UPLOAD_DIR = os.path.join("data", "raw_docs", "uploads")

uploaded_file = st.file_uploader(
    "Belge görseli", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed"
)

if uploaded_file is not None:
    st.image(uploaded_file, width=280)

    if st.button("İşle", type="primary"):
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        image_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        source_doc = uploaded_file.name

        with st.status("Belge işleniyor...", expanded=True) as status:
            st.write("**OCR** — metin çıkarılıyor...")
            raw_text = ocr.extract_text_from_image(image_path)
            st.write(f"✓ OCR tamamlandı ({len(raw_text)} karakter)")

            st.write("**Chunking**")
            chunks = text_splitter.split_text(raw_text)
            if not chunks:
                status.update(label="Başarısız: OCR sonucu boş", state="error")
                st.error("Görselden anlamlı metin çıkarılamadı.")
                st.stop()
            for chunk in chunks:
                chunk["source_doc"] = source_doc
            st.write(f"✓ {len(chunks)} chunk oluşturuldu")

            st.write("**Sınıflandırma** — LLM ile kategori/etiket çıkarılıyor...")
            classification = classifier.classify_chunks(chunks)
            now_iso = datetime.now(timezone.utc).isoformat()
            labeled_chunks = classifier.attach_labels_to_chunks(chunks, {source_doc: classification})
            for chunk in labeled_chunks:
                chunk["ingested_at"] = now_iso
            st.write(f"✓ Sınıflar: {', '.join(classification['siniflar'])} (güven: {classification.get('guven')})")

            st.write("**Embedding**")
            embedded_chunks = embedder.embed_chunks(labeled_chunks)
            st.write(f"✓ {len(embedded_chunks)} chunk vektörleştirildi")

            st.write("**İndeksleme**")
            index_path = vector_store.load_index_path()
            if os.path.exists(index_path + ".faiss"):
                index, metadata = vector_store.load_index(index_path)
                index, metadata = vector_store.add_chunks(index, metadata, embedded_chunks)
            else:
                index, metadata = vector_store.build_index(embedded_chunks)
            vector_store.save_index(index, metadata, index_path)
            st.write("✓ Index güncellendi")

            status.update(label="Tamamlandı", state="complete", expanded=False)

        st.markdown(
            f"""
            <div style="background:var(--surface);border-radius:12px;padding:20px;box-shadow:var(--shadow);
                        display:flex;align-items:center;gap:18px;">
              {components.render_confidence_stamp(classification.get('guven'))}
              <div>
                <div class="mono" style="font-size:9.5px;font-weight:600;color:var(--text-tertiary);
                            letter-spacing:0.07em;text-transform:uppercase;margin-bottom:7px;">Sınıflandırma Sonucu</div>
                <div>{''.join(components.render_category_badge(c) for c in classification['siniflar'])}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
