"""Model Karşılaştırma sayfası: aynı belge metnini cloud (Anthropic/OpenAI,
config/settings.yaml -> cloud_model) ve local (huggingface, küçük bir açık
model) ile AYNI ANDA sınıflandırıp yan yana gösterir.

Neden config'teki varsayılan local model (meta-llama/Meta-Llama-3-8B-Instruct)
DEĞİL: o model gated (Meta onayı + .env'de HF_TOKEN gerektirir) ve bu ortamda
erişilemiyor (bkz. README "Devam eden" notları, notebooks/10). Bunun yerine
notebooks/12_multi_model_e2e_chain_test.ipynb'de zaten kurulmuş/belgelenmiş
"vekil model" pratiğiyle aynı şekilde, herkese açık ve CPU'da makul hızda
çalışan Qwen/Qwen2.5-0.5B-Instruct kullanılır. Kullanıcının Ayarlar
sayfasından seçtiği global active_mode/local_model'e HİÇ dokunulmaz -- bu
sayfa kendi sabit karşılaştırma istemcilerini kurar.

Local model ilk kullanımda indirilir (birkaç dakika sürebilir) ve
llm_factory._get_local_model()'in module-seviyesi önbelleğinde tutulur
(sunucu süreci ayakta kaldığı sürece tekrar indirilmez). İndirme/inference
başarısız olursa (ağ, bellek, vb.) sayfa ÇÖKMEZ -- cloud sonucu yine
gösterilir, local taraf için sadece bir hata mesajı gösterilir (bkz.
ocr.extract_word_boxes'daki "sessizce devre dışı kal" desenine benzer bir
tolerans)."""
from __future__ import annotations

import html
import logging
import time

import classifier
import components
import data_access
import llm_factory
import streamlit as st
import vector_store
from styles import inject_global_styles

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

LOCAL_COMPARE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

inject_global_styles()

st.markdown('<div class="disp" style="font-size:28px;">Model Karşılaştırma</div>', unsafe_allow_html=True)
st.caption("Aynı belge metnini cloud ve local bir modelle aynı anda sınıflandırıp doğruluk/hız farkını yan yana gösterir")

index_path = vector_store.load_index_path()
try:
    _, all_metadata = data_access.get_index(index_path)
except FileNotFoundError:
    all_metadata = {}

# review.py'deki "belge -> ilgili chunk'lar" gruplama deseniyle aynı: bir
# belgenin OCR metnini chunk_id sırasına göre birleştirip tek bir metin elde
# ederiz (chunk chunk değil, belge bütünü üzerinden karşılaştırma yapılır).
chunks_by_doc: dict[str, list[dict]] = {}
for m in all_metadata.values():
    doc = m.get("source_doc", "bilinmiyor")
    chunks_by_doc.setdefault(doc, []).append(m)

doc_options = ["(seçilmedi)"] + sorted(chunks_by_doc)
selected_doc = st.selectbox("İndekslenmiş bir belge seç (opsiyonel)", doc_options)

pasted_text = st.text_area(
    "...veya doğrudan bir metin yapıştır (opsiyonel)",
    height=140,
    placeholder="Sınıflandırılacak belge metnini buraya yapıştırın.",
)

if pasted_text.strip():
    input_text = pasted_text.strip()
elif selected_doc != "(seçilmedi)":
    ordered_chunks = sorted(chunks_by_doc[selected_doc], key=lambda c: c.get("chunk_id", 0))
    input_text = "\n".join(c.get("text", "") for c in ordered_chunks)
else:
    input_text = ""

compare_clicked = st.button("Karşılaştır", type="primary", disabled=not input_text)
if not input_text:
    st.caption("Karşılaştırmak için bir belge seçin veya metin yapıştırın.")


def _run_classification(client) -> tuple[dict | None, float, Exception | None]:
    t0 = time.perf_counter()
    try:
        result = classifier.classify_document(input_text, client=client)
        return result, time.perf_counter() - t0, None
    except Exception as exc:  # local/cloud çağrısı her nedenle başarısız olabilir
        return None, time.perf_counter() - t0, exc


def _render_result(title: str, result: dict | None, duration: float, error: Exception | None) -> None:
    st.markdown(f"**{title}**")
    if error is not None:
        st.error(f"Şu an kullanılamıyor: {error}")
        return
    if result is None:
        st.info("Sonuç yok.")
        return
    st.markdown("".join(components.render_category_badge(c) for c in result.get("siniflar", [])) or "—", unsafe_allow_html=True)
    st.markdown(components.render_confidence_stamp(result.get("guven")), unsafe_allow_html=True)
    etiketler = ", ".join(result.get("etiketler", [])) or "—"
    st.caption(f"Etiketler: {html.escape(etiketler)}")
    st.caption(html.escape(result.get("gerekce") or "—"))
    st.caption(f"⏱ {duration:.2f} sn")


if compare_clicked and input_text:
    cloud_col, local_col = st.columns(2)

    with cloud_col:
        with st.spinner("Cloud model çalışıyor..."):
            config = llm_factory.load_llm_config()
            cloud_client = llm_factory.get_llm_client({"active_mode": "cloud", "cloud_model": config.get("cloud_model", {})})
            cloud_result, cloud_duration, cloud_error = _run_classification(cloud_client)
        _render_result(f"Cloud ({config.get('cloud_model', {}).get('model_name', '—')})", cloud_result, cloud_duration, cloud_error)

    with local_col:
        with st.spinner("Yerel model ilk kez indiriliyor, birkaç dakika sürebilir..."):
            local_result, local_duration, local_error = _run_classification(llm_factory.LocalHFClient(LOCAL_COMPARE_MODEL))
        _render_result(f"Local ({LOCAL_COMPARE_MODEL})", local_result, local_duration, local_error)

    st.caption(
        "Not: config'teki varsayılan hedef local model (Llama-3-8B-Instruct) gated/HF_TOKEN gerektirdiği için "
        "burada küçük, herkese açık bir vekil model (Qwen2.5-0.5B-Instruct) kullanılıyor — "
        "notebooks/12'deki gözlem: küçük local modellerde sınıflandırma doğruluğu cloud'a göre belirgin düşüyor."
    )
