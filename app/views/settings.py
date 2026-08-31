"""Ayarlar sayfasi: config/settings.yaml -> llm_settings.active_mode/provider/
model_name'i UI'dan degistirir.

Onceden bu bilgi sidebar'da/Dashboard'da SADECE goruntuleniyordu -- LLM
Factory'nin (bkz. src/llm_factory.py) en guclu ozelligi olan "tek config
degisikligiyle cloud/local gecisi" UI'dan hic yapilamiyordu. Bu sayfa
settings.yaml'a DOKUNMAZ (yorumlari korumak icin) -- llm_factory.
save_llm_settings_override() sadece ayri, yorumsuz bir 'settings.local.yaml'
yazar; uygulamanin yeniden baslatilmasi GEREKMEZ (get_llm_client() her
cagrida config'i taze okur)."""
from __future__ import annotations

from pathlib import Path

import llm_factory
import streamlit as st
from styles import inject_global_styles

inject_global_styles()

st.markdown('<div class="disp" style="font-size:28px;">Ayarlar</div>', unsafe_allow_html=True)
st.caption("Aktif LLM sağlayıcısını/modelini değiştirin — kaydettikten sonra yeniden başlatmaya gerek yok")

CLOUD_PROVIDERS = ["anthropic", "openai"]
LOCAL_PROVIDERS = ["huggingface"]

config = llm_factory.load_llm_config()
current_mode = config.get("active_mode", "cloud")
current_cloud = config.get("cloud_model", {}) or {}
current_local = config.get("local_model", {}) or {}

override_path = Path(llm_factory.DEFAULT_CONFIG_PATH).parent / "settings.local.yaml"
if override_path.exists():
    st.caption(f"ℹ️ Şu an bir override dosyası aktif ({override_path.name}) — base `settings.yaml` değişmedi.")

active_mode = st.radio("Aktif mod", ["cloud", "local"], index=["cloud", "local"].index(current_mode), horizontal=True)

st.markdown("**Cloud model**")
c1, c2 = st.columns(2)
cloud_provider = c1.selectbox("Sağlayıcı", CLOUD_PROVIDERS, index=CLOUD_PROVIDERS.index(current_cloud.get("provider", "anthropic")) if current_cloud.get("provider") in CLOUD_PROVIDERS else 0)
# NOT: model adi (orn. "claude-sonnet-5") kasitli olarak gizleniyor
# (type="password") -- alan hala DUZENLENEBILIR (yazilan deger kaydedilir),
# sadece mevcut deger ekranda acik metin gorunmuyor.
cloud_model_name = c2.text_input("Model adı", value=current_cloud.get("model_name", ""), type="password")

st.markdown("**Local model (huggingface)**")
l1, l2 = st.columns(2)
local_provider = l1.selectbox("Sağlayıcı", LOCAL_PROVIDERS, index=0)
local_model_name = l2.text_input("Model adı", value=current_local.get("model_name", ""), type="password")

if st.button("Kaydet", type="primary"):
    updates = {
        "active_mode": active_mode,
        "cloud_model": {"provider": cloud_provider, "model_name": cloud_model_name},
        "local_model": {"provider": local_provider, "model_name": local_model_name},
    }
    llm_factory.save_llm_settings_override(updates)
    st.success("Kaydedildi. Bir sonraki LLM çağrısından itibaren geçerli olacak.")
    st.rerun()
