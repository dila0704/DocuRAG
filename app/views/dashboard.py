"""Dashboard sayfasi: sadece GERCEK verilerle doldurulan KPI'lar, sinif
dagilimi ve model maliyet/gecikme paneli -- bir BENTO GRID duzeninde.

Bento grid: farkli boyutlarda kutular + iki "hero" (renk-bloklu petrol/bronz
zeminli) vurgu kutusuyla gorsel hiyerarsi kuruyor (bkz. app/styles.py
".bento-*" siniflari). Native Streamlit widget'lari (orn. st.bar_chart) tek
bir buyuk HTML grid'e guvenilir sekilde gomulemedigi icin ("acik div" +
aralara serpistirilmis native widget'lar Streamlit'in DOM modelinde
kirilgan), tum grid TEK bir st.markdown(..., unsafe_allow_html=True)
cagrisiyla, hazir HTML parcalari (donut, aktivite listesi, gecikme bar'lari)
birlestirilerek basiliyor. Indirme butonlari (native oldugu icin) grid'in
ALTINDA, ayri bir satirda kalir.

"Model Performansi" paneli gercek veriyle doluyor: her basarili
LLMClient.generate() cagrisi llm_factory._append_usage_log() ile
data/processed/usage_log.jsonl'a yaziliyor (bkz. DOC-30 Oncelik 3),
data_access.load_usage_log() burada onu okuyor. Hic cagri yapilmamissa
panel bos bir durum mesaji gosterir -- sahte/mockup sayi YOK."""
from __future__ import annotations

from collections import Counter

import components
import data_access
import llm_factory
import streamlit as st
import vector_store
from styles import inject_global_styles

inject_global_styles()

st.markdown('<div class="disp" style="font-size:28px;">Dashboard</div>', unsafe_allow_html=True)
st.caption("Sistem durumu ve belge işleme özeti")

index_path = vector_store.load_index_path()
try:
    _, metadata = data_access.get_index(index_path)
except FileNotFoundError:
    st.info("Henüz indekslenmiş belge yok.")
    st.stop()

documents = vector_store.group_latest_by_source_doc(metadata)

total_docs = len(documents)
total_chunks = len(metadata)
pending_count = sum(1 for m in documents.values() if m.get("human_review"))

llm_settings = llm_factory.load_llm_config()
active_mode = llm_settings.get("active_mode", "cloud")

class_counter = Counter()
for m in documents.values():
    for category in m.get("siniflar", []):
        class_counter[category] += 1

timestamped = [(doc, m.get("ingested_at")) for doc, m in documents.items() if m.get("ingested_at")]
timestamped.sort(key=lambda item: item[1], reverse=True)

usage_df = data_access.load_usage_log()
usage_summary = None
usage_tile_body = '<p style="color:var(--text-tertiary);font-size:13px;">Henüz kaydedilmiş bir LLM çağrısı yok. Belge işleyin veya arama yapın.</p>'
if not usage_df.empty:
    total_calls = len(usage_df)
    total_cost = usage_df["cost_usd"].dropna().sum()
    known_cost_calls = usage_df["cost_usd"].notna().sum()
    avg_duration = usage_df["duration_s"].mean()
    usage_summary = {"total_calls": total_calls, "total_cost": total_cost, "avg_duration": avg_duration}

    cost_display = f"${total_cost:.4f}" if known_cost_calls else "—"
    by_provider = usage_df.groupby("provider")["duration_s"].mean().to_dict()
    # NOT: bu fragment daha SONRA baska bir cok-satirli f-string'in ORTASINA
    # gomulecek -- leading/trailing bos satir birakirsak Markdown'un HTML
    # block algisi erken kapanir (donut chart'ta yasanan ayni sorun, bkz.
    # asagidaki .strip() cagrilari). Bu yuzden tek satirlik parcalar halinde,
    # bos satir birakmadan birlestiriyoruz.
    usage_tile_body = (
        '<div style="display:flex;gap:22px;margin-bottom:16px;">'
        f'<div><span class="bento-label" style="margin-bottom:4px;">Çağrı</span><span class="mono" style="font-size:20px;font-weight:700;">{total_calls}</span></div>'
        f'<div><span class="bento-label" style="margin-bottom:4px;">Maliyet</span><span class="mono" style="font-size:20px;font-weight:700;">{cost_display}</span></div>'
        f'<div><span class="bento-label" style="margin-bottom:4px;">Ort. Gecikme</span><span class="mono" style="font-size:20px;font-weight:700;">{avg_duration:.2f} sn</span></div>'
        "</div>"
        + components.render_latency_bars(by_provider)
    )

recent_rows_html = ""
if not timestamped:
    recent_rows_html = '<p style="color:var(--text-tertiary);font-size:13px;">Henüz zaman damgalı işlem yok.</p>'
else:
    for doc, ts in timestamped[:6]:
        recent_rows_html += (
            f'<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;'
            f'border-bottom:1px solid var(--border);">'
            f'<span style="font-size:12.5px;font-weight:600;">{doc}</span>'
            f'<span class="mono" style="font-size:10.5px;color:var(--text-tertiary);">{components.format_relative_time(ts)}</span>'
            f"</div>"
        )

avg_chunks_sub = f"Belge başına ortalama {total_chunks / total_docs:.1f} parça" if total_docs else "—"
pending_variant = "bento-tile-bronze" if pending_count > 0 else ""
pending_sub = "İnceleme Kuyruğu'nda bekliyor" if pending_count > 0 else "Tümü onaylandı"

st.markdown(
    f"""
    <div class="bento-grid">
      <div class="bento-tile bento-tile-hero bento-span-2">
        <span class="bento-label">Toplam Belge</span>
        <span class="bento-number">{total_docs}</span>
        <div class="bento-sub">{len(class_counter)} kategoriye ayrıldı</div>
      </div>
      <div class="bento-tile">
        <span class="bento-label">Toplam Chunk</span>
        <span class="bento-number">{total_chunks}</span>
        <div class="bento-sub">{avg_chunks_sub}</div>
      </div>
      <div class="bento-tile {pending_variant}">
        <span class="bento-label">İnceleme Bekleyen</span>
        <span class="bento-number">{pending_count}</span>
        <div class="bento-sub">{pending_sub}</div>
      </div>

      <div class="bento-tile bento-tile-bronze bento-span-2">
        <span class="bento-label">Aktif LLM Modu</span>
        <span class="bento-number" style="font-size:26px;">{active_mode.capitalize()}</span>
        <div class="bento-sub">{"API üzerinden çalışıyor" if active_mode == "cloud" else "Yerel donanımda çalışıyor"}</div>
      </div>
      <div class="bento-tile bento-span-2">
        <span class="bento-label">Sınıf Dağılımı</span>
        {components.render_class_distribution_donut(class_counter).strip()}
      </div>

      <div class="bento-tile bento-span-2">
        <span class="bento-label">Son İşlemler</span>
        {recent_rows_html}
      </div>
      <div class="bento-tile bento-span-2">
        <span class="bento-label">Model Maliyet &amp; Gecikme</span>
        {usage_tile_body}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

report_html = components.render_printable_report_html(documents, class_counter, usage_summary)
st.download_button(
    "Yazdırılabilir Rapor (HTML)", report_html.encode("utf-8"), "docurag_rapor.html", "text/html",
    help="İndirip tarayıcıda açtıktan sonra 'PDF olarak yazdır' ile PDF'e çevirebilirsiniz.",
)
