"""
DocuRAG Streamlit uygulamasinin 4 sayfasinda da kullanilan ortak
bilesenler: "Onay Damgasi" (ConfidenceStamp), kategori rozeti, arama
sonuclarini belgeye gore gruplama, gercek highlighter vurgusu ve sinif
dagilimi donut grafigi.
"""
from __future__ import annotations

import html
import re
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Callable

import pandas as pd

_CATEGORY_VAR = {
    "fatura": "fatura",
    "sözleşme": "sozlesme",
    "dilekçe": "dilekce",
    "talep formu": "talep",
    "diğer": "diger",
}

_CATEGORY_ORDER = ["talep formu", "fatura", "sözleşme", "dilekçe", "diğer"]


def render_confidence_stamp(score: float | None) -> str:
    """Guven skorunu tasarimdaki "Onay Damgasi" olarak HTML dondurur.

    Esikler: >=0.85 dolu vurgu kenarlik, 0.6-0.85 soluk kenarlik,
    <0.6 (veya skor yoksa) kesik cizgili + "INCELEME BEKLIYOR" etiketi.
    """
    if not isinstance(score, (int, float)):
        return '<div class="stamp stamp-low"><span class="v">—</span><span class="l">İNCELEME BEKLİYOR</span></div>'

    pct = round(score * 100)
    if score >= 0.85:
        return f'<div class="stamp stamp-high"><span class="v">%{pct}</span></div>'
    if score >= 0.6:
        return f'<div class="stamp stamp-mid"><span class="v">%{pct}</span></div>'
    return f'<div class="stamp stamp-low"><span class="v">%{pct}</span><span class="l">İNCELEME BEKLİYOR</span></div>'


def render_category_badge(category: str) -> str:
    var = _CATEGORY_VAR.get(category, "diger")
    return f'<span class="badge" style="background:var(--cat-{var}-bg);color:var(--cat-{var});">{html.escape(category)}</span>'


def format_chunk_score(chunk: dict) -> str:
    """Sonuc kartlarindaki [n] skor etiketini bicimlendirir.

    retrieval.hybrid_search()'un RRF skoru cosine benzerligi DEGIL (kucuk,
    0-1 disina cikabilen bir sayi) -- eskiden burada her zaman "%{skor*100}"
    gosteriliyordu, hibrit arama sonrasi bu yanıltıcı olurdu (orn. RRF 0.03
    -> "%3" gibi anlamsiz gorunurdu). score_type alanina gore ayirt edilir."""
    index = chunk.get("_citation_index")
    score = chunk.get("score", 0.0)
    if chunk.get("score_type") == "rrf":
        return f"[{index}] RRF {score:.3f}"
    return f"[{index}] %{round(score * 100)}"


def group_results_by_document(chunks: list[dict]) -> list[dict]:
    """Ayni source_doc'a ait chunk'lari tek bir gruba toplar; her chunk
    kendi skorunu (ve varsa "_citation_index"'ini) korur -- boylece UI
    hem chunk-seviyesinde relevance skorunu hem de kaynak-numarasi
    esleme mantigini kaybetmez.

    Girdi zaten en alakaliden en az alakaliya siraliysa (vector_store.search()
    ciktisi gibi), gruplar en yuksek chunk skoruna gore siralanir.
    """
    groups: dict[str, dict] = {}
    order: list[str] = []

    for chunk in chunks:
        doc = chunk.get("source_doc", "bilinmiyor")
        if doc not in groups:
            groups[doc] = {
                "source_doc": doc,
                "chunks": [],
                "siniflar": chunk.get("siniflar", []),
                "guven": chunk.get("guven"),
                "human_review": chunk.get("human_review", False),
            }
            order.append(doc)
        groups[doc]["chunks"].append(chunk)

    result = []
    for doc in order:
        group = groups[doc]
        group["best_score"] = max(c.get("score", 0.0) for c in group["chunks"])
        result.append(group)

    result.sort(key=lambda g: g["best_score"], reverse=True)
    return result


def highlight_terms(text: str, query: str) -> str:
    """Sorgudaki (2 karakterden uzun) kelimeleri metinde gercek bir
    highlighter kalemi gibi vurgular. Girdi metni HTML-escape edilir,
    sonuc unsafe_allow_html ile guvenle render edilebilir."""
    escaped = html.escape(text)
    terms = sorted({t for t in re.findall(r"\w+", query, flags=re.UNICODE) if len(t) > 2}, key=len, reverse=True)
    if not terms:
        return escaped
    pattern = re.compile("(" + "|".join(re.escape(t) for t in terms) + ")", re.IGNORECASE | re.UNICODE)
    return pattern.sub(r'<span class="mark">\1</span>', escaped)


def render_class_distribution_donut(counter: Counter) -> str:
    """Sinif dagilimini tasarimdaki conic-gradient donut + legend olarak
    HTML dondurur."""
    total = sum(counter.values())
    if total == 0:
        return '<p style="color:var(--text-tertiary);font-size:13px;">Henüz sınıflandırılmış belge yok.</p>'

    stops = []
    legend_rows = []
    cursor = 0.0
    for category in _CATEGORY_ORDER:
        count = counter.get(category, 0)
        if count == 0:
            continue
        pct = count / total * 100
        var = _CATEGORY_VAR[category]
        stops.append(f"var(--cat-{var}) {cursor:.2f}% {cursor + pct:.2f}%")
        legend_rows.append(
            f'<div style="display:flex;align-items:center;gap:9px;">'
            f'<span class="legend-dot" style="background:var(--cat-{var});"></span>'
            f'<span style="font-size:11.5px;color:var(--text-primary);flex-grow:1;">{category.capitalize()}</span>'
            f'<span class="mono" style="font-size:11px;color:var(--text-tertiary);">%{pct:.0f}</span></div>'
        )
        cursor += pct

    gradient = ", ".join(stops)
    legend_html = "".join(legend_rows)
    return f"""
    <div style="display:flex;flex-direction:column;align-items:center;gap:18px;">
      <div style="position:relative;width:120px;height:120px;border-radius:50%;background:conic-gradient({gradient});">
        <div style="position:absolute;inset:19px;background:var(--surface);border-radius:50%;display:flex;flex-direction:column;align-items:center;justify-content:center;">
          <span class="disp" style="font-size:18px;">{total}</span>
          <span class="mono" style="font-size:8px;color:var(--text-tertiary);">BELGE</span>
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:9px;width:100%;">{legend_html}</div>
    </div>
    """


def render_document_graph_svg(graph, width: int = 640, height: int = 420) -> str:
    """Belge iliski grafigini (bkz. graph_builder.build_document_graph) inline
    bir <svg> string'i olarak render eder -- render_class_distribution_donut()'un
    conic-gradient deseniyle tutarli: yeni bir JS grafik kutuphanesi EKLENMEZ,
    koordinatlar Python tarafinda (networkx.spring_layout, sabit seed=42 ile
    DETERMINISTIK) hesaplanip duz SVG'ye donusturulur."""
    if graph.number_of_nodes() == 0:
        return '<p style="color:var(--text-tertiary);font-size:13px;">Henüz belge ilişkisi bulunamadı (ortak taraf içeren birden fazla belge gerekir).</p>'

    import networkx as nx

    pos = nx.spring_layout(graph, seed=42)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad = 36

    def _scale(point):
        x, y = point
        sx = pad + (x - min_x) / (max_x - min_x) * (width - 2 * pad) if max_x > min_x else width / 2
        sy = pad + (y - min_y) / (max_y - min_y) * (height - 2 * pad) if max_y > min_y else height / 2
        return sx, sy

    lines = []
    for a, b, data in graph.edges(data=True):
        x1, y1 = _scale(pos[a])
        x2, y2 = _scale(pos[b])
        stroke_width = min(1 + data.get("weight", 1), 5)
        shared = html.escape(", ".join(data.get("shared", [])))
        lines.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="var(--border)" stroke-width="{stroke_width}"><title>{shared}</title></line>'
        )

    nodes = []
    for node in graph.nodes():
        x, y = _scale(pos[node])
        label = html.escape(str(node))
        nodes.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="var(--bronze)" stroke="var(--surface)" stroke-width="2"><title>{label}</title></circle>'
            f'<text x="{x:.1f}" y="{y + 20:.1f}" font-size="9.5" text-anchor="middle" fill="var(--text-secondary)">{label}</text>'
        )

    body = "".join(lines) + "".join(nodes)
    return f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" style="max-width:{width}px;">{body}</svg>'


def render_latency_bars(values: dict[str, float]) -> str:
    """Saglayici basina ortalama gecikmeyi duz HTML/CSS bar'lar olarak render
    eder (donut chart deseniyle tutarli -- yeni bir grafik kutuphanesi
    eklenmiyor). Dashboard'un bento grid'i icinde `st.bar_chart` (native bir
    widget, raw HTML grid'e gomulemez) yerine kullaniliyor."""
    if not values:
        return '<p style="color:var(--text-tertiary);font-size:13px;">Henüz veri yok.</p>'

    max_val = max(values.values()) or 1
    rows = []
    for label, val in sorted(values.items(), key=lambda kv: kv[1], reverse=True):
        pct = (val / max_val) * 100
        rows.append(
            f'<div style="margin-bottom:11px;">'
            f'<div style="display:flex;justify-content:space-between;font-size:11.5px;margin-bottom:5px;">'
            f'<span style="font-weight:600;">{html.escape(label)}</span>'
            f'<span class="mono" style="color:var(--text-tertiary);">{val:.2f} sn</span></div>'
            f'<div style="background:var(--surface-alt);border-radius:6px;height:8px;overflow:hidden;">'
            f'<div style="width:{pct:.1f}%;height:100%;background:var(--accent);border-radius:6px;"></div>'
            f'</div></div>'
        )
    return "".join(rows)


def stream_sentences(
    rendered_sentences: list[str],
    placeholder,
    wrap: Callable[[str], str] = lambda body: body,
    delay: float = 0.35,
) -> None:
    """Zaten olusturulmus (HTML-render edilmis) cumleleri bir st.empty()
    placeholder'ina kademeli olarak basar (DOC-30 C2).

    ONEMLI (algisal, gercek degil): bu fonksiyon GERCEK bir LLM token-stream'i
    DEGILDIR -- cagrilmadan once TUM cumleler zaten uretilmis/dogrulanmis
    olmalidir (bkz. answer.generate_grounded_answer -> _enforce_grounding).
    Amac SADECE okuma deneyimini "canli yaziliyor" gibi hissettirmek; ilk
    cumle ekranda belirmeden once backend uretimi ZATEN bitmistir, gercek bir
    time-to-first-token kazanci YOKTUR.

    Args:
        rendered_sentences: her biri zaten HTML-guvenli (escape edilmis) bir
            cumle string'i (bkz. app/views/search.py sentence_parts).
        placeholder: st.empty() ciktisi (ya da .markdown(str, **kwargs) metodu
            olan herhangi bir nesne -- testlerde sahte bir placeholder verilebilir).
        wrap: her adimda biriken metni saran bir fonksiyon (orn. cevap
            kartinin HTML sablonu). Varsayilan: metni oldugu gibi dondurur.
        delay: cumleler arasi bekleme (saniye). Testlerde 0 verilerek
            gercek bekleme YAPILMADAN cagri sirasi/sayisi dogrulanabilir.
    """
    accumulated: list[str] = []
    for rendered in rendered_sentences:
        accumulated.append(rendered)
        placeholder.markdown(wrap(" ".join(accumulated)), unsafe_allow_html=True)
        if delay:
            time.sleep(delay)


def search_results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Arama sonuclarini (vector_store.search()/retrieval.hybrid_search()
    ciktisi) CSV disa aktarimi icin duz bir tabloya cevirir (DOC-30 C3)."""
    rows = [
        {
            "belge": r.get("source_doc", ""),
            "skor": r.get("score"),
            "skor_turu": r.get("score_type", "cosine"),
            "siniflar": ", ".join(r.get("siniflar", [])),
            "guven": r.get("guven"),
            "metin": r.get("text", ""),
        }
        for r in results
    ]
    return pd.DataFrame(rows)


def render_printable_report_html(
    documents: dict[str, dict],
    class_counter: Counter,
    usage_summary: dict | None = None,
) -> str:
    """Dashboard icin, tarayicinin "PDF olarak yazdir"iyla kullanilacak
    bagimsiz/gomulu-stilli bir HTML rapor sayfasi uretir (DOC-30 C3).

    Yeni bir PDF kutuphanesi EKLENMEZ (proje felsefesiyle tutarli, minimum
    yeni bagimlilik) -- tarayicinin yerlesik yazdirma islevi kullanilir.
    `-webkit-print-color-adjust: exact` olmadan bazi tarayicilar arkaplan
    renklerini/grafikleri yazdirmadan atlar, bu yuzden acikca eklenir.
    """
    rows_html = "".join(
        f"<tr><td>{html.escape(doc)}</td><td>{html.escape(', '.join(info.get('siniflar', [])))}</td>"
        f"<td>{info.get('guven') if info.get('guven') is not None else '—'}</td>"
        f"<td>{html.escape(format_relative_time(info.get('ingested_at')))}</td></tr>"
        for doc, info in sorted(documents.items(), key=lambda kv: kv[1].get("ingested_at", ""), reverse=True)
    )
    class_rows_html = "".join(
        f"<tr><td>{html.escape(category)}</td><td>{count}</td></tr>"
        for category, count in class_counter.most_common()
    )
    usage_html = ""
    if usage_summary:
        usage_html = f"""
        <h2>Model Kullanımı</h2>
        <p>Toplam çağrı: {usage_summary.get('total_calls', 0)} ·
           Tahmini maliyet: ${usage_summary.get('total_cost', 0):.4f} ·
           Ortalama gecikme: {usage_summary.get('avg_duration', 0):.2f} sn</p>
        """

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="tr"><head><meta charset="utf-8"><title>DocuRAG Raporu</title>
<style>
  * {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
  body {{ font-family: Georgia, serif; color: #2B2620; background: #F7F3EC; padding: 32px; }}
  h1 {{ font-size: 22px; }}
  h2 {{ font-size: 15px; margin-top: 28px; border-bottom: 1px solid #D8CFBE; padding-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12px; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #E6DFD1; }}
  th {{ text-transform: uppercase; font-size: 10px; letter-spacing: 0.05em; color: #6B6153; }}
  .meta {{ color: #6B6153; font-size: 11px; }}
</style></head>
<body>
  <h1>DocuRAG — Sistem Raporu</h1>
  <p class="meta">Oluşturulma: {generated_at} · {len(documents)} belge</p>
  {usage_html}
  <h2>Sınıf Dağılımı</h2>
  <table><thead><tr><th>Kategori</th><th>Belge Sayısı</th></tr></thead><tbody>{class_rows_html}</tbody></table>
  <h2>Belgeler</h2>
  <table><thead><tr><th>Belge</th><th>Sınıflar</th><th>Güven</th><th>İşlenme</th></tr></thead><tbody>{rows_html}</tbody></table>
</body></html>"""


def format_relative_time(iso_timestamp: str | None) -> str:
    """"3 dk önce" bicimli goreli zaman metni. Gecersiz/eksik girdi icin "-"."""
    if not iso_timestamp:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_timestamp)
    except ValueError:
        return "-"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    seconds = (datetime.now(timezone.utc) - dt).total_seconds()
    if seconds < 60:
        return "az önce"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes} dk önce"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} sa önce"
    days = hours // 24
    return f"{days} gün önce"
