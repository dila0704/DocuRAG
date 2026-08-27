"""
DocuRAG Streamlit uygulamasinin 4 sayfasinda da kullanilan ortak
bilesenler: "Onay Damgasi" (ConfidenceStamp), kategori rozeti, arama
sonuclarini belgeye gore gruplama, gercek highlighter vurgusu ve sinif
dagilimi donut grafigi.
"""
from __future__ import annotations

import html
import re
from collections import Counter
from datetime import datetime, timezone

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
