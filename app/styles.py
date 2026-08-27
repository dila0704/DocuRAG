"""
Onaylanan tasarim canvasindan (bkz. proje sohbet gecmisi -- "sessiz luks"
palet: krem zemin + murekkep + petrol yesili + bronz) tasinan CSS token'lari
ve tum sayfalarda ortak kullanilan yardimci siniflar (.stamp, .badge, .mark).

.streamlit/config.toml zaten Streamlit'in kendi bilesenlerini (buton,
input, sidebar) bu palete gore temalandiriyor; burasi sadece ozel
bilesenler (ConfidenceStamp, kategori rozeti, highlighter) ve yazi tipi
ithalatlari icin.
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600;700&display=swap');

:root{
  --bg: #F7F4EE;
  --surface: #FCFBF7;
  --surface-alt: #F0EBE1;
  --border: #E4DFD3;
  --border-strong: #D6CFBE;
  --text-primary: #1C1A17;
  --text-secondary: #5B5750;
  --text-tertiary: #8C8579;

  --accent: #0E5F55;
  --accent-faded: #8FB3AC;
  --accent-soft-bg: #E3EDEA;

  --bronze: #8B6F47;
  --bronze-soft-bg: #EFE7DA;

  --warn: #B4472A;
  --warn-soft-bg: #F5E2DB;

  --highlight: #FAC775;

  --cat-fatura: #3C4568; --cat-fatura-bg: #E8E9F0;
  --cat-sozlesme: #6B4267; --cat-sozlesme-bg: #EFE6EE;
  --cat-dilekce: #7A3B36; --cat-dilekce-bg: #F0E3E1;
  --cat-talep: #4B5A3A; --cat-talep-bg: #E9ECE1;
  --cat-diger: #6B6459; --cat-diger-bg: #EAE7E1;

  --shadow: 0 1px 2px rgba(28,26,23,0.04), 0 3px 10px rgba(28,26,23,0.05);
  --shadow-lg: 0 2px 4px rgba(28,26,23,0.05), 0 12px 28px rgba(28,26,23,0.09);
}

.stApp, body { font-family: 'IBM Plex Sans', system-ui, sans-serif; }

/* st.markdown ile enjekte edilen bu <style> Streamlit'in kendi CSS'inden
   SONRA yuklendigi icin ustteki kural, Material ikon span'larinin kendi
   font-family'sini (ayni ozgullukte, ama sonradan gelerek) eziyordu --
   ikonlar duz metin ("search" vb.) olarak goruntuleniyordu. Bu kural
   ikon fontunu acikca geri yukler. */
[data-testid="stIconMaterial"] { font-family: "Material Symbols Rounded" !important; }

.disp{font-family:'Instrument Serif',Georgia,serif; color:var(--text-primary);}
.mono{font-family:'IBM Plex Mono',monospace;}

.badge{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:600;padding:4px 10px;border-radius:6px;white-space:nowrap;}

.mark{background:var(--highlight);padding:0 3px;border-radius:2px;}

.legend-dot{width:7px;height:7px;border-radius:50%;display:inline-block;flex:none;}

/* "Onay Damgasi" -- guven skoru damgasi */
.stamp{display:inline-flex;flex-direction:column;align-items:center;justify-content:center;transform:rotate(-4deg);border-radius:4px;padding:3px 9px;line-height:1.35;}
.stamp-high{border:2px solid var(--accent);background:var(--accent-soft-bg);}
.stamp-high .v{color:var(--accent);}
.stamp-mid{border:2px solid var(--accent-faded);}
.stamp-mid .v{color:var(--text-secondary);}
.stamp-low{border:2px dashed var(--warn);background:var(--warn-soft-bg);}
.stamp-low .v{color:var(--warn);}
.stamp .v{font-family:'IBM Plex Mono',monospace;font-weight:700;letter-spacing:0.02em;font-size:13px;}
.stamp .l{font-family:'IBM Plex Mono',monospace;font-size:7px;font-weight:600;letter-spacing:0.09em;color:var(--warn);}

/* arama sonuc karti */
.result-card{background:var(--surface);border-radius:12px;padding:18px 20px;box-shadow:var(--shadow);margin-bottom:14px;transition:box-shadow 0.2s ease, border-color 0.2s ease;border:2px solid transparent;}
.result-card.highlighted{border-color:var(--accent);box-shadow:var(--shadow-lg);}

/* AI Ozet karti */
.answer-card{background:var(--surface);border-radius:12px;padding:20px 24px;box-shadow:var(--shadow);border-left:2.5px solid var(--bronze);margin-bottom:16px;}
.citation-link{font-family:'IBM Plex Mono',monospace;font-weight:700;font-size:11.5px;color:var(--accent);text-decoration:none;padding:0 2px;}
.citation-link:hover{text-decoration:underline;}
</style>
"""


def inject_global_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
