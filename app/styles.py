"""
Uygulama genelinde kullanilan CSS token'lari (:root degiskenleri) ve ortak
yardimci siniflar (.stamp, .badge, .mark). Palet, kullanicinin secimiyle
"Gece Lacivert & Buz Mavisi/Gumus" temasina donusturuldu: canli/doygun
lacivert zemin (--bg) + buz mavisi/gumus vurgu (--accent/--bronze). Onceki
asamalar ("sessiz luks" acik krem tema, "Gece Indigo", "Altin & Amber")
artik gecerli degil; degisken ADLARI ayni birakildi, sadece DEGERLERI
degisti (bkz. :root altindaki yorum) -- boylece tema tek noktadan
degistirilebiliyor.

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

/* Koyu/premium tema -- degisken ADLARI (--accent, --bronze vb.) BILEREK
   ayni birakildi -- tum sayfalar/bilesenler bu degiskenler uzerinden
   renklendigi icin, sadece DEGERLERI degistirerek uygulama genelinde tek
   noktadan tema degisikligi yapilabiliyor (kod tekrar duzenlenmedi). */
:root{
  /* Onceki --bg (#0F1420) dusuk doygunluklu bir "gri-lacivert" idi;
     kullanici isteği uzerine tam doygun, canli bir koyu lacivert olarak
     degistirildi. Diger yuzeyler de ayni renk ailesinden (mavi hue'su
     sabit, sadece parlaklik artan) tonlarla yeniden ayarlandi. */
  --bg: #0A1550;
  --surface: #101C63;
  --surface-alt: #172569;
  --border: #2A3A8A;
  --border-strong: #3B4DA0;
  --text-primary: #F2F3FC;
  --text-secondary: #BAC1EC;
  --text-tertiary: #8790C9;

  /* "Buz Mavisi & Gumus" vurgu paleti -- lacivert zemin uzerine kullanici
     secimiyle uygulandi (minimal, kurumsal/enterprise hissi). */
  --accent: #7DD3FC;
  --accent-faded: #BAE6FD;
  --accent-soft-bg: #123246;

  --bronze: #CBD5E1;
  --bronze-soft-bg: #2A3142;

  --warn: #FF6B6B;
  --warn-soft-bg: #3A1E20;

  /* --accent/--bronze artik ikisi de soguk mavi/gumus ailesinde oldugu icin,
     arama-terimi vurgusu (.mark) KASITLI olarak SICAK bir renkten (amber)
     seciliyor -- aksi halde "vurgulanan kelime" genel vurgu renginden
     ayirt edilemez hale gelirdi. */
  --highlight: #FBBF24;

  --cat-fatura: #7DB8E8; --cat-fatura-bg: #1B3350;
  --cat-sozlesme: #E39BD1; --cat-sozlesme-bg: #3A1F38;
  --cat-dilekce: #F0A576; --cat-dilekce-bg: #3D2418;
  --cat-talep: #8FD474; --cat-talep-bg: #1F3517;
  --cat-diger: #B8B2A0; --cat-diger-bg: #302C24;

  --shadow: 0 1px 2px rgba(0,0,0,0.35), 0 3px 12px rgba(0,0,0,0.4);
  --shadow-lg: 0 2px 6px rgba(0,0,0,0.4), 0 14px 34px rgba(0,0,0,0.55);
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

/* color acikca koyu tutuluyor: --highlight parlak/acik bir renk (amber),
   surrounding metin koyu temada ACIK renkli oldugu icin miras alinirsa
   vurgulanan kelime kendi zemininde okunmaz olurdu. */
.mark{background:var(--highlight);color:#1C1A17;padding:0 3px;border-radius:2px;}

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

.citation-link{font-family:'IBM Plex Mono',monospace;font-weight:700;font-size:11.5px;color:var(--accent);text-decoration:none;padding:0 2px;}
.citation-link:hover{text-decoration:underline;}
/* Ozet karti artik bento-tile-bronze (renk-bloklu zeminli) -- --accent
   rengi bu zemin uzerinde yeterince kontrastli olmayabilir, sade beyaza
   geciyor (hangi vurgu paleti secilirse secilsin guvenli/evrensel). */
.bento-tile-bronze .citation-link{color:#FFFFFF;text-decoration:underline;}

/* Belge onizleme karti -- gercek belge goruntusu hafifce bulanik
   (blur) halde arka plan, ustunde konu/dosya adi bir scrim uzerinde.
   Hover'da netlesir -- "tikla ve gor" davranisini ima eder. */
.doc-preview-chip{position:relative;display:block;width:132px;height:96px;border-radius:10px;overflow:hidden;text-decoration:none;box-shadow:var(--shadow);flex:none;background:var(--surface-alt);}
.doc-preview-chip img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;filter:blur(3px) brightness(0.88) saturate(0.9);transform:scale(1.08);transition:filter 0.2s ease, transform 0.2s ease;}
.doc-preview-chip:hover img{filter:blur(0) brightness(0.97);transform:scale(1.0);}
.doc-preview-chip .scrim{position:absolute;inset:0;background:linear-gradient(180deg, rgba(28,26,23,0) 40%, rgba(28,26,23,0.72) 100%);}
.doc-preview-chip .label{position:absolute;left:9px;right:9px;bottom:7px;color:#FBF9F4;font-family:'IBM Plex Sans',sans-serif;font-size:11px;font-weight:600;line-height:1.3;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;}
.doc-preview-chip .lens{position:absolute;top:8px;right:8px;width:20px;height:20px;border-radius:50%;background:rgba(252,251,247,0.9);display:flex;align-items:center;justify-content:center;font-size:11px;opacity:0;transition:opacity 0.2s ease;}
.doc-preview-chip:hover .lens{opacity:1;}

/* Bento grid -- Dashboard ve Tum Belgeler sayfalari icin. Mevcut duz
   ("her kart ayni boyut/renk") kart deseninden bilincli bir sapma: farkli
   boyutlarda kutular + iki "hero" (renk-bloklu, koyu petrol/bronz zeminli)
   vurgu varyanti ile gorsel hiyerarsi kuruluyor. Renk token'lari (--accent,
   --bronze, vb.) korunuyor -- yeni olan DUZEN/TIPOGRAFI, marka paleti degil. */
.bento-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:6px;}
.bento-tile{background:var(--surface);border:1px solid var(--border);border-radius:22px;padding:22px 24px;box-shadow:var(--shadow);transition:transform 0.18s ease, box-shadow 0.18s ease;}
.bento-tile:hover{transform:translateY(-3px);box-shadow:var(--shadow-lg);}
.bento-tile-hero{background:linear-gradient(135deg, var(--accent) 0%, #0D2E42 100%);border:none;}
.bento-tile-hero .bento-label{color:rgba(247,244,238,0.72);}
.bento-tile-hero .bento-number, .bento-tile-hero .bento-sub{color:#FBF9F4;}
.bento-tile-bronze{background:linear-gradient(135deg, var(--bronze) 0%, #2A3142 100%);border:none;}
.bento-tile-bronze .bento-label{color:rgba(251,247,240,0.72);}
.bento-tile-bronze .bento-number, .bento-tile-bronze .bento-sub{color:#FBF9F4;}
.bento-span-2{grid-column:span 2;}
.bento-span-3{grid-column:span 3;}
.bento-span-4{grid-column:span 4;}
.bento-label{font-family:'IBM Plex Mono',monospace;font-size:10px;font-weight:600;letter-spacing:0.09em;text-transform:uppercase;color:var(--text-tertiary);margin-bottom:12px;display:block;}
.bento-number{font-family:'Instrument Serif',Georgia,serif;font-size:42px;line-height:1;color:var(--text-primary);}
.bento-sub{font-size:12px;color:var(--text-secondary);margin-top:8px;line-height:1.5;}
@media (max-width: 900px){ .bento-grid{grid-template-columns:repeat(2,1fr);} .bento-span-3, .bento-span-4{grid-column:span 2;} }

/* Tum Belgeler bento kart -- st.container(border=True) icine gomulu; tam
   dokunmayla renklenen bir st.container CSS'i degil, icerigin kendisi
   (banner + buyuk kucuk resim) bir bento kutusu HISSI verecek sekilde
   tasarlandi (Streamlit'in native container/widget DOM modeliyle celismeden). */
.doc-bento-banner{margin:-1rem -1rem 14px -1rem;padding:14px 18px;border-radius:12px 12px 0 0;display:flex;align-items:center;justify-content:space-between;gap:10px;}
.doc-bento-banner.is-hero{background:linear-gradient(135deg, var(--accent) 0%, #0D2E42 100%);color:#FBF9F4;}
.doc-bento-banner.is-plain{background:var(--surface-alt);}
.doc-bento-banner .title{font-family:'Instrument Serif',Georgia,serif;font-size:15px;}

/* Metin-agirlikli icerik icin "hero" varyanti -- Dashboard'daki dolu renk
   bloklu hero (bento-tile-hero, sayi-agirlikli KPI'lar icin uygun) burada
   okunurlugu bozardi. Bunun yerine acik zemin + kalin vurgu kenarligi +
   kose etiketi ile "spotlight" hissi veriliyor (arama sonuclari, en iyi
   eslesme; inceleme kuyrugu, en dusuk guvenli kayit gibi). */
.bento-tile-spotlight{border:2px solid var(--accent);position:relative;}
.bento-tile-spotlight .spotlight-tag{position:absolute;top:-11px;right:20px;background:var(--accent);color:#FBF9F4;font-family:'IBM Plex Mono',monospace;font-size:9px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;padding:3px 10px;border-radius:20px;}
.bento-tile-warn-outline{border:2px solid var(--warn);}
.bento-tile-warn-outline .spotlight-tag{position:absolute;top:-11px;right:20px;background:var(--warn);color:#FBF9F4;font-family:'IBM Plex Mono',monospace;font-size:9px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;padding:3px 10px;border-radius:20px;}

/* Arama sayfasi kontrolleri -- onceden hicbir cerceve/stil olmadan
   "ciplak" native widget'lar olarak duruyordu. Ham HTML ile SARMAK
   (Dashboard/Inventory'deki gibi) native widget'lar icin guvenilir degil;
   bunun yerine Streamlit'in resmi `key=` -> `.st-key-<key>` CSS class
   mekanizmasi kullanildi (bkz. st.container/st.text_input dokumantasyonu)
   -- hicbir ham HTML sarmalama YOK, sadece native DOM'a class-bazli CSS. */
.st-key-search-query input{
  background:var(--surface);
  border:1.5px solid var(--border);
  border-radius:16px;
  padding-top:14px;
  padding-bottom:14px;
  font-size:16px;
  color:var(--text-primary);
  box-shadow:var(--shadow);
  transition:border-color 0.15s ease, box-shadow 0.15s ease;
}
.st-key-search-query input:focus{
  border-color:var(--accent);
  box-shadow:0 0 0 3px var(--accent-soft-bg);
}
.st-key-search-toolbar{
  background:var(--surface);
  border:1px solid var(--border);
  border-radius:20px;
  padding:18px 22px 6px 22px;
  box-shadow:var(--shadow);
  margin-bottom:14px;
}
.st-key-search-toolbar [data-baseweb="select"] > div{
  background:var(--surface-alt);
  border-color:var(--border);
  border-radius:10px;
}
.st-key-search-toolbar [data-testid="stSliderThumbValue"],
.st-key-search-toolbar [data-testid="stSliderTickBar"]{color:var(--text-tertiary);}
</style>
"""


def inject_global_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
