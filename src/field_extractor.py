"""
LLM tabanli yapilandirilmis alan cikarimi modulu (DOC-30, "wow" ozellik seti
B1).

classifier.py belgeyi SINIFLANDIRIRKEN, bu modul belgeden somut ALANLAR
(tarih, tutar, taraflar, belge numarasi, konu) cikarir -- boylece arama
sadece semantik degil, "tutari 1000 TL uzerinde olan faturalar" gibi
FILTRELI sorgularla da yapilabilir hale gelir (bkz. vector_store.search()
-> metadata_filter, build_amount_range_filter/build_date_range_filter).

classifier.py ile AYNI desen kullanilir: JSON semasi artik prompt
talimatina degil, client.generate_structured() uzerinden API seviyesinde
zorlanir (DOC-34, bkz. src/llm_factory.py); LLM baglantisi yine dogrudan
degil, llm_factory.get_llm_client() Factory'si uzerinden kurulur. Belge
metni (OCR'dan geldigi icin GUVENILMEZ) UNTRUSTED_CONTENT_NOTICE +
wrap_untrusted() ile prompt injection'a karsi isaretlenir.
"""
from __future__ import annotations

import logging
import re
from typing import Callable

from llm_factory import LLMClient, get_llm_client
from llm_json_utils import UNTRUSTED_CONTENT_NOTICE, wrap_untrusted

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Belge sinifindan BAGIMSIZ, ortak alanlar. Belgede yoksa/uygulanamiyorsa
# None/bos liste donmesi beklenir -- olmayan bilgi asla UYDURULMAZ.
DEFAULT_FIELD_SCHEMA = ["tarih", "tutar", "taraflar", "belge_no", "konu"]

DEFAULT_MAX_JSON_ATTEMPTS = 2
DEFAULT_TEMPERATURE = 0.0

# generate_structured()'un AnthropicClient/OpenAIClient uzerinde API
# seviyesinde zorladigi sema (DOC-34); LocalHFClient fallback yolunda
# kullanilmaz (bkz. SYSTEM_PROMPT'taki insan-okunur aciklama).
FIELD_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "tarih": {"type": ["string", "null"], "description": "Belgedeki tarih, oldugu gibi (orn. '12.08.2026')."},
        "tutar": {"type": ["string", "null"], "description": "Belgedeki parasal tutar, oldugu gibi (orn. '1.234,56 TL')."},
        "taraflar": {"type": "array", "items": {"type": "string"}, "description": "Belgede gecen kisi/kurum isimleri."},
        "belge_no": {"type": ["string", "null"], "description": "Fatura/belge/talep numarasi (varsa)."},
        "konu": {"type": ["string", "null"], "description": "Belgeyi 3-6 kelimede ozetleyen kisa bir baslik."},
    },
    "required": ["tarih", "tutar", "taraflar", "belge_no", "konu"],
}

SYSTEM_PROMPT = """Sen bir belge alan cikarim asistanisin. Sana bir belgenin metni verilecek.

Gorevin, belgeden asagidaki semaya uygun somut alanlari cikarmaktir:

{
  "tarih": string | null,     // belgedeki tarih, oldugu gibi (orn. "12.08.2026")
  "tutar": string | null,     // belgedeki parasal tutar, oldugu gibi (orn. "1.234,56 TL")
  "taraflar": [string],       // belgede gecen kisi/kurum isimleri (bos liste olabilir)
  "belge_no": string | null,  // fatura/belge/talep numarasi (varsa)
  "konu": string | null       // belgeyi 3-6 kelimede ozetleyen kisa bir baslik
}

KURALLAR:
- Belgede olmayan/emin olamadigin bir alani UYDURMA; null (ya da bos liste) birak.
- Degerleri belgede gectigi sekliyle (bicimini degistirmeden) yaz.
""" + UNTRUSTED_CONTENT_NOTICE

USER_INSTRUCTION = "Bu belge metninden yukaridaki semaya gore alanlari cikar ve JSON olarak dondur."

# "1.234,56 TL" / "1234.56" / "500 TL" gibi bicimlerden sayisal degeri
# cikarmaya calisir (Turkce bicim varsayimi: nokta binlik, virgul ondalik
# ayraci). Ayirt edilemeyen bicimler icin None doner -- sayisal filtreleme
# (build_amount_range_filter) yanlis bir sayi UYDURMAZ.
_AMOUNT_PATTERN = re.compile(r"(\d{1,3}(?:\.\d{3})*(?:,\d+)?|\d+(?:[.,]\d+)?)")


def _validate_fields(result: dict) -> dict:
    """Beklenmedik tipte gelen alanlari (orn. "taraflar" string donerse)
    guvenli varsayilanlara dusurur -- cagiran/UI kodu asla beklenmedik tip
    almaz."""
    tarih = result.get("tarih")
    tutar = result.get("tutar")
    belge_no = result.get("belge_no")
    konu = result.get("konu")
    taraflar = result.get("taraflar")

    return {
        "tarih": tarih if isinstance(tarih, str) and tarih.strip() else None,
        "tutar": tutar if isinstance(tutar, str) and tutar.strip() else None,
        "belge_no": belge_no if isinstance(belge_no, str) and belge_no.strip() else None,
        "konu": konu if isinstance(konu, str) and konu.strip() else None,
        "taraflar": [t for t in taraflar if isinstance(t, str) and t.strip()] if isinstance(taraflar, list) else [],
    }


def extract_fields(
    text: str,
    client: LLMClient | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = 512,
    max_json_attempts: int = DEFAULT_MAX_JSON_ATTEMPTS,
) -> dict:
    """Belge metninden yapilandirilmis alanlari (tarih/tutar/taraflar/belge_no/konu) cikarir.

    Args:
        text: OCR'dan gelen ham metin.
        client: disaridan verilebilecek LLMClient (testlerde sahte istemci icin).
            None ise get_llm_client() Factory'si uzerinden kurulur.

    Returns:
        {"tarih": str|None, "tutar": str|None, "taraflar": list[str],
         "belge_no": str|None, "konu": str|None}

    Raises:
        ValueError: text bossa.
    """
    if not text or not text.strip():
        raise ValueError("text bos olamaz.")

    client = client or get_llm_client()

    logger.info("extract_fields basladi: metin_uzunlugu=%d", len(text))
    result = client.generate_structured(
        system_prompt=SYSTEM_PROMPT,
        user_message=f"{USER_INSTRUCTION}\n\n---\n{wrap_untrusted(text.strip())}\n---",
        schema=FIELD_EXTRACTION_SCHEMA,
        tool_name="extract_fields",
        tool_description="Belgeden tarih/tutar/taraflar/belge_no/konu alanlarini cikarir.",
        max_tokens=max_tokens,
        temperature=temperature,
        max_json_attempts=max_json_attempts,
    )
    validated = _validate_fields(result)
    logger.info("extract_fields tamamlandi: %s", validated)
    return validated


def parse_turkish_amount(text: str | None) -> float | None:
    """"1.234,56 TL" / "500 TL" gibi bicimlerden sayisal degeri cikarir.

    Bilinen sinir: Turkce bicim varsayilir (virgul varsa nokta binlik
    ayrac sayilir); ayirt edilemeyen/beklenmedik bicimler icin None doner
    -- yanlis bir sayi asla UYDURULMAZ, o kayit sadece filtre disi kalir."""
    if not text:
        return None
    match = _AMOUNT_PATTERN.search(text)
    if not match:
        return None
    raw = match.group(1)

    if "," in raw:
        # "1.234,56" -> nokta binlik, virgul ondalik.
        normalized = raw.replace(".", "").replace(",", ".")
    elif "." in raw and len(raw.rsplit(".", 1)[1]) == 3:
        # Virgul yok ama son nokta grubu tam 3 haneli ("2.500", "1.234.567")
        # -> Turkce binlik gruplama, ondalik degil.
        normalized = raw.replace(".", "")
    else:
        # Nokta yok, ya da son grup 3 hane degil ("12.99" gibi) -> ondalik nokta.
        normalized = raw

    try:
        return float(normalized)
    except ValueError:
        return None


def parse_date_to_iso(text: str | None) -> str | None:
    """"12.08.2026" ya da "2026-08-12" bicimlerini "YYYY-MM-DD" karsilastirilabilir
    bir stringe cevirir. Taninmayan bicimler icin None doner (bkz. parse_turkish_amount
    ile ayni "uydurmama" ilkesi)."""
    if not text:
        return None
    text = text.strip()
    iso_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if iso_match:
        return iso_match.group(0)[:10]
    tr_match = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})", text)
    if tr_match:
        day, month, year = tr_match.groups()
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return None


def build_amount_range_filter(min_amount: float | None, max_amount: float | None) -> Callable[[dict], bool]:
    """vector_store.search(..., metadata_filter=...) ile kullanilacak bir
    filtre fonksiyonu dondurur: chunk'in "alanlar.tutar" alani ayristirilamiyorsa
    (None) ya da aralik disindaysa eler."""

    def _filter(chunk_metadata: dict) -> bool:
        amount = parse_turkish_amount((chunk_metadata.get("alanlar") or {}).get("tutar"))
        if amount is None:
            return False
        if min_amount is not None and amount < min_amount:
            return False
        if max_amount is not None and amount > max_amount:
            return False
        return True

    return _filter


def build_date_range_filter(min_date: str | None, max_date: str | None) -> Callable[[dict], bool]:
    """build_amount_range_filter ile ayni desen, "alanlar.tarih" uzerinden.
    min_date/max_date "YYYY-MM-DD" biciminde string kabul eder (orn.
    st.date_input(...).isoformat())."""

    def _filter(chunk_metadata: dict) -> bool:
        date_iso = parse_date_to_iso((chunk_metadata.get("alanlar") or {}).get("tarih"))
        if date_iso is None:
            return False
        if min_date is not None and date_iso < min_date:
            return False
        if max_date is not None and date_iso > max_date:
            return False
        return True

    return _filter


# --- Tutar agregasyon sorgulari (DOC-35) ------------------------------------
#
# "En yuksek tutarli fatura hangisi?" gibi sorular KARSILASTIRMA/AGREGASYON
# gerektirir: dogru cevap icin TUM belgelerin taranmasi lazim. Ama standart
# RAG akisi (hybrid_search -> top_k parca -> LLM) sadece "en alakali 5 parca"yi
# getirir -- LLM bu 5 parcayi gorup, gorMEDIGI digerlerini varmis gibi
# "en yuksek" diye sunabilir (yanlis olmadigini SOYLEMEDEN yanlis olabilir,
# cunku hicbir zaman TUM verilere erisemedi). Bu iki fonksiyon, boyle
# sorgulari yakalayip cevabi LLM'e SORMADAN, TUM metadata uzerinde deterministik
# olarak hesaplar -- anomaly.py'nin "tam corpus, LLM'siz, kod-dogrulanmis"
# ilkesiyle AYNI yaklasim.
_SUPERLATIVE_MAX_PATTERN = re.compile(r"en\s+(y[uü]ksek|fazla|b[uü]y[uü]k|pahal[iı])", re.IGNORECASE)
_SUPERLATIVE_MIN_PATTERN = re.compile(r"en\s+(d[uü][sş][uü]k|az|k[uü][cç][uü]k|ucuz)", re.IGNORECASE)
_AMOUNT_CONTEXT_WORDS = ("tutar", "fiyat", "fatura", "ucret", "ücret", "meblag", "meblağ", "para")

# Sorguda gecen bir kategori ipucunu ("faturanın" -> "fatura") classifier.
# DEFAULT_CATEGORIES'teki gercek sinif adina esler -- "en yuksek TUTAR"
# sorgusu TUM belge turlerini tarasin, ama "en yuksek tutarli FATURA"
# sorgusu SADECE fatura siniflandirilmis belgelerle sinirlandirilsin diye
# (aksi halde bir kira sozlesmesi "en yuksek tutar" oldugu icin yanlislikla
# "en yuksek tutarli fatura" olarak sunulabilir -- gercek uygulamada boyle
# bir hata bulunup buraya eklendi).
_CATEGORY_HINTS: dict[str, str] = {
    "fatura": "fatura",
    "sözleşme": "sözleşme", "sozlesme": "sözleşme",
    "dilekçe": "dilekçe", "dilekce": "dilekçe",
    "talep formu": "talep formu", "talep": "talep formu",
}


def detect_amount_superlative_query(query: str) -> str | None:
    """Sorgu bir tutar karsilastirmasi mi (`"max"`/`"min"`), yoksa normal bir
    sorgu mu (`None`) tespit eder. Hem bir ustunluk ifadesi ("en yuksek",
    "en ucuz" vb.) HEM DE tutar baglamina isaret eden bir kelime ("tutar",
    "fatura" vb.) birlikte gecmezse None doner -- "en buyuk departman" gibi
    tutarla ilgisiz sorgulari yanlislikla yakalamamak icin."""
    normalized = query.lower()
    if not any(w in normalized for w in _AMOUNT_CONTEXT_WORDS):
        return None
    if _SUPERLATIVE_MAX_PATTERN.search(normalized):
        return "max"
    if _SUPERLATIVE_MIN_PATTERN.search(normalized):
        return "min"
    return None


def detect_category_hint(query: str) -> str | None:
    """Sorguda gecen bir kategori kelimesini ("faturanın", "sözleşme" vb.)
    classifier.DEFAULT_CATEGORIES'teki gercek sinif adina esler. Bulunamazsa
    None doner -- find_amount_superlative_document() bu durumda TUM
    kategorilerde arar (eski davranis)."""
    normalized = query.lower()
    for hint, category in _CATEGORY_HINTS.items():
        if hint in normalized:
            return category
    return None


def find_amount_superlative_document(metadata: dict[int, dict], direction: str, category: str | None = None) -> dict | None:
    """metadata'daki TUM belgeleri (retrieval'in top-k'siyla SINIRLI degil)
    tarayip "alanlar.tutar" alani ayristirilabilen belgeler arasinda GERCEK
    en yuksek (`direction="max"`) ya da en dusuk (`direction="min"`) tutarli
    olani bulur. Tutari ayristirilamayan (None donen) belgeler sessizce
    atlanir -- uydurma bir karsilastirma yapilmaz.

    Args:
        category: verilirse (bkz. detect_category_hint), SADECE "siniflar"
            alaninda bu kategoriyi tasiyan belgeler karsilastirmaya girer
            (orn. "fatura" -> bir kira sozlesmesi, tutari daha yuksek olsa
            bile "en yuksek tutarli fatura" cevabina asla giremez).

    Returns:
        {"source_doc", "amount", "tutar_raw", "belge_no", "konu", "taraflar"}
        ya da hicbir belge (kategori filtresinden gecip) tutari ayristirilamadiysa None.
    """
    best: dict | None = None
    seen_docs: set[str] = set()
    for m in metadata.values():
        doc = m.get("source_doc")
        if doc is None or doc in seen_docs:
            continue
        if category is not None and category not in (m.get("siniflar") or []):
            continue
        alanlar = m.get("alanlar") or {}
        amount = parse_turkish_amount(alanlar.get("tutar"))
        if amount is None:
            continue
        seen_docs.add(doc)
        if best is None or (direction == "max" and amount > best["amount"]) or (direction == "min" and amount < best["amount"]):
            best = {
                "source_doc": doc, "amount": amount, "tutar_raw": alanlar.get("tutar"),
                "belge_no": alanlar.get("belge_no"), "konu": alanlar.get("konu"),
                "taraflar": alanlar.get("taraflar", []),
            }
    return best


def attach_fields_to_chunks(chunks: list[dict], fields_by_doc: dict[str, dict]) -> list[dict]:
    """classifier.attach_labels_to_chunks() ile birebir ayni desen: belge
    bazli extract_fields() ciktisini "source_doc" alani uzerinden ilgili
    chunk'lara "alanlar" anahtari altinda meta veri olarak ekler.

    Bir belgenin alan cikarimi henuz yapilmamissa (fields_by_doc'ta yoksa)
    chunk yine listede kalir, "alanlar" tum degerleri None/bos olan bir
    sozluk olur (filtreleme kodu bunu güvenle atlar).
    """
    empty_fields: dict[str, object] = {"tarih": None, "tutar": None, "taraflar": [], "belge_no": None, "konu": None}
    labeled = []
    for chunk in chunks:
        fields = fields_by_doc.get(chunk["source_doc"], empty_fields)
        labeled.append({**chunk, "alanlar": fields})
    return labeled
