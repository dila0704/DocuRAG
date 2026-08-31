"""
LLM tabanli yapilandirilmis alan cikarimi modulu (DOC-30, "wow" ozellik seti
B1).

classifier.py belgeyi SINIFLANDIRIRKEN, bu modul belgeden somut ALANLAR
(tarih, tutar, taraflar, belge numarasi, konu) cikarir -- boylece arama
sadece semantik degil, "tutari 1000 TL uzerinde olan faturalar" gibi
FILTRELI sorgularla da yapilabilir hale gelir (bkz. vector_store.search()
-> metadata_filter, build_amount_range_filter/build_date_range_filter).

classifier.py ile AYNI JSON semasi zorlama deseni kullanilir (artik ortak
src/llm_json_utils.py uzerinden); LLM baglantisi yine dogrudan degil,
llm_factory.get_llm_client() Factory'si uzerinden kurulur.
"""
from __future__ import annotations

import logging
import re
from typing import Callable

from llm_factory import LLMClient, get_llm_client
from llm_json_utils import generate_and_parse_json as _generate_and_parse_json

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Belge sinifindan BAGIMSIZ, ortak alanlar. Belgede yoksa/uygulanamiyorsa
# None/bos liste donmesi beklenir -- olmayan bilgi asla UYDURULMAZ.
DEFAULT_FIELD_SCHEMA = ["tarih", "tutar", "taraflar", "belge_no", "konu"]

DEFAULT_MAX_JSON_ATTEMPTS = 2
DEFAULT_TEMPERATURE = 0.0

SYSTEM_PROMPT = """Sen bir belge alan cikarim asistanisin. Sana bir belgenin metni verilecek.

Gorevin, belgeden asagidaki JSON semasina uygun somut alanlari cikarmaktir:

{
  "tarih": string | null,     // belgedeki tarih, oldugu gibi (orn. "12.08.2026")
  "tutar": string | null,     // belgedeki parasal tutar, oldugu gibi (orn. "1.234,56 TL")
  "taraflar": [string],       // belgede gecen kisi/kurum isimleri (bos liste olabilir)
  "belge_no": string | null,  // fatura/belge/talep numarasi (varsa)
  "konu": string | null       // belgeyi 3-6 kelimede ozetleyen kisa bir baslik
}

KURALLAR:
- Yanitin SADECE gecerli bir JSON nesnesi olmali, baska hicbir metin ekleme.
- Belgede olmayan/emin olamadigin bir alani UYDURMA; null (ya da bos liste) birak.
- Degerleri belgede gectigi sekliyle (bicimini degistirmeden) yaz.
"""

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
    result = _generate_and_parse_json(
        client=client,
        system_prompt=SYSTEM_PROMPT,
        user_message=f"{USER_INSTRUCTION}\n\n---\n{text.strip()}\n---",
        max_tokens=max_tokens,
        temperature=temperature,
        max_json_attempts=max_json_attempts,
        caller_name="extract_fields",
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


def attach_fields_to_chunks(chunks: list[dict], fields_by_doc: dict[str, dict]) -> list[dict]:
    """classifier.attach_labels_to_chunks() ile birebir ayni desen: belge
    bazli extract_fields() ciktisini "source_doc" alani uzerinden ilgili
    chunk'lara "alanlar" anahtari altinda meta veri olarak ekler.

    Bir belgenin alan cikarimi henuz yapilmamissa (fields_by_doc'ta yoksa)
    chunk yine listede kalir, "alanlar" tum degerleri None/bos olan bir
    sozluk olur (filtreleme kodu bunu güvenle atlar).
    """
    empty_fields = {"tarih": None, "tutar": None, "taraflar": [], "belge_no": None, "konu": None}
    labeled = []
    for chunk in chunks:
        fields = fields_by_doc.get(chunk["source_doc"], empty_fields)
        labeled.append({**chunk, "alanlar": fields})
    return labeled
