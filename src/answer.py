"""
Arama sonuclarindan kaynak gosterimli (grounded) bir dogal dil cevabi
ureten modul (DOC-30, Oncelik 1: Kaynak Gosterimli Cevaplama).

vector_store.search() ciktisindaki (zaten skorlu/siralanmis) chunk'lar
numaralandirilip LLM'e baglam olarak verilir; modelden SADECE verilen
kaynaklardan alintilanan, her cumlesi en az bir kaynak indeksine atifta
bulunan bir JSON yaniti istenir.

Grounding (kaynaklanma) zorunlulugu modele GUVENEREK degil, backend'de
DOGRULANARAK saglanir: "sources" alani bos olan veya gecerli araligin
disinda bir indeks iceren her cumle, modelin cevabindan sonra kod
tarafindan elenir (bkz. _enforce_grounding). Boylece "kaynagi olmayan
hicbir cumle ozete girmemeli" kurali, prompt'a degil dogrulanabilir bir
son-isleme adimina dayanir.

classify_document() (bkz. classifier.py) ile ayni JSON parse/retry
desenini kullanir; iki modul de kucuk oldugu icin bu yardimci fonksiyon
ortak bir modulde degil, burada kendi kopyasi olarak tutuldu.
"""
from __future__ import annotations

import json
import logging

from llm_factory import LLMClient, get_llm_client

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 768
DEFAULT_MAX_JSON_ATTEMPTS = 2

SYSTEM_PROMPT = """Sen bir kaynak gosterimli (grounded) soru-cevap asistanisin. Sana numaralandirilmis kaynak parcalari ve bir kullanici sorgusu verilecek.

Gorevin, SADECE asagidaki kaynaklarda yer alan bilgilerle sorguyu yanitlamak ve yanitini asagidaki JSON semasina uygun sekilde dondurmektir:

{{
  "sentences": [
    {{"text": string, "sources": [int, ...]}}
  ]
}}

KURALLAR:
- Yanitin SADECE gecerli bir JSON nesnesi olmali, baska hicbir metin ekleme.
- Her "sentences" ogesindeki "sources" listesi, o cumlenin dayandigi kaynak(lar)in numarasini/numaralarini icermeli ve BOS OLAMAZ.
- Kaynaklarda olmayan hicbir bilgiyi UYDURMA. Bir cumleyi hicbir kaynaga dayandiramiyorsan o cumleyi YAZMA.
- Sorgu, verilen kaynaklarla hic yanitlanamiyorsa "sentences": [] dondur.
- Kisa ve net ol; gereksiz tekrar veya giris/sonuc cumlesi ekleme.

KAYNAKLAR:
{sources_block}
"""

USER_INSTRUCTION_TEMPLATE = "Kullanici sorgusu: {query}"


def _format_sources_block(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        lines.append(f"[{i}] (kaynak: {chunk.get('source_doc', 'bilinmiyor')}) {chunk['text']}")
    return "\n\n".join(lines)


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return json.loads(cleaned)


def _generate_and_parse_json(
    client: LLMClient,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    temperature: float,
    max_json_attempts: int,
) -> dict:
    """classifier._generate_and_parse_json ile ayni desen: gecerli JSON
    alinana kadar (en fazla max_json_attempts kez) modelden duzeltmesini
    ister."""
    current_user_message = user_message
    last_error: json.JSONDecodeError | None = None

    for attempt in range(max_json_attempts):
        raw_text = client.generate(
            system_prompt=system_prompt,
            user_message=current_user_message,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            return _extract_json(raw_text)
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                "generate_grounded_answer: gecersiz JSON yaniti (deneme %d/%d): %s",
                attempt + 1, max_json_attempts, exc,
            )
            current_user_message = (
                f"{user_message}\n\n---\n"
                f"Onceki yanitin gecerli bir JSON nesnesi degildi (hata: {exc}). "
                f"Onceki yanitin: {raw_text!r}\n"
                "Lutfen SADECE gecerli bir JSON nesnesi dondur, baska hicbir metin ekleme."
            )

    logger.error("generate_grounded_answer: %d denemeden sonra gecerli JSON alinamadi.", max_json_attempts)
    raise last_error


def _enforce_grounding(sentences: object, num_sources: int) -> list[dict]:
    """Her cumlenin gecerli en az bir kaynaga dayandigini backend'de
    dogrular; dogrulanamayan cumleleri sessizce eler (modele guvenmez).

    Args:
        sentences: modelin dondurdugu "sentences" alani (herhangi bir tur
            olabilir; beklenmedik bicimliyse bos liste kabul edilir).
        num_sources: baglamda verilen kaynak sayisi (gecerli indeks araligi: 1..num_sources).

    Returns:
        Sadece gecerli sekilde kaynaklanmis {"text": str, "sources": list[int]} ogelerinden olusan liste.
    """
    if not isinstance(sentences, list):
        return []

    grounded = []
    for item in sentences:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        sources = item.get("sources")
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(sources, list) or not sources:
            logger.warning("generate_grounded_answer: kaynaksiz cumle elendi: %r", text)
            continue
        valid_sources = [s for s in sources if isinstance(s, int) and 1 <= s <= num_sources]
        if not valid_sources:
            logger.warning("generate_grounded_answer: gecersiz kaynak indeksli cumle elendi: %r (sources=%r)", text, sources)
            continue
        grounded.append({"text": text.strip(), "sources": valid_sources})
    return grounded


def generate_grounded_answer(
    query: str,
    chunks: list[dict],
    client: LLMClient | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_json_attempts: int = DEFAULT_MAX_JSON_ATTEMPTS,
) -> dict:
    """Arama sonuclarindan (chunks) kaynak gosterimli bir cevap uretir.

    Args:
        query: kullanicinin dogal dil sorgusu.
        chunks: vector_store.search() ciktisi (her ogede en az "text" ve
            "source_doc" alanlari bulunmali). Bos liste verilirse cevap
            uretilmeden bos sonuc donulur.
        client: disaridan verilebilecek LLMClient (testlerde sahte istemci
            icin). None ise config/settings.yaml'a gore get_llm_client()
            Factory'si uzerinden kurulur.
        temperature: varsayilan 0.0 (deterministik).
        max_tokens: LLM yanitindan beklenen azami token sayisi.
        max_json_attempts: gecersiz JSON durumunda en fazla deneme sayisi.

    Returns:
        {
            "grounded": bool,  # en az bir cumle basariyla kaynaklandiysa True
            "sentences": [{"text": str, "sources": [int, ...]}, ...],
            "chunks": chunks,  # UI'nin kaynak numaralarini chunk'a eslemesi icin oldugu gibi geri verilir
        }
    """
    if not chunks:
        logger.info("generate_grounded_answer: chunks bos, cevap uretilmeyecek.")
        return {"grounded": False, "sentences": [], "chunks": []}

    client = client or get_llm_client()

    system_prompt = SYSTEM_PROMPT.format(sources_block=_format_sources_block(chunks))
    user_message = USER_INSTRUCTION_TEMPLATE.format(query=query)

    logger.info("generate_grounded_answer basladi: sorgu=%r kaynak_sayisi=%d", query, len(chunks))
    result = _generate_and_parse_json(
        client=client,
        system_prompt=system_prompt,
        user_message=user_message,
        max_tokens=max_tokens,
        temperature=temperature,
        max_json_attempts=max_json_attempts,
    )

    sentences = _enforce_grounding(result.get("sentences"), num_sources=len(chunks))
    logger.info(
        "generate_grounded_answer tamamlandi: %d/%d cumle kaynaklandi.",
        len(sentences), len(result.get("sentences") or []) if isinstance(result.get("sentences"), list) else 0,
    )

    return {"grounded": len(sentences) > 0, "sentences": sentences, "chunks": chunks}
