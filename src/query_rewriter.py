"""
Sorgu genisletme (HyDE) ve cok-turlu (multi-turn) konusma yogunlastirma
modulu (DOC-30, "wow" ozellik seti A3/A4).

Ikisi de sadece ARAMA SORGUSUNU degistirir -- nihai cevap hala
answer.generate_grounded_answer()'dan gecip _enforce_grounding ile
dogrulanir, yani grounding zinciri BOZULMAZ. Bu yuzden burada JSON semasi
zorlanmiyor (classifier.py/answer.py'nin _extract_json deseni gerekmiyor),
duz metin yaniti yeterli.
"""
from __future__ import annotations

import logging

from llm_factory import LLMClient, get_llm_client

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_HYDE_TEMPERATURE = 0.3  # bilinclii sapma: HyDE cesitlilik/terim zenginligi ister,
# projenin genelindeki temperature=0.0 (deterministik) varsayilanindan farkli.
DEFAULT_HYDE_MAX_TOKENS = 256

HYDE_SYSTEM_PROMPT = """Sen bir belge arama asistanisin. Sana bir kullanici sorgusu verilecek.

Gorevin, bu soruya cevap verecek TURDE bir belgenin (fatura, sozlesme, dilekce,
talep formu vb.) icerecegi VARSAYIMSAL (hipotetik) kisa bir metin parcasi
yazmaktir -- gercek bilgi olmasina gerek YOK, amac sorgunun konusuna uygun
terimleri/ifadeleri icermesi (boylece anlam bazli arama daha isabetli
calisir).

KURALLAR:
- 2-3 cumleyi gecme.
- SADECE varsayimsal metni yaz, aciklama/giris cumlesi EKLEME.
- Uydurma bir isim/tarih/tutar kullanabilirsin, bunlar gercek veri gibi
  sunulmayacak (sadece embedding icin kullanilir)."""

CONDENSE_SYSTEM_PROMPT = """Sen bir arama sorgusu yeniden yazma asistanisin. Sana onceki
konusma gecmisi ve kullanicinin YENI (takip) sorusu verilecek.

Gorevin, takip sorusunu onceki baglamdan BAGIMSIZ, tek basina anlasilir tek
bir arama sorgusuna donusturmektir (orn. "peki tarihi neydi?" + onceki soru
"laptop talebi" ise -> "laptop talebi belgesinin tarihi").

KURALLAR:
- SADECE yeniden yazilmis sorguyu yaz, aciklama EKLEME.
- Takip sorusu zaten bagimsizsa (onceki baglama ihtiyac duymuyorsa), oldugu
  gibi dondur."""


def _format_history(history: list[dict]) -> str:
    lines = []
    for turn in history:
        lines.append(f"Soru: {turn.get('query', '')}")
        if turn.get("answer_summary"):
            lines.append(f"Cevap ozeti: {turn['answer_summary']}")
    return "\n".join(lines)


def generate_hypothetical_answer(
    query: str,
    client: LLMClient | None = None,
    temperature: float = DEFAULT_HYDE_TEMPERATURE,
    max_tokens: int = DEFAULT_HYDE_MAX_TOKENS,
) -> str:
    """HyDE: sorguya cevap verecek varsayimsal bir belge parcasi uretir.

    Cagiran kod (bkz. pipeline.search_documents(expand_query=True)) bu metni
    orijinal sorguyla birlikte embed eder -- saf HyDE yerine bu "hybrid HyDE"
    tercih edildi, boylece orijinal sorgunun terimleri kaybolmaz.
    """
    client = client or get_llm_client()
    logger.info("generate_hypothetical_answer: sorgu=%r", query)
    return client.generate(
        system_prompt=HYDE_SYSTEM_PROMPT,
        user_message=query,
        max_tokens=max_tokens,
        temperature=temperature,
    ).strip()


def condense_conversation(
    history: list[dict],
    follow_up: str,
    client: LLMClient | None = None,
    temperature: float = 0.0,
    max_tokens: int = 128,
) -> str:
    """Onceki konusma gecmisi + yeni takip sorusunu, bagimsiz tek bir arama
    sorgusuna yogunlastirir.

    Args:
        history: [{"query": str, "answer_summary": str}, ...] (bkz.
            app/views/search.py session_state["conversation_history"]).
        follow_up: kullanicinin yeni sorusu.

    Gecmis bossa LLM'e hic gidilmez, follow_up oldugu gibi donulur (gereksiz
    LLM cagrisi/gecikme eklenmez).
    """
    if not history:
        return follow_up

    client = client or get_llm_client()
    user_message = f"Konusma gecmisi:\n{_format_history(history)}\n\nYeni soru: {follow_up}"
    logger.info("condense_conversation: %d onceki tur, takip_sorusu=%r", len(history), follow_up)
    return client.generate(
        system_prompt=CONDENSE_SYSTEM_PROMPT,
        user_message=user_message,
        max_tokens=max_tokens,
        temperature=temperature,
    ).strip()
