"""
LLM'den "SADECE JSON dondur" seklinde yanit isteyen moduller (classifier.py,
answer.py, field_extractor.py) arasinda paylasilan ortak ayristirma/yeniden-
deneme yardimcilari.

Onceden bu kod classifier.py ve answer.py'de BIREBIR KOPYA olarak duruyordu
("iki modul kucuk" gerekcesiyle bilincli birakilmisti); field_extractor.py
(DOC-30 B1) ile UCUNCU bir kopya acilacakti -- bu noktada tekrari onlemek
icin tek bir yere cikarildi. Davranis AYNEN korunur, sadece konum degisti.
"""
from __future__ import annotations

import json
import logging

from llm_factory import LLMClient

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return json.loads(cleaned)


def generate_and_parse_json(
    client: LLMClient,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    temperature: float,
    max_json_attempts: int,
    caller_name: str = "generate_and_parse_json",
) -> dict:
    """client.generate() cagirip yanit gecerli JSON olana kadar (en fazla
    max_json_attempts kez) dener; bozuk yanit alinirsa bir sonraki denemede
    modelden hatayi duzeltmesi istenir (DOC-31).

    caller_name: log mesajlarinda hangi ust-seviye fonksiyondan (orn.
    "classify_document") cagrildigini belirtmek icin -- ayristirma mantigi
    ortak olsa da, hangi ozelligin basarisiz oldugunu loglardan ayirt
    edebilmek onemli.
    """
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
            return extract_json(raw_text)
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning(
                "%s: gecersiz JSON yaniti (deneme %d/%d): %s",
                caller_name, attempt + 1, max_json_attempts, exc,
            )
            current_user_message = (
                f"{user_message}\n\n---\n"
                f"Onceki yanitin gecerli bir JSON nesnesi degildi (hata: {exc}). "
                f"Onceki yanitin: {raw_text!r}\n"
                "Lutfen SADECE gecerli bir JSON nesnesi dondur, baska hicbir metin ekleme."
            )

    logger.error("%s: %d denemeden sonra gecerli JSON alinamadi.", caller_name, max_json_attempts)
    raise last_error
