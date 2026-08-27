"""
Belge gorsellerinden (fatura/sozlesme/dilekce/talep formu) metin cikaran
multimodal OCR modulu.

Notebook 01'deki dogrudan Anthropic vision cagrisini src/'e tasir (DOC-31):
boylece uctan uca pipeline (src/pipeline.py) OCR adimini da notebook'a
bagimli kalmadan kod olarak cagirabilir.

Not (bilinen mimari sinir): llm_factory.LLMClient arayuzu (generate) su an
sadece metin (system_prompt + user_message) alir, goruntu girdisini
desteklemez. Bu yuzden bu modul llm_factory uzerinden degil, dogrudan
Anthropic'in multimodal mesaj formatini kullanarak calisir -- yani OCR adimi
su an icin llm_factory'nin cloud/local Factory soyutlamasinin DISINDADIR.
Local (huggingface) bir gorsel-dil modeliyle OCR yapmak istenirse,
LLMClient arayuzunun goruntu girdisi de destekleyecek sekilde genisletilmesi
gerekir.
"""
from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_MODEL_NAME = "claude-sonnet-5"
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_BASE = 1.0

SYSTEM_PROMPT = (
    "Sen bir dokuman OCR asistanisin. Sana bir belge gorseli verilecek. "
    "Gorevin, gorseldeki tum metni satir satir, birebir (yorum katmadan, "
    "ozetlemeden, duzeltmeden) okumaktir. Yanitin SADECE okudugun metin "
    "olmali, ekstra aciklama ekleme."
)

USER_INSTRUCTION = "Bu belge gorselindeki metni birebir oku ve oldugu gibi dondur."

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _media_type(image_path: str) -> str:
    ext = os.path.splitext(image_path)[1].lower()
    return _MEDIA_TYPES.get(ext, "image/png")


def extract_text_from_image(
    image_path: str,
    client: Any | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_tokens: int = 1024,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_backoff_base: float = DEFAULT_RETRY_BACKOFF_BASE,
) -> str:
    """Bir belge gorselinin tum metnini Claude'un multimodal (vision)
    yetenegiyle cikarir.

    Args:
        image_path: OCR yapilacak gorsel dosyasinin yolu (.png/.jpg/.webp/.gif).
        client: disaridan verilebilecek bir Anthropic-uyumlu istemci
            (testlerde sahte istemci vermek icin kullanislidir). None ise
            ANTHROPIC_API_KEY (.env) ile gercek bir anthropic.Anthropic()
            olusturulur.
        model_name: kullanilacak (vision destekleyen) model adi.
        max_tokens: beklenen azami cikti token sayisi.
        max_retries: gecici hatalarda (agdaki kesinti, rate limit vb.)
            yapilacak ek deneme sayisi (DOC-31).
        retry_backoff_base: denemeler arasi ussel bekleme suresinin tabani (sn).

    Returns:
        Gorselden okunan ham metin.

    Raises:
        FileNotFoundError: image_path bulunamazsa.
        RuntimeError: max_retries denemeden sonra da istek basarisiz olursa.
    """
    base64_image = _encode_image(image_path)

    if client is None:
        import anthropic

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    total_attempts = max_retries + 1
    last_exc: Exception | None = None

    for attempt in range(total_attempts):
        t0 = time.time()
        logger.info(
            "OCR basladi: dosya=%s model=%s deneme=%d/%d",
            image_path, model_name, attempt + 1, total_attempts,
        )
        try:
            response = client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": _media_type(image_path),
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": USER_INSTRUCTION},
                    ],
                }],
            )
        except Exception as exc:
            last_exc = exc
            logger.exception(
                "OCR basarisiz (deneme %d/%d): dosya=%s sure=%.2fsn",
                attempt + 1, total_attempts, image_path, time.time() - t0,
            )
            if attempt < max_retries:
                delay = retry_backoff_base * (2 ** attempt)
                logger.warning("OCR %.2fsn sonra yeniden denenecek.", delay)
                time.sleep(delay)
                continue
            raise RuntimeError(
                f"OCR {total_attempts} denemeden sonra basarisiz oldu: {image_path}"
            ) from last_exc

        text = "".join(block.text for block in response.content if block.type == "text").strip()
        logger.info(
            "OCR tamamlandi: dosya=%s sure=%.2fsn metin_uzunlugu=%d",
            image_path, time.time() - t0, len(text),
        )
        return text

    raise RuntimeError(f"OCR basarisiz: {image_path}") from last_exc  # pragma: no cover
