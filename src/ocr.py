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
import difflib
import io
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


# --- Gorsel vurgulama (DOC-30, "wow" ozellik seti B4) -----------------------
#
# Mühendislik karari: Claude vision'dan dogrudan bounding-box/koordinat
# ISTENMIYOR -- vision LLM'lerin piksel-hassas koordinat uretme guvenilirligi
# dusuk (halusinasyon riski) ve kod tarafinda "bu koordinat dogru mu" diye
# BAGIMSIZ dogrulamak mumkun degil (answer._enforce_grounding'in "kaynak
# indeksi gecerli mi" diye kontrol edebilmesinin aksine) -- bu, projenin
# "grounding kodda dogrulanir, promptta degil" ilkesini ihlal eder.
#
# Bunun yerine: Tesseract (klasik/geometrik OCR), Claude OCR'in YANINDA,
# OPSIYONEL ve LAZY (sadece kullanici bir kaynaga tikladiginda) calisan bir
# "geometri sidecar'i" olarak kullanilir. Claude OCR ana metin/dogruluk
# kaynagi olarak KALIR; Tesseract sadece "bu metin gorselde YAKLASIK nerede"
# sorusuna cevap arar.
#
# Bilinen operasyonel sinir: Tesseract-OCR, pip paketinin (pytesseract)
# YANI SIRA sistem duzeyinde bir binary + "tur.traineddata" Turkce dil
# paketi gerektirir (pip install ile gelmez). Kurulu degilse bu ozellik
# SESSIZCE devre disi kalir -- ana pipeline (OCR->RAG->arama) buna hicbir
# sekilde bagimli DEGILDIR.

MIN_WORD_MATCH_RATIO = 0.4


def extract_word_boxes(image_path: str, lang: str = "tur") -> list[dict] | None:
    """Tesseract ile gorseldeki her kelimenin yaklasik piksel konumunu
    cikarir. Tesseract (binary) kurulu degilse ya da herhangi bir sekilde
    basarisiz olursa exception FIRLATMAZ -- None doner, cagiran kod bunu
    "konum bilgisi yok" olarak ele alip vurgusuz tam gorseli gosterir.

    Returns:
        [{"text": str, "left": int, "top": int, "width": int, "height": int}, ...]
        ya da None.
    """
    try:
        import pytesseract
        from PIL import Image
        from pytesseract import Output
    except ImportError:
        logger.warning("extract_word_boxes: pytesseract/Pillow kurulu degil, gorsel vurgulama devre disi.")
        return None

    try:
        with Image.open(image_path) as image:
            data = pytesseract.image_to_data(image, lang=lang, output_type=Output.DICT)
    except Exception:
        logger.warning(
            "extract_word_boxes: Tesseract calistirilamadi (binary/dil paketi kurulu olmayabilir), "
            "gorsel vurgulama bu belge icin devre disi.", exc_info=True,
        )
        return None

    boxes = []
    for i, text in enumerate(data.get("text", [])):
        text = text.strip()
        if not text:
            continue
        boxes.append({
            "text": text,
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
        })
    logger.info("extract_word_boxes: %s icin %d kelime konumu cikarildi.", image_path, len(boxes))
    return boxes


def locate_chunk_bbox(chunk_text: str, word_boxes: list[dict] | None) -> dict | None:
    """Bir chunk metnini, Tesseract'in cikardigi kelime kutulariyla BULANIK
    (fuzzy) eslestirip eslesen kelimelerin bounding box'larinin BIRLESIMINI
    dondurur.

    Birebir string eslesmesi ARANMAZ: Claude OCR ile Tesseract farkli metin
    uretebilir (noktalama/bosluk farklari, OCR hatalari). Eslesme orani
    MIN_WORD_MATCH_RATIO'nun altindaysa None doner -- yanlis/yaniltici bir
    kutu gostermektense hic gostermemek tercih edilir.

    Returns:
        {"left": int, "top": int, "width": int, "height": int, "match_ratio": float} ya da None.
    """
    if not word_boxes or not chunk_text or not chunk_text.strip():
        return None

    chunk_words = [w.lower() for w in chunk_text.split()]
    box_words = [b["text"].lower() for b in word_boxes]
    if not chunk_words or not box_words:
        return None

    matcher = difflib.SequenceMatcher(a=chunk_words, b=box_words, autojunk=False)
    matched_indices = [
        block.b + offset
        for block in matcher.get_matching_blocks()
        for offset in range(block.size)
    ]

    match_ratio = len(matched_indices) / len(chunk_words)
    if not matched_indices or match_ratio < MIN_WORD_MATCH_RATIO:
        return None

    matched_boxes = [word_boxes[i] for i in matched_indices]
    left = min(b["left"] for b in matched_boxes)
    top = min(b["top"] for b in matched_boxes)
    right = max(b["left"] + b["width"] for b in matched_boxes)
    bottom = max(b["top"] + b["height"] for b in matched_boxes)
    return {"left": left, "top": top, "width": right - left, "height": bottom - top, "match_ratio": match_ratio}


def render_highlighted_image(image_path: str, bbox: dict | None) -> bytes:
    """Orijinal belge gorselini, verilirse bbox uzerinde bir dikdortgen
    vurgusuyla PNG bayt dizisi olarak dondurur (bbox None ise vurgusuz tam
    gorsel). app/views/search.py bunu dogrudan st.image()'a verir."""
    from PIL import Image, ImageDraw

    with Image.open(image_path) as original:
        image = original.convert("RGB")

    if bbox:
        draw = ImageDraw.Draw(image)
        x0, y0 = bbox["left"], bbox["top"]
        x1, y1 = x0 + bbox["width"], y0 + bbox["height"]
        draw.rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], outline=(139, 111, 71), width=4)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
