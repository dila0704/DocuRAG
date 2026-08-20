"""
LLM tabanli otomatik belge siniflandirma ve etiketleme modulu.

OCR ile cikarilan belge metnini (veya notebook 02'deki yapilandirilmis JSON
ciktisinin metne cevrilmis halini) bir LLM'e (Claude) vererek belgeyi
onceden tanimlanmis bir sinifa (fatura/sozlesme/dilekce/talep formu vb.)
atar ve serbest metinli etiketler cikarir.

Notebook 02'deki (structured OCR) "yanit SADECE JSON" deseniyle tutarli
calisir; boylece proje genelinde LLM ciktisini ayristirma yaklasimi aynidir.
"""
from __future__ import annotations

import json
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL_NAME = "claude-sonnet-5"

DEFAULT_CATEGORIES = ["fatura", "sözleşme", "dilekçe", "talep formu", "diğer"]

FALLBACK_CATEGORY = "diğer"

SYSTEM_PROMPT_TEMPLATE = """Sen bir belge siniflandirma asistanisin. Sana bir belgenin metni verilecek.

Gorevin, belgeyi asagidaki JSON semasina uygun sekilde siniflandirmak ve etiketlemektir:

{{
  "sinif": string,        // Asagidaki listeden SECILMELI: {categories}
  "guven": number,        // 0.0-1.0 arasi, siniflandirmaya olan guvenin
  "etiketler": [string],  // Belgenin icerigini ozetleyen 2-5 serbest metin etiket (Turkce, kucuk harf)
  "gerekce": string       // Bu sinifi neden sectigine dair tek cumlelik kisa aciklama
}}

KURALLAR:
- Yanitin SADECE gecerli bir JSON nesnesi olmali.
- Markdown kod blogu (uc backtick), aciklama cumlesi veya baska hicbir metin EKLEME. Yanitin '{{' ile baslayip '}}' ile bitmeli.
- "sinif" alani MUTLAKA verilen listeden birisi olmali; hicbiri uymuyorsa "{fallback}" kullan.
- Belgede olmayan bilgi UYDURMA.
"""

USER_INSTRUCTION = "Bu belge metnini yukaridaki semaya gore siniflandir ve JSON olarak dondur."


def _get_client(api_key: str | None = None) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))


def _extract_json(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return json.loads(cleaned)


def classify_document(
    text: str,
    categories: list[str] | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """Belge metnini LLM ile siniflandirir ve etiketler.

    Args:
        text: OCR'dan gelen ham metin ya da yapilandirilmis alanlarin metne
            cevrilmis hali.
        categories: izin verilen sinif listesi (varsayilan: DEFAULT_CATEGORIES).
        model_name: kullanilacak Claude modeli.
        client: disaridan verilebilecek anthropic.Anthropic istemcisi
            (testlerde sahte/mock istemci vermek icin kullanislidir).

    Returns:
        {"sinif": str, "guven": float, "etiketler": list[str], "gerekce": str}
    """
    if not text or not text.strip():
        raise ValueError("text bos olamaz.")

    categories = categories or DEFAULT_CATEGORIES
    client = client or _get_client()

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        categories=", ".join(categories), fallback=FALLBACK_CATEGORY
    )

    response = client.messages.create(
        model=model_name,
        max_tokens=512,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": f"{USER_INSTRUCTION}\n\n---\n{text.strip()}\n---",
            }
        ],
    )

    raw_text = "".join(block.text for block in response.content if block.type == "text")
    result = _extract_json(raw_text)

    if result.get("sinif") not in categories:
        result["sinif"] = FALLBACK_CATEGORY

    return result


def classify_chunks(chunks: list[dict], **kwargs) -> dict:
    """text_splitter.split_text() ciktisindaki chunk'lari birlestirip tek
    seferde siniflandirir.

    Chunk chunk degil belge butunu uzerinden siniflandirma yapmak, tek ve
    tutarli bir sinif/etiket kumesi elde etmek icin daha dogru sonuc verir.
    """
    full_text = "\n".join(c["text"] for c in chunks)
    return classify_document(full_text, **kwargs)
