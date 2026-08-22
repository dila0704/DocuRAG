"""
LLM tabanli otomatik belge siniflandirma ve etiketleme modulu.

OCR ile cikarilan belge metnini (veya notebook 02'deki yapilandirilmis JSON
ciktisinin metne cevrilmis halini) bir LLM'e (Claude) vererek belgeye
onceden tanimlanmis siniflardan (fatura/sozlesme/dilekce/talep formu vb.)
uygun olan BIRDEN FAZLASINI atar (multi-label) ve serbest metinli etiketler
cikarir.

Notebook 02'deki (structured OCR) "yanit SADECE JSON" deseniyle tutarli
calisir; boylece proje genelinde LLM ciktisini ayristirma yaklasimi aynidir.

Belirsiz durumlar icin ayri bir "belirsiz" sinifi ACILMAZ: LLM'in dondugu
"guven" skoru bir esik degeriyle karsilastirilir, esigin altinda kalan
sonuclar "human_review": true olarak isaretlenir. Bu sayede kategori
listesi temiz kalir ve dusuk guvenli belgeler ayri bir inceleme kuyruguna
yonlendirilebilir (bkz. classify_document dokumantasyonu).
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

# guven bu esigin altinda kalirsa belge human_review=True olarak isaretlenir.
# OCR/embedding sureclerini bloklamaz: belge yine indekslenip aranabilir
# kalir, sadece kategori/metadata insan incelemesi sonrasi guncellenir.
# Gercek API ile kalibre edildi (bkz. notebooks/08, bolum 3): net/coklu
# sinifli belgeler 0.85-0.98 bandinda, kasitli belirsiz bir metin 0.40'a
# dusuyor; 0.7 bu iki kumenin arasindaki bosluga denk geliyor.
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

SYSTEM_PROMPT_TEMPLATE = """Sen bir belge siniflandirma asistanisin. Sana bir belgenin metni verilecek.

Gorevin, belgeyi asagidaki JSON semasina uygun sekilde siniflandirmak ve etiketlemektir:

{{
  "siniflar": [string],   // Asagidaki listeden SECILMELI, belgeye uyan BIRDEN FAZLA sinif olabilir: {categories}
  "guven": number,        // 0.0-1.0 arasi, siniflandirmaya olan genel guvenin
  "etiketler": [string],  // Belgenin icerigini ozetleyen 2-5 serbest metin etiket (Turkce, kucuk harf)
  "gerekce": string       // Bu siniflari neden sectigine dair tek cumlelik kisa aciklama
}}

KURALLAR:
- Yanitin SADECE gecerli bir JSON nesnesi olmali.
- Markdown kod blogu (uc backtick), aciklama cumlesi veya baska hicbir metin EKLEME. Yanitin '{{' ile baslayip '}}' ile bitmeli.
- "siniflar" alani en az bir eleman icermeli ve her eleman MUTLAKA verilen listeden biri olmali; hicbiri uymuyorsa ["{fallback}"] kullan.
- Belge birden fazla kategoriye uyuyorsa (orn. hem fatura hem sozlesme icerikli ek), ilgili tum siniflari listele.
- "guven" dusukse ayri bir "belirsiz" sinifi UYDURMA; sadece guven degerini dusuk ver.
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
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> dict:
    """Belge metnini LLM ile siniflandirir ve etiketler.

    Args:
        text: OCR'dan gelen ham metin ya da yapilandirilmis alanlarin metne
            cevrilmis hali.
        categories: izin verilen sinif listesi (varsayilan: DEFAULT_CATEGORIES).
        model_name: kullanilacak Claude modeli.
        client: disaridan verilebilecek anthropic.Anthropic istemcisi
            (testlerde sahte/mock istemci vermek icin kullanislidir).
        confidence_threshold: "guven" bu degerin altinda kalirsa
            "human_review" True olarak isaretlenir.

    Returns:
        {"siniflar": list[str], "guven": float, "etiketler": list[str],
         "gerekce": str, "human_review": bool}
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

    siniflar = result.get("siniflar")
    if not isinstance(siniflar, list):
        siniflar = [siniflar] if siniflar else []
    valid_siniflar = [s for s in dict.fromkeys(siniflar) if s in categories]
    result["siniflar"] = valid_siniflar or [FALLBACK_CATEGORY]

    guven = result.get("guven")
    result["human_review"] = not isinstance(guven, (int, float)) or guven < confidence_threshold

    return result


def classify_chunks(chunks: list[dict], **kwargs) -> dict:
    """text_splitter.split_text() ciktisindaki chunk'lari birlestirip tek
    seferde siniflandirir.

    Chunk chunk degil belge butunu uzerinden siniflandirma yapmak, tek ve
    tutarli bir sinif/etiket kumesi elde etmek icin daha dogru sonuc verir.
    """
    full_text = "\n".join(c["text"] for c in chunks)
    return classify_document(full_text, **kwargs)


def attach_labels_to_chunks(chunks: list[dict], classifications: dict[str, dict]) -> list[dict]:
    """Belge bazli classify_document() ciktilarini, "source_doc" alani
    uzerinden ilgili chunk'lara meta veri olarak ekler.

    vector_store.build_index(), embedding disindaki tum anahtarlari oldugu
    gibi FAISS metadata'sina kopyaladigi icin, buradan donen chunk'lar
    embed_chunks() + build_index() zincirine verildiginde "siniflar",
    "guven", "etiketler" ve "human_review" alanlari da arama sonuclarinin
    bir parcasi olarak donmus olur.

    Bir belgenin siniflandirmasi henuz yapilmamissa (classifications
    sozlugunde yoksa) chunk yine de listede kalir ve human_review=True
    olarak isaretlenir; boylece inceleme bekleyen belgeler indeksten
    disarida birakilmaz, sadece insan onayi bekleyen olarak gorunur
    kalir.

    Args:
        chunks: text_splitter/embed_chunks ciktisi, her ogede "source_doc"
            (kaynak dosya adi) alani bulunmali.
        classifications: {source_doc: classify_document() ciktisi}.

    Returns:
        Her ogeye "siniflar", "guven", "etiketler", "human_review" eklenmis
        yeni bir liste.
    """
    labeled = []
    for chunk in chunks:
        classification = classifications.get(chunk["source_doc"])
        if classification is None:
            classification = {"siniflar": [FALLBACK_CATEGORY], "guven": None, "etiketler": [], "human_review": True}
        labeled.append({
            **chunk,
            "siniflar": classification.get("siniflar", [FALLBACK_CATEGORY]),
            "guven": classification.get("guven"),
            "etiketler": classification.get("etiketler", []),
            "human_review": classification.get("human_review", True),
        })
    return labeled
