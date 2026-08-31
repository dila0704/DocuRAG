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

LLM baglantisi dogrudan degil, llm_factory.get_llm_client() Factory'si
uzerinden kuruluyor (DOC-27): hangi saglayicinin (Anthropic/OpenAI/yerel
huggingface) kullanilacagi config/settings.yaml -> llm_settings.active_mode
tarafindan belirlenir, bu modul degistirilmeden cloud/local arasinda gecis
yapilabilir.
"""
from __future__ import annotations

import logging

from llm_factory import LLMClient, get_llm_client
from llm_json_utils import generate_and_parse_json as _generate_and_parse_json

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_CATEGORIES = ["fatura", "sözleşme", "dilekçe", "talep formu", "diğer"]

# classify_document(), LLM gecerli JSON dondurmezse bu kadar deneme yapar
# (ilk deneme + tekrar istekler). Kucuk/zayif local modellerde (bkz.
# notebooks/12) JSON formatinin bozulmasi beklenen bir durum oldugu icin,
# hatali yanit dogrudan cagirana firlatilmadan once modelden duzeltmesi
# istenir (DOC-31).
DEFAULT_MAX_JSON_ATTEMPTS = 2

# classify_document() varsayilan olarak deterministik (temperature=0) calisir:
# siniflandirma gibi tekrarlanabilirlik gereken bir gorevde ayni belgenin
# farkli calistirmalarda farkli sinif/guven skoru uretmesi istenmez (DOC-31).
DEFAULT_TEMPERATURE = 0.0

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


def classify_document(
    text: str,
    categories: list[str] | None = None,
    client: LLMClient | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_tokens: int = 512,
    temperature: float = DEFAULT_TEMPERATURE,
    max_json_attempts: int = DEFAULT_MAX_JSON_ATTEMPTS,
) -> dict:
    """Belge metnini LLM ile siniflandirir ve etiketler.

    Args:
        text: OCR'dan gelen ham metin ya da yapilandirilmis alanlarin metne
            cevrilmis hali.
        categories: izin verilen sinif listesi (varsayilan: DEFAULT_CATEGORIES).
        client: disaridan verilebilecek LLMClient (testlerde sahte/mock
            istemci vermek icin kullanislidir). None ise config/settings.yaml
            -> llm_settings.active_mode'a gore get_llm_client() Factory'si
            uzerinden kurulur (bkz. llm_factory.py, DOC-26/DOC-27).
        confidence_threshold: "guven" bu degerin altinda kalirsa
            "human_review" True olarak isaretlenir.
        max_tokens: LLM yanitindan beklenen azami token sayisi. Cok sayida
            "etiketler"/uzun "gerekce" iceren yanitlar icin varsayilan 512
            yetersiz kalirsa artirilabilir.
        temperature: LLM'e gonderilen sicaklik degeri. Varsayilan 0.0
            (deterministik): siniflandirma sonucunun ayni belge icin
            calistirmalar arasinda tutarli kalmasi hedeflenir.
        max_json_attempts: LLM gecersiz JSON dondurursa (kucuk/zayif local
            modellerde gorulebilir, bkz. notebooks/12) modelden duzeltmesi
            istenerek en fazla bu kadar deneme yapilir.

    Returns:
        {"siniflar": list[str], "guven": float, "etiketler": list[str],
         "gerekce": str, "human_review": bool}

    Raises:
        ValueError: text bossa.
        json.JSONDecodeError: max_json_attempts denemeden sonra da LLM
            gecerli JSON dondurmezse.
    """
    if not text or not text.strip():
        raise ValueError("text bos olamaz.")

    categories = categories or DEFAULT_CATEGORIES
    client = client or get_llm_client()

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        categories=", ".join(categories), fallback=FALLBACK_CATEGORY
    )

    logger.info("classify_document basladi: metin_uzunlugu=%d kategori_sayisi=%d", len(text), len(categories))
    result = _generate_and_parse_json(
        client=client,
        system_prompt=system_prompt,
        user_message=f"{USER_INSTRUCTION}\n\n---\n{text.strip()}\n---",
        max_tokens=max_tokens,
        temperature=temperature,
        max_json_attempts=max_json_attempts,
        caller_name="classify_document",
    )

    siniflar = result.get("siniflar")
    if not isinstance(siniflar, list):
        siniflar = [siniflar] if siniflar else []
    valid_siniflar = [s for s in dict.fromkeys(siniflar) if s in categories]
    invalid_siniflar = [s for s in siniflar if s not in categories]
    if invalid_siniflar:
        logger.warning("classify_document: taninmayan sinif(lar) elendi: %s", invalid_siniflar)
    result["siniflar"] = valid_siniflar or [FALLBACK_CATEGORY]

    guven = result.get("guven")
    result["human_review"] = not isinstance(guven, (int, float)) or guven < confidence_threshold

    logger.info(
        "classify_document tamamlandi: siniflar=%s guven=%s human_review=%s",
        result["siniflar"], guven, result["human_review"],
    )
    return result


def classify_chunks(chunks: list[dict], **kwargs) -> dict:
    """text_splitter.split_text() ciktisindaki chunk'lari birlestirip tek
    seferde siniflandirir.

    Chunk chunk degil belge butunu uzerinden siniflandirma yapmak, tek ve
    tutarli bir sinif/etiket kumesi elde etmek icin daha dogru sonuc verir.
    """
    logger.info("classify_chunks: %d chunk birlestirilip siniflandirilacak.", len(chunks))
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
    missing_docs = set()
    for chunk in chunks:
        classification = classifications.get(chunk["source_doc"])
        if classification is None:
            missing_docs.add(chunk["source_doc"])
            classification = {"siniflar": [FALLBACK_CATEGORY], "guven": None, "etiketler": [], "human_review": True}
        labeled.append({
            **chunk,
            "siniflar": classification.get("siniflar", [FALLBACK_CATEGORY]),
            "guven": classification.get("guven"),
            "etiketler": classification.get("etiketler", []),
            "human_review": classification.get("human_review", True),
        })
    if missing_docs:
        logger.info("attach_labels_to_chunks: siniflandirmasi olmayan %d belge human_review=True isaretlendi.", len(missing_docs))
    return labeled
