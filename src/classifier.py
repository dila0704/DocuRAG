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

JSON semasi artik prompt talimatina degil (DOC-34), client.generate_structured()
uzerinden API seviyesinde zorlaniyor -- AnthropicClient/OpenAIClient icin
gercek tool_use/function-calling, destegi olmayan saglayicilar (orn.
LocalHFClient) icin ise llm_factory.LLMClient.generate_structured()'daki
eski "JSON iste, bozuksa duzelt" fallback'ine otomatik duser (bkz.
llm_factory.py). Ayrica belge metni (OCR'dan geldigi icin GUVENILMEZ)
UNTRUSTED_CONTENT_NOTICE + wrap_untrusted() ile prompt injection'a karsi
isaretlenir (bkz. llm_json_utils.py).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from llm_factory import LLMClient, get_llm_client
from llm_json_utils import UNTRUSTED_CONTENT_NOTICE, wrap_untrusted

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

# generate_structured()'un AnthropicClient/OpenAIClient uzerinde API
# seviyesinde zorladigi sema (DOC-34). Fallback yolda (LocalHFClient)
# kullanilmaz ama JSON aciklamasi SYSTEM_PROMPT_TEMPLATE'te hala
# insan-okunur sekilde tekrarlanir (o yolda model semayi promptdan
# ogrenmek zorunda).
CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "siniflar": {"type": "array", "items": {"type": "string"}, "description": "Belgeye uyan sinif(lar), verilen listeden secilir."},
        "guven": {"type": "number", "description": "0.0-1.0 arasi genel guven skoru."},
        "etiketler": {"type": "array", "items": {"type": "string"}, "description": "2-5 serbest metin etiket (Turkce, kucuk harf)."},
        "gerekce": {"type": "string", "description": "Siniflandirma karari icin tek cumlelik kisa aciklama."},
    },
    "required": ["siniflar", "guven", "etiketler", "gerekce"],
}

SYSTEM_PROMPT_TEMPLATE = """Sen bir belge siniflandirma asistanisin. Sana bir belgenin metni verilecek.

Gorevin, belgeyi asagidaki semaya uygun sekilde siniflandirmak ve etiketlemektir:

{{
  "siniflar": [string],   // Asagidaki listeden SECILMELI, belgeye uyan BIRDEN FAZLA sinif olabilir: {categories}
  "guven": number,        // 0.0-1.0 arasi, siniflandirmaya olan genel guvenin
  "etiketler": [string],  // Belgenin icerigini ozetleyen 2-5 serbest metin etiket (Turkce, kucuk harf)
  "gerekce": string       // Bu siniflari neden sectigine dair tek cumlelik kisa aciklama
}}

KURALLAR:
- "siniflar" alani en az bir eleman icermeli ve her eleman MUTLAKA verilen listeden biri olmali; hicbiri uymuyorsa ["{fallback}"] kullan.
- Belge birden fazla kategoriye uyuyorsa (orn. hem fatura hem sozlesme icerikli ek), ilgili tum siniflari listele.
- "guven" dusukse ayri bir "belirsiz" sinifi UYDURMA; sadece guven degerini dusuk ver.
- Belgede olmayan bilgi UYDURMA.
{few_shot_block}{untrusted_notice}"""

USER_INSTRUCTION = "Bu belge metnini yukaridaki semaya gore siniflandir ve JSON olarak dondur."

# --- Few-shot geri besleme (DOC-34, "dogru" aktif ogrenme yerine dis-uygun
# bir karsiligi): Inceleme Kuyrugu'nda (app/views/review.py) bir insan bir
# siniflandirmayi duzelttiginde, o duzeltme burada JSONL olarak biriktirilir.
# classify_document(use_few_shot=True) verildiginde, en SON birkac duzeltme
# ornek olarak sistem promptuna eklenir -- boylece model AYNI hatayi tekrar
# etme riski azalir. Model agirliklarini DEGISTIRMEZ (gercek egitim/fine-tune
# degil), sadece in-context ogrenme uygular; bu yuzden "aktif ogrenme
# dongusu" degil, onun ucuz/dis-uygun bir karsiligi olarak sunulmali.
CORRECTIONS_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "human_corrections.jsonl"


def record_correction(source_doc: str, original: dict, corrected: dict, text_snippet: str = "", path: Path | None = None) -> None:
    """Bir insan duzeltmesini (Inceleme Kuyrugu'ndan) data/processed/human_corrections.jsonl'a
    ekler -- SADECE gercekten bir seyi degistiren duzeltmeler icin (siniflar
    veya etiketler orijinalden farkliysa); aksi halde gurultu birikir.

    Loglama basarisiz olursa (disk/izin) ana akisi KESMEZ, sadece uyari
    loglanir (llm_factory._append_usage_log ile ayni tolerans deseni).
    `path` verilmezse CORRECTIONS_PATH cagri aninda okunur (testler
    monkeypatch.setattr(classifier, "CORRECTIONS_PATH", ...) ile izole
    olabilsin diye)."""
    orig_siniflar = sorted(original.get("siniflar") or [])
    corrected_siniflar = sorted(corrected.get("siniflar") or [])
    orig_etiketler = sorted(original.get("etiketler") or [])
    corrected_etiketler = sorted(corrected.get("etiketler") or [])
    if orig_siniflar == corrected_siniflar and orig_etiketler == corrected_etiketler:
        return

    path = path or CORRECTIONS_PATH
    record = {
        "timestamp": time.time(),
        "source_doc": source_doc,
        "text_snippet": text_snippet[:600],
        "original_siniflar": orig_siniflar,
        "corrected_siniflar": corrected_siniflar,
        "corrected_etiketler": corrected_etiketler,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        logger.warning("record_correction: human_corrections.jsonl yazilamadi (atlandi).", exc_info=True)


def _load_recent_corrections(max_examples: int, path: Path | None = None) -> list[dict]:
    path = path or CORRECTIONS_PATH
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
    except OSError:
        logger.warning("_load_recent_corrections: human_corrections.jsonl okunamadi.", exc_info=True)
        return []
    usable = [r for r in records if r.get("text_snippet")]
    return usable[-max_examples:]


def _format_few_shot_block(corrections: list[dict]) -> str:
    if not corrections:
        return ""
    examples = []
    for r in corrections:
        examples.append(
            f'- Metin: "{r["text_snippet"][:200]}..." -> Dogru siniflar: {r["corrected_siniflar"]}, '
            f'dogru etiketler: {r["corrected_etiketler"]} (bir insan tarafindan {r["original_siniflar"]}\'dan duzeltildi)'
        )
    return (
        "\nGECMISTE INSAN TARAFINDAN DUZELTILEN BENZER ORNEKLER (referans al, birebir kopyalama):\n"
        + "\n".join(examples) + "\n"
    )


def classify_document(
    text: str,
    categories: list[str] | None = None,
    client: LLMClient | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_tokens: int = 512,
    temperature: float = DEFAULT_TEMPERATURE,
    max_json_attempts: int = DEFAULT_MAX_JSON_ATTEMPTS,
    use_few_shot: bool = False,
    few_shot_examples: int = 3,
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
        max_json_attempts: SADECE structured output desteklemeyen
            saglayicilarda (fallback yolu, bkz. llm_factory.LLMClient.
            generate_structured) kullanilir -- LLM gecersiz JSON dondurursa
            modelden duzeltmesi istenerek en fazla bu kadar deneme yapilir.
        use_few_shot: True ise, data/processed/human_corrections.jsonl'daki
            en son duzeltmeler (bkz. record_correction, app/views/review.py)
            sistem promptuna ornek olarak eklenir. Varsayilan KAPALI --
            duzeltme birikmemisse hicbir etkisi olmaz, birikmisse bile her
            cagriya prompt uzunlugu/maliyeti ekler.
        few_shot_examples: use_few_shot=True iken eklenecek en fazla ornek sayisi.

    Returns:
        {"siniflar": list[str], "guven": float, "etiketler": list[str],
         "gerekce": str, "human_review": bool}

    Raises:
        ValueError: text bossa.
        json.JSONDecodeError: max_json_attempts denemeden sonra da LLM
            gecerli JSON dondurmezse (sadece fallback yolda).
    """
    if not text or not text.strip():
        raise ValueError("text bos olamaz.")

    categories = categories or DEFAULT_CATEGORIES
    client = client or get_llm_client()

    few_shot_block = _format_few_shot_block(_load_recent_corrections(few_shot_examples)) if use_few_shot else ""
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        categories=", ".join(categories), fallback=FALLBACK_CATEGORY,
        few_shot_block=few_shot_block, untrusted_notice=UNTRUSTED_CONTENT_NOTICE,
    )

    logger.info("classify_document basladi: metin_uzunlugu=%d kategori_sayisi=%d", len(text), len(categories))
    result = client.generate_structured(
        system_prompt=system_prompt,
        user_message=f"{USER_INSTRUCTION}\n\n---\n{wrap_untrusted(text.strip())}\n---",
        schema=CLASSIFICATION_SCHEMA,
        tool_name="classify_document",
        tool_description="Belgeyi siniflandirir, guven skoru ve etiketler atar.",
        max_tokens=max_tokens,
        temperature=temperature,
        max_json_attempts=max_json_attempts,
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
