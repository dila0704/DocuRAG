"""
Kural tabanli (LLM'siz) anomali tespiti modulu (DOC-30, "wow" ozellik seti B2).

field_extractor.py'nin cikardigi yapilandirilmis alanlara (DOC-30 B1)
dayanir: ayni belge numarasinin tekrari, tutar aykiri degerleri gibi basit
istatistiksel kontroller yapar. Bilerek LLM CAGIRMAZ -- tamamen deterministik
ve kod-tabanli calisir, projenin "grounding kodda dogrulanir, promptta degil"
ilkesinin dogal bir uzantisidir (bkz. answer._enforce_grounding).

Girdi olarak vector_store.group_latest_by_source_doc() ciktisiyla ayni
sekli (dict[source_doc, metadata]) bekler; metadata icinde "alanlar" (bkz.
field_extractor.attach_fields_to_chunks) alani bulunmali.
"""
from __future__ import annotations

import logging
import statistics

from field_extractor import parse_turkish_amount

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Z-skor anlamli olsun diye en az bu kadar sayisallastirilabilir tutar
# gerekir; altinda yanıltıcı/sahte bir anomali UYDURULMAZ, bos liste doner.
MIN_SAMPLE_SIZE = 5

DEFAULT_Z_THRESHOLD = 2.5


def find_duplicate_document_numbers(documents: dict[str, dict]) -> list[dict]:
    """Ayni "belge_no"ya sahip birden fazla belge varsa (farkli source_doc'larda)
    her grubu raporlar.

    Returns:
        [{"belge_no": str, "documents": [source_doc, ...]}, ...]
    """
    by_number: dict[str, list[str]] = {}
    for source_doc, info in documents.items():
        belge_no = (info.get("alanlar") or {}).get("belge_no")
        if not belge_no:
            continue
        by_number.setdefault(belge_no, []).append(source_doc)

    duplicates = [
        {"belge_no": belge_no, "documents": sorted(docs)}
        for belge_no, docs in by_number.items()
        if len(docs) > 1
    ]
    if duplicates:
        logger.info("find_duplicate_document_numbers: %d tekrarlanan belge numarasi bulundu.", len(duplicates))
    return duplicates


def find_amount_outliers(documents: dict[str, dict], z_threshold: float = DEFAULT_Z_THRESHOLD) -> list[dict]:
    """Tutari sayisallastirilabilen belgeler arasinda basit bir z-skor
    (ortalamadan kac standart sapma uzakta) hesaplayip esigi asanlari
    raporlar.

    Ornek sayisi MIN_SAMPLE_SIZE'in altindaysa (istatistik anlamsizlasir)
    BOS LISTE doner -- kucuk bir ornekte "anomali" UYDURULMAZ.

    Returns:
        [{"source_doc": str, "tutar": float, "z_score": float}, ...]
        (en yuksek |z_score|'dan en dusuge siralanmis)
    """
    amounts_by_doc: dict[str, float] = {}
    for source_doc, info in documents.items():
        amount = parse_turkish_amount((info.get("alanlar") or {}).get("tutar"))
        if amount is not None:
            amounts_by_doc[source_doc] = amount

    if len(amounts_by_doc) < MIN_SAMPLE_SIZE:
        logger.info(
            "find_amount_outliers: sadece %d sayisallastirilabilir tutar var (min %d gerekli), anomali hesaplanmadi.",
            len(amounts_by_doc), MIN_SAMPLE_SIZE,
        )
        return []

    values = list(amounts_by_doc.values())
    mean = statistics.mean(values)
    stdev = statistics.stdev(values)
    if stdev == 0:
        return []

    outliers = []
    for source_doc, amount in amounts_by_doc.items():
        z_score = (amount - mean) / stdev
        if abs(z_score) >= z_threshold:
            outliers.append({"source_doc": source_doc, "tutar": amount, "z_score": z_score})

    # mypy: outliers'daki dict'ler karma deger tipli (str + float) oldugu icin
    # "z_score" object olarak cikarsanir; degerin gercekte her zaman float
    # oldugu biliniyor (yukarida atandi).
    outliers.sort(key=lambda o: abs(o["z_score"]), reverse=True)  # type: ignore[arg-type]
    if outliers:
        logger.info("find_amount_outliers: %d tutar aykiri degeri bulundu.", len(outliers))
    return outliers
