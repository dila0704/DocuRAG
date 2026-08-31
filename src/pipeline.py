"""
Uctan uca DocuRAG orkestrasyon modulu (DOC-31).

Su ana kadar bagimsiz notebook'larda (01-12) dogrulanan adimlari
(OCR -> chunking -> embedding -> siniflandirma -> FAISS indeksleme ->
semantik arama) src/ altindaki modulleri sirayla cagirarak tek bir
komutla calistirilabilir hale getirir:

    python pipeline.py ingest data/raw_docs/test_talep_01.png
    python pipeline.py search "laptop talebi"

Notebook'lar hala gecerlidir (adim adim deney/dogrulama icin), ama bir
kullanicinin/CI'in tum zinciri calistirmasi icin artik bu modul yeterlidir.
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from typing import Callable

import classifier
import embedder
import field_extractor
import ocr
import query_rewriter
import retrieval
import text_splitter
import vector_store

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def ingest_document(
    image_path: str,
    index_path: str | None = None,
    on_step: Callable[[str, dict], None] | None = None,
) -> dict:
    """Bir belge gorselini uctan uca isler.

    Adimlar: OCR -> chunking -> siniflandirma + alan cikarimi (chunk'lara
    meta veri olarak eklenir) -> embedding -> FAISS index'e ekleme (index
    yoksa yeni kurulur, varsa add_chunks() ile genisletilir, index sifirdan
    kurulmaz).

    Args:
        image_path: islenecek belge gorselinin yolu.
        index_path: FAISS index dosyalarinin (uzantisiz) yolu. None ise
            config/settings.yaml -> vector_db.path kullanilir.
        on_step: verilirse, her adimdan HEMEN SONRA `on_step(step_name, info)`
            cagrilir (step_name: "ocr"/"chunking"/"classification"/
            "field_extraction"/"embedding"/"indexing"). Streamlit'in
            st.status() ile canli ilerleme gostermesi icin eklendi (bkz.
            app/views/upload.py) -- varsayilan None ise davranis TAMAMEN
            eskisiyle aynidir (geriye donuk uyumlu).

    Returns:
        {"source_doc": str, "num_chunks": int, "classification": dict, "fields": dict}
    """
    index_path = index_path or vector_store.load_index_path()
    source_doc = os.path.basename(image_path)

    logger.info("ingest_document basladi: %s", source_doc)

    raw_text = ocr.extract_text_from_image(image_path)
    if on_step:
        on_step("ocr", {"raw_text": raw_text, "char_count": len(raw_text)})

    chunks = text_splitter.split_text(raw_text)
    if not chunks:
        raise ValueError(f"'{image_path}' icin OCR sonucu bos/anlamli metin uretmedi.")

    for chunk in chunks:
        chunk["source_doc"] = source_doc
    if on_step:
        on_step("chunking", {"chunk_count": len(chunks)})

    classification = classifier.classify_chunks(chunks)
    labeled_chunks = classifier.attach_labels_to_chunks(chunks, {source_doc: classification})
    if on_step:
        on_step("classification", {"classification": classification})

    fields = field_extractor.extract_fields(raw_text)
    labeled_chunks = field_extractor.attach_fields_to_chunks(labeled_chunks, {source_doc: fields})
    if on_step:
        on_step("field_extraction", {"fields": fields})

    now_iso = datetime.now(timezone.utc).isoformat()
    for chunk in labeled_chunks:
        chunk["ingested_at"] = now_iso

    embedded_chunks = embedder.embed_chunks(labeled_chunks)
    if on_step:
        on_step("embedding", {"chunk_count": len(embedded_chunks)})

    if os.path.exists(index_path + ".faiss"):
        index, metadata = vector_store.load_index(index_path)
        index, metadata = vector_store.add_chunks(index, metadata, embedded_chunks)
    else:
        index, metadata = vector_store.build_index(embedded_chunks)

    vector_store.save_index(index, metadata, index_path)
    if on_step:
        on_step("indexing", {"total_chunks": len(metadata)})

    logger.info(
        "ingest_document tamamlandi: %s (%d chunk, siniflar=%s, human_review=%s, alanlar=%s)",
        source_doc, len(chunks), classification.get("siniflar"), classification.get("human_review"), fields,
    )
    return {"source_doc": source_doc, "num_chunks": len(chunks), "classification": classification, "fields": fields}


def search_documents(
    query: str,
    top_k: int = 5,
    index_path: str | None = None,
    use_hybrid: bool = True,
    bm25_index: tuple | None = None,
    use_reranker: bool = False,
    expand_query: bool = False,
    metadata_filter=None,
) -> list[dict]:
    """Dogal dil sorgusuyla FAISS index'inde semantik arama yapar.

    Args:
        query: dogal dil sorgusu.
        top_k: dondurulecek azami sonuc sayisi.
        index_path: None ise config/settings.yaml -> vector_db.path kullanilir.
        use_hybrid: True (varsayilan) ise dense (FAISS) + BM25 siralamalari
            RRF ile birlestirilir (retrieval.hybrid_search, bkz. DOC-30 A1).
            False verilirse eski salt-dense davranisina (vector_store.search)
            donulur -- karsilastirma/test edilebilirlik icin.
        bm25_index: onceden kurulmus (BM25Okapi, id_order) cifti (orn.
            app/data_access.get_bm25_index()'in onbellegi) -- verilirse BM25
            indeksi bu cagrida sifirdan kurulmaz. use_hybrid=False iken
            yoksayilir.
        use_reranker: True ise RRF sonrasi ilk (top_k*4, en az 20) aday bir
            cross-encoder ile yeniden siralanir (bkz. retrieval.rerank()).
            Varsayilan KAPALI -- yeni model indirme/inference gecikmesi
            varsayilan arama akisini yavaslatmasin diye. use_hybrid=False
            iken yoksayilir (reranking su an sadece hybrid_search uzerinden).
        expand_query: True ise sorgu, embed edilmeden once HyDE (query_rewriter.
            generate_hypothetical_answer) ile genisletilir: embed edilen metin
            "{sorgu}\\n{varsayimsal_cevap}" olur (orijinal sorgu terimleri
            kaybolmasin diye "saf" HyDE yerine bu "hybrid" yaklasim
            kullanilir). Varsayilan KAPALI -- her aramaya bir LLM cagrisi
            (ve gecikmesi) ekliyor.
        metadata_filter: verilirse (bkz. field_extractor.build_amount_range_filter()/
            build_date_range_filter(), ya da basit bir lambda), SADECE bu
            fonksiyonun True dondugu chunk'lar sonuca girer. Hem hybrid hem
            dense-only yolda desteklenir.

    Returns:
        use_hybrid=True: retrieval.hybrid_search() ciktisi (score alani RRF
        skorudur, "score_type": "rrf" iceerir).
        use_hybrid=False: vector_store.search() ciktisi (score alani cosine
        benzerligidir): [{"score": float, **chunk_metadata}, ...]
    """
    index_path = index_path or vector_store.load_index_path()
    index, metadata = vector_store.load_index(index_path)

    # NOT: expand_query=True olsa bile BM25/rerank/loglama icin hala
    # ORIJINAL `query` kullanilir -- sadece EMBED EDILEN metin genisletilir
    # (embed_text). HyDE'nin varsayimsal/gurultulu metni anahtar kelime
    # eslesmesini/cross-encoder skorunu bozmasin diye.
    embed_text = query
    if expand_query:
        hypothetical = query_rewriter.generate_hypothetical_answer(query)
        embed_text = f"{query}\n{hypothetical}"

    query_chunk = {"chunk_id": 0, "text": embed_text, "token_count": text_splitter.count_tokens(embed_text)}
    query_embedding = embedder.embed_chunks([query_chunk])[0]["embedding"]

    if use_hybrid:
        rerank_top_n = max(top_k * 4, 20) if use_reranker else None
        results = retrieval.hybrid_search(
            index, metadata, query, query_embedding, top_k=top_k,
            bm25_index=bm25_index, rerank_top_n=rerank_top_n, metadata_filter=metadata_filter,
        )
    else:
        results = vector_store.search(
            index, metadata, query_embedding, top_k=top_k, metadata_filter=metadata_filter,
        )
    logger.info("search_documents: sorgu=%r hybrid=%s -> %d sonuc.", query, use_hybrid, len(results))
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description="DocuRAG uctan uca pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Bir belge gorselini isleyip index'e ekler")
    ingest_parser.add_argument("image_path")

    search_parser = subparsers.add_parser("search", help="Index uzerinde semantik arama yapar")
    search_parser.add_argument("query")
    search_parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "ingest":
        result = ingest_document(args.image_path)
        print(result)
    elif args.command == "search":
        for rank, item in enumerate(search_documents(args.query, top_k=args.top_k), start=1):
            print(f"{rank}. [{item['score']:.3f}] {item.get('source_doc')}: {item['text'][:120]}")


if __name__ == "__main__":
    _main()
