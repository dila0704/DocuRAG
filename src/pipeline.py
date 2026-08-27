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

import classifier
import embedder
import ocr
import text_splitter
import vector_store

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def ingest_document(image_path: str, index_path: str | None = None) -> dict:
    """Bir belge gorselini uctan uca isler.

    Adimlar: OCR -> chunking -> siniflandirma (chunk'lara etiket olarak
    eklenir) -> embedding -> FAISS index'e ekleme (index yoksa yeni kurulur,
    varsa add_chunks() ile genisletilir, index sifirdan kurulmaz).

    Args:
        image_path: islenecek belge gorselinin yolu.
        index_path: FAISS index dosyalarinin (uzantisiz) yolu. None ise
            config/settings.yaml -> vector_db.path kullanilir.

    Returns:
        {"source_doc": str, "num_chunks": int, "classification": dict}
    """
    index_path = index_path or vector_store.load_index_path()
    source_doc = os.path.basename(image_path)

    logger.info("ingest_document basladi: %s", source_doc)

    raw_text = ocr.extract_text_from_image(image_path)
    chunks = text_splitter.split_text(raw_text)
    if not chunks:
        raise ValueError(f"'{image_path}' icin OCR sonucu bos/anlamli metin uretmedi.")

    for chunk in chunks:
        chunk["source_doc"] = source_doc

    classification = classifier.classify_chunks(chunks)
    labeled_chunks = classifier.attach_labels_to_chunks(chunks, {source_doc: classification})
    embedded_chunks = embedder.embed_chunks(labeled_chunks)

    if os.path.exists(index_path + ".faiss"):
        index, metadata = vector_store.load_index(index_path)
        index, metadata = vector_store.add_chunks(index, metadata, embedded_chunks)
    else:
        index, metadata = vector_store.build_index(embedded_chunks)

    vector_store.save_index(index, metadata, index_path)

    logger.info(
        "ingest_document tamamlandi: %s (%d chunk, siniflar=%s, human_review=%s)",
        source_doc, len(chunks), classification.get("siniflar"), classification.get("human_review"),
    )
    return {"source_doc": source_doc, "num_chunks": len(chunks), "classification": classification}


def search_documents(query: str, top_k: int = 5, index_path: str | None = None) -> list[dict]:
    """Dogal dil sorgusuyla FAISS index'inde semantik arama yapar.

    Args:
        query: dogal dil sorgusu.
        top_k: dondurulecek azami sonuc sayisi.
        index_path: None ise config/settings.yaml -> vector_db.path kullanilir.

    Returns:
        vector_store.search() ciktisi: [{"score": float, **chunk_metadata}, ...]
    """
    index_path = index_path or vector_store.load_index_path()
    index, metadata = vector_store.load_index(index_path)

    query_chunk = {"chunk_id": 0, "text": query, "token_count": text_splitter.count_tokens(query)}
    query_embedding = embedder.embed_chunks([query_chunk])[0]["embedding"]

    results = vector_store.search(index, metadata, query_embedding, top_k=top_k)
    logger.info("search_documents: sorgu=%r -> %d sonuc.", query, len(results))
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
