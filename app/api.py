"""DocuRAG icin ayri/ek bir REST servisi (DOC-30 Oncelik: FastAPI iskeleti).

Streamlit uygulamasina (app/app.py) DOKUNMAZ ve ondan tamamen bagimsiz
calisir -- Streamlit hala src/ modullerini dogrudan import ediyor, bu dosya
AYNI src/ fonksiyonlarini (pipeline, vector_store) saran, paralel bir HTTP
katmani. Calistirma:

    uvicorn app.api:app --reload

Otomatik OpenAPI/Swagger dokumantasyonu /docs adresinde acilir.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pipeline  # noqa: E402
import vector_store  # noqa: E402

UPLOAD_DIR = Path("data") / "raw_docs" / "uploads"

app = FastAPI(
    title="DocuRAG API",
    description="Streamlit arayuzune ek, src/ pipeline'ini saran REST servisi.",
    version="1.0",
)


class ClassificationResponse(BaseModel):
    siniflar: list[str] = []
    guven: float | None = None
    etiketler: list[str] = []
    human_review: bool = True


class FieldsResponse(BaseModel):
    tarih: str | None = None
    tutar: str | None = None
    taraflar: list[str] = []
    belge_no: str | None = None
    konu: str | None = None


class DocumentIngestResponse(BaseModel):
    source_doc: str
    num_chunks: int
    classification: ClassificationResponse
    fields: FieldsResponse = FieldsResponse()


class SearchResultItem(BaseModel):
    score: float
    score_type: str = "cosine"
    text: str
    source_doc: str
    siniflar: list[str] = []
    guven: float | None = None
    etiketler: list[str] = []
    human_review: bool = False


class DocumentSummary(BaseModel):
    source_doc: str
    siniflar: list[str] = []
    guven: float | None = None
    human_review: bool = False
    ingested_at: str | None = None
    chunk_count: int


@app.post("/documents", response_model=DocumentIngestResponse)
def ingest_document(image: UploadFile) -> DocumentIngestResponse:
    """Bir belge gorseli yukleyip OCR->chunking->siniflandirma->embedding->
    FAISS indeksleme zincirini calistirir (pipeline.ingest_document)."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = UPLOAD_DIR / image.filename

    with open(dest_path, "wb") as f:
        f.write(image.file.read())

    try:
        result = pipeline.ingest_document(str(dest_path))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Belge islenemedi: {exc}") from exc

    return DocumentIngestResponse(**result)


@app.get("/search", response_model=list[SearchResultItem])
def search(
    q: str,
    top_k: int = 5,
    use_hybrid: bool = True,
    use_reranker: bool = False,
    expand_query: bool = False,
) -> list[SearchResultItem]:
    """Dogal dil sorgusuyla FAISS index'inde semantik arama yapar (pipeline.search_documents).

    use_hybrid=True (varsayilan): dense+BM25 RRF fusion. False: eski salt-dense davranis.
    use_reranker=True: RRF sonrasi cross-encoder ile yeniden siralama (daha yavas).
    expand_query=True: HyDE ile sorgu genisletme. NOT: cok-turlu (multi-turn) baglam
    yogunlastirma REST API'de desteklenmiyor -- bu stateless bir servis, konusma
    gecmisini tutmak/gondermek istemcinin sorumlulugunda (bkz. Streamlit'teki
    session_state["conversation_history"], app/views/search.py)."""
    try:
        results = pipeline.search_documents(
            q, top_k=top_k, use_hybrid=use_hybrid, use_reranker=use_reranker, expand_query=expand_query,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Henuz indekslenmis belge yok.")
    return [SearchResultItem(**r) for r in results]


@app.get("/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    """Indekslenen tum belgelerin envanterini dondurur (vector_store.group_latest_by_source_doc)."""
    index_path = vector_store.load_index_path()
    try:
        _, metadata = vector_store.load_index(index_path)
    except FileNotFoundError:
        return []

    documents = vector_store.group_latest_by_source_doc(metadata)
    chunk_counts: dict[str, int] = {}
    for m in metadata.values():
        doc = m.get("source_doc", "bilinmiyor")
        chunk_counts[doc] = chunk_counts.get(doc, 0) + 1

    return [
        DocumentSummary(
            source_doc=doc,
            siniflar=info.get("siniflar", []),
            guven=info.get("guven"),
            human_review=info.get("human_review", False),
            ingested_at=info.get("ingested_at"),
            chunk_count=chunk_counts.get(doc, 0),
        )
        for doc, info in documents.items()
    ]


@app.delete("/documents/{source_doc}")
def delete_document(source_doc: str) -> dict:
    """Bir belgeye ait tum vektorleri/metadata'yi index'i yeniden kurmadan siler
    (vector_store.delete_by_source_doc)."""
    index_path = vector_store.load_index_path()
    try:
        index, metadata = vector_store.load_index(index_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Henuz indekslenmis belge yok.")

    if not any(m.get("source_doc") == source_doc for m in metadata.values()):
        raise HTTPException(status_code=404, detail=f"'{source_doc}' bulunamadi.")

    updated_index, updated_metadata = vector_store.delete_by_source_doc(index, metadata, source_doc)
    vector_store.save_index(updated_index, updated_metadata, index_path)
    return {"source_doc": source_doc, "deleted": True}
