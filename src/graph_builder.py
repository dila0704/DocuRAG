"""
Belge iliski grafigi olusturma modulu (DOC-30, "wow" ozellik seti B3).

field_extractor.py'nin cikardigi "taraflar" alanina dayanir: iki belge
arasinda en az bir ortak taraf (kisi/kurum ismi) varsa aralarinda bir kenar
(edge) kurulur. LLM cagirmaz -- tamamen deterministik, anomaly.py ile ayni
"kod-tabanli, prompt'a bagimli degil" ilkesini paylasir.
"""
from __future__ import annotations

import logging

import networkx as nx

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def build_document_graph(documents: dict[str, dict]) -> nx.Graph:
    """vector_store.group_latest_by_source_doc() ciktisindan (dict[source_doc,
    metadata]) bir iliski grafigi kurar.

    Dugumler: source_doc. Kenarlar: iki belgenin "alanlar.taraflar" listesi
    kesisiyorsa (en az bir ortak isim), agirlik=ortak isim sayisi,
    "shared"=ortak isimlerin siralanmis listesi.
    """
    graph = nx.Graph()
    parties_by_doc: dict[str, set[str]] = {}

    for source_doc, info in documents.items():
        graph.add_node(source_doc)
        parties_by_doc[source_doc] = set((info.get("alanlar") or {}).get("taraflar") or [])

    docs = list(documents.keys())
    for i, doc_a in enumerate(docs):
        for doc_b in docs[i + 1:]:
            shared = parties_by_doc[doc_a] & parties_by_doc[doc_b]
            if shared:
                graph.add_edge(doc_a, doc_b, weight=len(shared), shared=sorted(shared))

    logger.info(
        "build_document_graph: %d dugum, %d kenar.", graph.number_of_nodes(), graph.number_of_edges(),
    )
    return graph
