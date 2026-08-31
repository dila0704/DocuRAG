import graph_builder


def _doc(taraflar=None):
    return {"alanlar": {"taraflar": taraflar or []}}


def test_build_document_graph_adds_edge_for_shared_party():
    documents = {
        "a.png": _doc(["Dila Alpay", "ABC A.Ş."]),
        "b.png": _doc(["Dila Alpay"]),
        "c.png": _doc(["Mehmet Demir"]),
    }
    graph = graph_builder.build_document_graph(documents)

    assert set(graph.nodes()) == {"a.png", "b.png", "c.png"}
    assert graph.has_edge("a.png", "b.png")
    assert not graph.has_edge("a.png", "c.png")
    assert graph["a.png"]["b.png"]["shared"] == ["Dila Alpay"]


def test_build_document_graph_no_shared_parties_no_edges():
    documents = {"a.png": _doc(["X"]), "b.png": _doc(["Y"])}
    graph = graph_builder.build_document_graph(documents)
    assert graph.number_of_edges() == 0


def test_build_document_graph_empty_documents_returns_empty_graph():
    graph = graph_builder.build_document_graph({})
    assert graph.number_of_nodes() == 0


def test_build_document_graph_missing_alanlar_treated_as_no_parties():
    documents = {"a.png": {}, "b.png": {}}
    graph = graph_builder.build_document_graph(documents)
    assert graph.number_of_edges() == 0
