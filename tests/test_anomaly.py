import anomaly


def _doc(belge_no=None, tutar=None):
    return {"alanlar": {"belge_no": belge_no, "tutar": tutar}}


def test_find_duplicate_document_numbers_finds_repeated_number():
    documents = {
        "a.png": _doc(belge_no="TF-001"),
        "b.png": _doc(belge_no="TF-001"),
        "c.png": _doc(belge_no="TF-002"),
    }
    duplicates = anomaly.find_duplicate_document_numbers(documents)
    assert duplicates == [{"belge_no": "TF-001", "documents": ["a.png", "b.png"]}]


def test_find_duplicate_document_numbers_ignores_missing_belge_no():
    documents = {"a.png": _doc(belge_no=None), "b.png": _doc(belge_no=None)}
    assert anomaly.find_duplicate_document_numbers(documents) == []


def test_find_duplicate_document_numbers_no_duplicates_returns_empty():
    documents = {"a.png": _doc(belge_no="TF-001"), "b.png": _doc(belge_no="TF-002")}
    assert anomaly.find_duplicate_document_numbers(documents) == []


def test_find_amount_outliers_below_min_sample_size_returns_empty():
    documents = {f"doc_{i}.png": _doc(tutar=f"{100 + i} TL") for i in range(3)}
    assert anomaly.find_amount_outliers(documents) == []


def test_find_amount_outliers_detects_clear_outlier():
    documents = {f"doc_{i}.png": _doc(tutar="100 TL") for i in range(6)}
    documents["outlier.png"] = _doc(tutar="100.000 TL")

    outliers = anomaly.find_amount_outliers(documents, z_threshold=1.5)

    assert len(outliers) == 1
    assert outliers[0]["source_doc"] == "outlier.png"
    assert outliers[0]["z_score"] > 0


def test_find_amount_outliers_ignores_unparseable_amounts():
    documents = {f"doc_{i}.png": _doc(tutar="100 TL") for i in range(5)}
    documents["belirsiz.png"] = _doc(tutar="bilinmiyor")
    # sadece 5 sayisallastirilabilir tutar var (MIN_SAMPLE_SIZE tam siniri),
    # hepsi ayni deger oldugu icin stdev=0 -> anomali hesaplanamaz.
    assert anomaly.find_amount_outliers(documents) == []


def test_find_amount_outliers_no_outliers_when_values_similar():
    documents = {f"doc_{i}.png": _doc(tutar=f"{100 + i} TL") for i in range(6)}
    assert anomaly.find_amount_outliers(documents, z_threshold=2.5) == []
