import json

import pytest

import field_extractor


def _response(**fields):
    base = {"tarih": None, "tutar": None, "taraflar": [], "belge_no": None, "konu": None}
    base.update(fields)
    return json.dumps(base, ensure_ascii=False)


def test_extract_fields_happy_path(fake_llm_client):
    client = fake_llm_client([_response(tarih="12.08.2026", tutar="1.234,56 TL", taraflar=["Dila Alpay"], belge_no="TF-001", konu="Laptop talebi")])
    result = field_extractor.extract_fields("bir belge metni", client=client)

    assert result["tarih"] == "12.08.2026"
    assert result["tutar"] == "1.234,56 TL"
    assert result["taraflar"] == ["Dila Alpay"]
    assert result["belge_no"] == "TF-001"
    assert result["konu"] == "Laptop talebi"


def test_extract_fields_missing_fields_default_to_none_or_empty(fake_llm_client):
    client = fake_llm_client([_response()])
    result = field_extractor.extract_fields("metin", client=client)
    assert result == {"tarih": None, "tutar": None, "taraflar": [], "belge_no": None, "konu": None}


def test_extract_fields_invalid_type_falls_back_to_default(fake_llm_client):
    client = fake_llm_client([json.dumps({"tarih": 123, "tutar": None, "taraflar": "tek string", "belge_no": None, "konu": None})])
    result = field_extractor.extract_fields("metin", client=client)
    assert result["tarih"] is None
    assert result["taraflar"] == []


def test_extract_fields_filters_non_string_items_in_taraflar(fake_llm_client):
    client = fake_llm_client([json.dumps({"tarih": None, "tutar": None, "taraflar": ["Dila", 5, ""], "belge_no": None, "konu": None})])
    result = field_extractor.extract_fields("metin", client=client)
    assert result["taraflar"] == ["Dila"]


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_extract_fields_empty_text_raises_value_error(text, fake_llm_client):
    client = fake_llm_client([_response()])
    with pytest.raises(ValueError):
        field_extractor.extract_fields(text, client=client)


def test_attach_fields_to_chunks_applies_matching_fields():
    chunks = [{"chunk_id": 0, "text": "x", "source_doc": "a.png"}]
    fields_by_doc = {"a.png": {"tarih": "12.08.2026", "tutar": None, "taraflar": [], "belge_no": None, "konu": "Talep"}}
    labeled = field_extractor.attach_fields_to_chunks(chunks, fields_by_doc)
    assert labeled[0]["alanlar"]["konu"] == "Talep"


def test_attach_fields_to_chunks_defaults_missing_doc_to_empty_fields():
    chunks = [{"chunk_id": 0, "text": "x", "source_doc": "unknown.png"}]
    labeled = field_extractor.attach_fields_to_chunks(chunks, {})
    assert labeled[0]["alanlar"] == {"tarih": None, "tutar": None, "taraflar": [], "belge_no": None, "konu": None}


@pytest.mark.parametrize("text,expected", [
    ("1.234,56 TL", 1234.56),
    ("500 TL", 500.0),
    ("2.500 TL", 2500.0),
    (None, None),
    ("belirsiz", None),
])
def test_parse_turkish_amount(text, expected):
    assert field_extractor.parse_turkish_amount(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("12.08.2026", "2026-08-12"),
    ("2026-08-12", "2026-08-12"),
    (None, None),
    ("belirsiz tarih", None),
])
def test_parse_date_to_iso(text, expected):
    assert field_extractor.parse_date_to_iso(text) == expected


def test_build_amount_range_filter_excludes_unparseable_amount():
    filter_fn = field_extractor.build_amount_range_filter(100.0, 2000.0)
    assert filter_fn({"alanlar": {"tutar": "1.500 TL"}}) is True
    assert filter_fn({"alanlar": {"tutar": None}}) is False
    assert filter_fn({"alanlar": {"tutar": "5 TL"}}) is False


def test_build_date_range_filter_excludes_out_of_range():
    filter_fn = field_extractor.build_date_range_filter("2026-01-01", "2026-12-31")
    assert filter_fn({"alanlar": {"tarih": "12.08.2026"}}) is True
    assert filter_fn({"alanlar": {"tarih": "15.03.2025"}}) is False
    assert filter_fn({"alanlar": {"tarih": None}}) is False
