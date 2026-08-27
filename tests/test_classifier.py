import json

import pytest

import classifier
from llm_factory import LLMClient


class FakeClient(LLMClient):
    """Sirali sahte yanitlar donduren, cagrilari kaydeden test istemcisi."""

    def __init__(self, responses, model_name: str = "fake"):
        self.model_name = model_name
        self._responses = list(responses)
        self.calls: list[dict] = []

    def _generate(self, system_prompt, user_message, max_tokens, temperature):
        self.calls.append({
            "system_prompt": system_prompt,
            "user_message": user_message,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return self._responses.pop(0)


def _valid_response(siniflar=None, guven=0.9, etiketler=None):
    return json.dumps({
        "siniflar": siniflar if siniflar is not None else ["fatura"],
        "guven": guven,
        "etiketler": etiketler if etiketler is not None else ["ornek"],
        "gerekce": "test gerekcesi",
    }, ensure_ascii=False)


def test_classify_document_happy_path():
    client = FakeClient([_valid_response(guven=0.95)])
    result = classifier.classify_document("bir fatura metni", client=client)
    assert result["siniflar"] == ["fatura"]
    assert result["human_review"] is False
    assert len(client.calls) == 1


def test_low_confidence_flags_human_review():
    client = FakeClient([_valid_response(guven=0.3)])
    result = classifier.classify_document("belirsiz bir metin", client=client)
    assert result["human_review"] is True


def test_missing_or_non_numeric_guven_flags_human_review():
    client = FakeClient([json.dumps({"siniflar": ["fatura"], "etiketler": []})])
    result = classifier.classify_document("metin", client=client)
    assert result["human_review"] is True


def test_unknown_category_falls_back_to_default():
    client = FakeClient([_valid_response(siniflar=["bilinmeyen_sinif"])])
    result = classifier.classify_document("metin", client=client)
    assert result["siniflar"] == [classifier.FALLBACK_CATEGORY]


def test_multi_label_keeps_only_valid_categories():
    client = FakeClient([_valid_response(siniflar=["fatura", "bilinmeyen", "sözleşme"])])
    result = classifier.classify_document("metin", client=client)
    assert result["siniflar"] == ["fatura", "sözleşme"]


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_text_raises_value_error(text):
    client = FakeClient([_valid_response()])
    with pytest.raises(ValueError):
        classifier.classify_document(text, client=client)


def test_malformed_json_is_retried_and_recovers():
    client = FakeClient(["bu gecerli bir json degil", _valid_response(guven=0.9)])
    result = classifier.classify_document("metin", client=client)

    assert result["siniflar"] == ["fatura"]
    assert len(client.calls) == 2
    assert "json" in client.calls[1]["user_message"].lower()


def test_markdown_fenced_json_is_parsed():
    fenced = "```json\n" + _valid_response() + "\n```"
    client = FakeClient([fenced])
    result = classifier.classify_document("metin", client=client)
    assert result["siniflar"] == ["fatura"]


def test_persistent_malformed_json_raises_after_max_attempts():
    client = FakeClient(["bozuk-1", "bozuk-2"])
    with pytest.raises(json.JSONDecodeError):
        classifier.classify_document("metin", client=client, max_json_attempts=2)
    assert len(client.calls) == 2


def test_default_temperature_is_zero_for_determinism():
    client = FakeClient([_valid_response()])
    classifier.classify_document("metin", client=client)
    assert client.calls[0]["temperature"] == 0.0


def test_custom_temperature_is_forwarded():
    client = FakeClient([_valid_response()])
    classifier.classify_document("metin", client=client, temperature=0.5)
    assert client.calls[0]["temperature"] == 0.5


def test_classify_chunks_joins_text_and_classifies():
    client = FakeClient([_valid_response(siniflar=["dilekçe"])])
    chunks = [
        {"chunk_id": 0, "text": "ilk parca"},
        {"chunk_id": 1, "text": "ikinci parca"},
    ]
    result = classifier.classify_chunks(chunks, client=client)
    assert result["siniflar"] == ["dilekçe"]
    assert "ilk parca" in client.calls[0]["user_message"]
    assert "ikinci parca" in client.calls[0]["user_message"]


def test_attach_labels_to_chunks_applies_matching_classification():
    chunks = [{"chunk_id": 0, "text": "x", "source_doc": "a.png"}]
    classifications = {"a.png": {"siniflar": ["fatura"], "guven": 0.9, "etiketler": ["e"], "human_review": False}}
    labeled = classifier.attach_labels_to_chunks(chunks, classifications)
    assert labeled[0]["siniflar"] == ["fatura"]
    assert labeled[0]["human_review"] is False


def test_attach_labels_to_chunks_defaults_missing_doc_to_human_review():
    chunks = [{"chunk_id": 0, "text": "x", "source_doc": "unclassified.png"}]
    labeled = classifier.attach_labels_to_chunks(chunks, {})
    assert labeled[0]["human_review"] is True
    assert labeled[0]["siniflar"] == [classifier.FALLBACK_CATEGORY]
