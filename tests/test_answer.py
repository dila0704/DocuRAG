import json

import pytest

import answer


def _chunks(n=2):
    return [
        {"text": f"chunk metni {i}", "source_doc": f"doc_{i}.png", "score": 0.9 - i * 0.1}
        for i in range(1, n + 1)
    ]


def _response(sentences):
    return json.dumps({"sentences": sentences}, ensure_ascii=False)


def test_empty_chunks_returns_ungrounded_without_calling_client(fake_llm_client):
    client = fake_llm_client([])
    result = answer.generate_grounded_answer("sorgu", [], client=client)
    assert result == {"grounded": False, "sentences": [], "chunks": []}
    assert client.calls == []


def test_happy_path_all_sentences_grounded(fake_llm_client):
    chunks = _chunks(2)
    client = fake_llm_client([_response([
        {"text": "Birinci cumle.", "sources": [1]},
        {"text": "Ikinci cumle.", "sources": [1, 2]},
    ])])
    result = answer.generate_grounded_answer("sorgu", chunks, client=client)

    assert result["grounded"] is True
    assert len(result["sentences"]) == 2
    assert result["sentences"][0] == {"text": "Birinci cumle.", "sources": [1]}
    assert result["chunks"] == chunks


def test_sentence_with_empty_sources_is_dropped(fake_llm_client):
    chunks = _chunks(2)
    client = fake_llm_client([_response([
        {"text": "Kaynaksiz cumle.", "sources": []},
        {"text": "Kaynakli cumle.", "sources": [2]},
    ])])
    result = answer.generate_grounded_answer("sorgu", chunks, client=client)

    assert result["grounded"] is True
    assert len(result["sentences"]) == 1
    assert result["sentences"][0]["text"] == "Kaynakli cumle."


def test_sentence_with_out_of_range_source_is_dropped(fake_llm_client):
    chunks = _chunks(2)
    client = fake_llm_client([_response([
        {"text": "Uydurma kaynakli cumle.", "sources": [5]},
    ])])
    result = answer.generate_grounded_answer("sorgu", chunks, client=client)

    assert result["grounded"] is False
    assert result["sentences"] == []


def test_sentence_missing_sources_key_is_dropped(fake_llm_client):
    chunks = _chunks(1)
    client = fake_llm_client([_response([{"text": "sources alani yok."}])])
    result = answer.generate_grounded_answer("sorgu", chunks, client=client)
    assert result["sentences"] == []


def test_all_sentences_invalid_returns_ungrounded(fake_llm_client):
    chunks = _chunks(1)
    client = fake_llm_client([_response([{"text": "x", "sources": []}])])
    result = answer.generate_grounded_answer("sorgu", chunks, client=client)
    assert result["grounded"] is False
    assert result["sentences"] == []


def test_model_returns_no_answer_when_instructed(fake_llm_client):
    chunks = _chunks(1)
    client = fake_llm_client([_response([])])
    result = answer.generate_grounded_answer("alakasiz sorgu", chunks, client=client)
    assert result == {"grounded": False, "sentences": [], "chunks": chunks}


def test_malformed_json_is_retried_and_recovers(fake_llm_client):
    chunks = _chunks(1)
    client = fake_llm_client(["gecersiz json", _response([{"text": "tamam.", "sources": [1]}])])
    result = answer.generate_grounded_answer("sorgu", chunks, client=client)

    assert result["grounded"] is True
    assert len(client.calls) == 2


def test_persistent_malformed_json_raises_after_max_attempts(fake_llm_client):
    chunks = _chunks(1)
    client = fake_llm_client(["bozuk-1", "bozuk-2"])
    with pytest.raises(json.JSONDecodeError):
        answer.generate_grounded_answer("sorgu", chunks, client=client, max_json_attempts=2)
    assert len(client.calls) == 2


def test_default_temperature_is_zero(fake_llm_client):
    chunks = _chunks(1)
    client = fake_llm_client([_response([{"text": "x.", "sources": [1]}])])
    answer.generate_grounded_answer("sorgu", chunks, client=client)
    assert client.calls[0]["temperature"] == 0.0


def test_sources_block_includes_source_doc_and_index(fake_llm_client):
    chunks = _chunks(2)
    client = fake_llm_client([_response([{"text": "x.", "sources": [1]}])])
    answer.generate_grounded_answer("sorgu", chunks, client=client)
    system_prompt = client.calls[0]["system_prompt"]
    assert "[1]" in system_prompt and "doc_1.png" in system_prompt
    assert "[2]" in system_prompt and "doc_2.png" in system_prompt
