import json

import pytest

import llm_json_utils


def test_extract_json_parses_plain_json():
    assert llm_json_utils.extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_strips_markdown_fence():
    fenced = "```json\n" + json.dumps({"a": 1}) + "\n```"
    assert llm_json_utils.extract_json(fenced) == {"a": 1}


def test_extract_json_strips_bare_backtick_fence_without_json_label():
    fenced = "```\n" + json.dumps({"a": 1}) + "\n```"
    assert llm_json_utils.extract_json(fenced) == {"a": 1}


def test_extract_json_invalid_raises_json_decode_error():
    with pytest.raises(json.JSONDecodeError):
        llm_json_utils.extract_json("bu bir json degil")


def test_generate_and_parse_json_happy_path(fake_llm_client):
    client = fake_llm_client(['{"x": 1}'])
    result = llm_json_utils.generate_and_parse_json(
        client=client, system_prompt="sys", user_message="user",
        max_tokens=100, temperature=0.0, max_json_attempts=2,
    )
    assert result == {"x": 1}
    assert len(client.calls) == 1


def test_generate_and_parse_json_retries_on_malformed_response(fake_llm_client):
    client = fake_llm_client(["bozuk json", '{"x": 1}'])
    result = llm_json_utils.generate_and_parse_json(
        client=client, system_prompt="sys", user_message="user",
        max_tokens=100, temperature=0.0, max_json_attempts=2,
    )
    assert result == {"x": 1}
    assert len(client.calls) == 2
    assert "json" in client.calls[1]["user_message"].lower()


def test_generate_and_parse_json_raises_after_exhausting_attempts(fake_llm_client):
    client = fake_llm_client(["bozuk-1", "bozuk-2"])
    with pytest.raises(json.JSONDecodeError):
        llm_json_utils.generate_and_parse_json(
            client=client, system_prompt="sys", user_message="user",
            max_tokens=100, temperature=0.0, max_json_attempts=2,
        )
    assert len(client.calls) == 2


def test_generate_and_parse_json_caller_name_appears_in_log(fake_llm_client, caplog):
    client = fake_llm_client(["bozuk", '{"x": 1}'])
    with caplog.at_level("WARNING", logger="llm_json_utils"):
        llm_json_utils.generate_and_parse_json(
            client=client, system_prompt="sys", user_message="user",
            max_tokens=100, temperature=0.0, max_json_attempts=2,
            caller_name="ozel_cagiran",
        )
    assert any("ozel_cagiran" in record.message for record in caplog.records)
