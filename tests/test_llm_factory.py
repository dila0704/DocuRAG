import json

import anthropic
import httpx
import pytest

import llm_factory
from llm_factory import AnthropicClient, LLMClient, _PROVIDER_REGISTRY, get_llm_client


class _FlakyClient(LLMClient):
    """Ilk `fail_times` cagriyi basarisiz kilan, sonra basarili donen sahte istemci."""

    def __init__(self, fail_times: int = 0, model_name: str = "flaky"):
        self.model_name = model_name
        self._fail_times = fail_times
        self.attempts = 0

    def _generate(self, system_prompt, user_message, max_tokens, temperature):
        self.attempts += 1
        if self.attempts <= self._fail_times:
            raise RuntimeError("gecici hata")
        return "ok"


def test_generate_retries_transient_failures_and_succeeds():
    client = _FlakyClient(fail_times=2)
    result = client.generate("sys", "user", max_retries=3, retry_backoff_base=0.001)
    assert result == "ok"
    assert client.attempts == 3


def test_generate_raises_last_error_after_exhausting_retries():
    client = _FlakyClient(fail_times=10)
    with pytest.raises(RuntimeError):
        client.generate("sys", "user", max_retries=2, retry_backoff_base=0.001)
    assert client.attempts == 3  # ilk deneme + 2 retry


def test_generate_no_retry_by_default_still_succeeds_first_try():
    client = _FlakyClient(fail_times=0)
    result = client.generate("sys", "user", retry_backoff_base=0.001)
    assert result == "ok"
    assert client.attempts == 1


def test_generate_passes_temperature_through_to_generate_impl():
    seen = {}

    class _Recorder(LLMClient):
        model_name = "rec"

        def _generate(self, system_prompt, user_message, max_tokens, temperature):
            seen["temperature"] = temperature
            return "ok"

    _Recorder().generate("sys", "user", temperature=0.7)
    assert seen["temperature"] == 0.7


def test_get_llm_client_dispatches_to_registered_provider(monkeypatch):
    created = {}

    class DummyClient(LLMClient):
        def __init__(self, model_name):
            self.model_name = model_name
            created["model_name"] = model_name

        def _generate(self, *a, **k):
            return ""

    monkeypatch.setitem(_PROVIDER_REGISTRY, "anthropic", DummyClient)

    client = get_llm_client({
        "active_mode": "cloud",
        "cloud_model": {"provider": "anthropic", "model_name": "claude-x"},
    })

    assert isinstance(client, DummyClient)
    assert created["model_name"] == "claude-x"


def test_get_llm_client_unknown_provider_raises_value_error():
    with pytest.raises(ValueError):
        get_llm_client({
            "active_mode": "cloud",
            "cloud_model": {"provider": "does-not-exist", "model_name": "x"},
        })


def test_get_llm_client_missing_mode_block_raises_value_error():
    with pytest.raises(ValueError):
        get_llm_client({"active_mode": "cloud"})


def test_get_llm_client_missing_model_name_raises_value_error():
    with pytest.raises(ValueError):
        get_llm_client({
            "active_mode": "cloud",
            "cloud_model": {"provider": "anthropic"},
        })


def test_get_llm_client_local_mode_selects_local_model_block(monkeypatch):
    created = {}

    class DummyClient(LLMClient):
        def __init__(self, model_name):
            self.model_name = model_name
            created["model_name"] = model_name

        def _generate(self, *a, **k):
            return ""

    monkeypatch.setitem(_PROVIDER_REGISTRY, "huggingface", DummyClient)

    client = get_llm_client({
        "active_mode": "local",
        "local_model": {"provider": "huggingface", "model_name": "Qwen/Qwen2.5-0.5B-Instruct"},
    })

    assert isinstance(client, DummyClient)
    assert created["model_name"] == "Qwen/Qwen2.5-0.5B-Instruct"


def _bad_request_error(message: str) -> anthropic.BadRequestError:
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"), json={})
    return anthropic.BadRequestError(message, response=response, body=None)


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _TextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_TextBlock(text)]


def _make_anthropic_client(responses) -> tuple[AnthropicClient, _FakeMessages]:
    # __init__ gercek bir anthropic.Anthropic() (API anahtari/ag) kurar;
    # testte buna gerek yok, __new__ ile atlanip _client elle sahte bir
    # nesneyle degistiriliyor.
    client = AnthropicClient.__new__(AnthropicClient)
    client.model_name = "claude-sonnet-5"
    fake_messages = _FakeMessages(responses)
    client._client = type("FakeAnthropic", (), {"messages": fake_messages})()
    return client, fake_messages


def test_anthropic_client_retries_without_temperature_when_deprecated():
    error = _bad_request_error("`temperature` is deprecated for this model.")
    client, fake_messages = _make_anthropic_client([error, _FakeResponse("ok")])

    text = client._generate("sys", "user", max_tokens=100, temperature=0.0)

    assert text == "ok"
    assert len(fake_messages.calls) == 2
    assert "temperature" in fake_messages.calls[0]
    assert "temperature" not in fake_messages.calls[1]


def test_anthropic_client_reraises_unrelated_bad_request_error():
    error = _bad_request_error("some other validation error")
    client, fake_messages = _make_anthropic_client([error])

    with pytest.raises(anthropic.BadRequestError):
        client._generate("sys", "user", max_tokens=100, temperature=0.0)
    assert len(fake_messages.calls) == 1


def _read_usage_log(path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class _UsageRecordingClient(LLMClient):
    model_name = "claude-sonnet-5"

    def _generate(self, system_prompt, user_message, max_tokens, temperature):
        self._last_usage = {"input_tokens": 100, "output_tokens": 50}
        return "ok"


def test_generate_appends_usage_log_on_success(tmp_path, monkeypatch):
    log_path = tmp_path / "usage_log.jsonl"
    monkeypatch.setattr(llm_factory, "DEFAULT_USAGE_LOG_PATH", log_path)

    _UsageRecordingClient().generate("sys", "user")

    records = _read_usage_log(log_path)
    assert len(records) == 1
    assert records[0]["provider"] == "_UsageRecordingClient"
    assert records[0]["model_name"] == "claude-sonnet-5"
    assert records[0]["input_tokens"] == 100
    assert records[0]["output_tokens"] == 50
    assert records[0]["cost_usd"] == pytest.approx(100 / 1_000_000 * 3.0 + 50 / 1_000_000 * 15.0)
    assert records[0]["duration_s"] >= 0


def test_generate_logs_none_usage_for_client_without_last_usage(tmp_path, monkeypatch):
    log_path = tmp_path / "usage_log.jsonl"
    monkeypatch.setattr(llm_factory, "DEFAULT_USAGE_LOG_PATH", log_path)

    class _NoUsageClient(LLMClient):
        model_name = "fake"

        def _generate(self, system_prompt, user_message, max_tokens, temperature):
            return "ok"

    _NoUsageClient().generate("sys", "user")

    records = _read_usage_log(log_path)
    assert len(records) == 1
    assert records[0]["input_tokens"] is None
    assert records[0]["cost_usd"] is None


def test_generate_does_not_raise_if_usage_logging_fails(tmp_path, monkeypatch):
    # Var olmayan bir dizinin ALTINDA (dosya olarak) bir yol vererek yazmayi
    # kasitli basarisiz kilariz; generate() yine de metni dondurmeli.
    unwritable = tmp_path / "not_a_directory" / "usage_log.jsonl"
    (tmp_path / "not_a_directory").write_text("bu bir dosya, dizin degil")
    monkeypatch.setattr(llm_factory, "DEFAULT_USAGE_LOG_PATH", unwritable)

    result = _UsageRecordingClient().generate("sys", "user")
    assert result == "ok"


def test_estimate_cost_returns_none_for_unknown_model():
    assert llm_factory._estimate_cost_usd("bilinmeyen-model", 100, 50) is None


def test_estimate_cost_returns_none_when_tokens_missing():
    assert llm_factory._estimate_cost_usd("claude-sonnet-5", None, None) is None


def test_estimate_cost_computes_known_model():
    cost = llm_factory._estimate_cost_usd("claude-sonnet-5", 1_000_000, 1_000_000)
    assert cost == pytest.approx(3.0 + 15.0)


def _write_yaml(path, data):
    import yaml

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)


def test_load_llm_config_without_override_matches_base(tmp_path):
    config_path = tmp_path / "settings.yaml"
    _write_yaml(config_path, {"llm_settings": {"active_mode": "cloud", "cloud_model": {"provider": "anthropic", "model_name": "claude-sonnet-5"}}})

    result = llm_factory.load_llm_config(config_path)
    assert result["active_mode"] == "cloud"
    assert result["cloud_model"]["model_name"] == "claude-sonnet-5"


def test_load_llm_config_merges_local_override(tmp_path):
    config_path = tmp_path / "settings.yaml"
    _write_yaml(config_path, {"llm_settings": {
        "active_mode": "cloud",
        "cloud_model": {"provider": "anthropic", "model_name": "claude-sonnet-5"},
        "local_model": {"provider": "huggingface", "model_name": "meta-llama/Meta-Llama-3-8B-Instruct"},
    }})
    _write_yaml(tmp_path / "settings.local.yaml", {"active_mode": "local"})

    result = llm_factory.load_llm_config(config_path)
    assert result["active_mode"] == "local"
    # override edilmeyen alanlar (cloud_model) korunmali
    assert result["cloud_model"]["model_name"] == "claude-sonnet-5"


def test_load_llm_config_merges_nested_dict_fields(tmp_path):
    config_path = tmp_path / "settings.yaml"
    _write_yaml(config_path, {"llm_settings": {
        "active_mode": "cloud",
        "cloud_model": {"provider": "anthropic", "model_name": "claude-sonnet-5"},
    }})
    _write_yaml(tmp_path / "settings.local.yaml", {"cloud_model": {"model_name": "gpt_4", "provider": "openai"}})

    result = llm_factory.load_llm_config(config_path)
    assert result["cloud_model"] == {"provider": "openai", "model_name": "gpt_4"}


# --- generate_structured() (DOC-34) --------------------------------------

_DUMMY_SCHEMA = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}


def test_generate_structured_base_fallback_delegates_to_json_parsing(fake_llm_client):
    """Structured output desteklemeyen bir saglayici (fake istemci taban
    sinifin generate_structured'ini MIRAS alir) eski JSON-prompt+retry
    desenine dusmeli -- schema/tool_name kullanilmaz, sadece generate()
    uzerinden normal bir metin yaniti istenir."""
    client = fake_llm_client(['{"x": 42}'])
    result = client.generate_structured("sys", "user", schema=_DUMMY_SCHEMA, tool_name="t")
    assert result == {"x": 42}
    assert len(client.calls) == 1


def test_generate_structured_base_fallback_retries_malformed_json(fake_llm_client):
    client = fake_llm_client(["bozuk", '{"x": 1}'])
    result = client.generate_structured("sys", "user", schema=_DUMMY_SCHEMA, tool_name="t", max_json_attempts=2)
    assert result == {"x": 1}
    assert len(client.calls) == 2


class _ToolUseBlock:
    def __init__(self, name, input_):
        self.type = "tool_use"
        self.name = name
        self.input = input_


class _FakeToolResponse:
    def __init__(self, blocks, usage=None):
        self.content = blocks
        self.usage = usage


def test_anthropic_generate_structured_returns_tool_input(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_factory, "DEFAULT_USAGE_LOG_PATH", tmp_path / "usage_log.jsonl")
    client, fake_messages = _make_anthropic_client([
        _FakeToolResponse([_ToolUseBlock("classify_document", {"siniflar": ["fatura"], "guven": 0.9})]),
    ])

    result = client.generate_structured("sys", "user", schema=_DUMMY_SCHEMA, tool_name="classify_document")

    assert result == {"siniflar": ["fatura"], "guven": 0.9}
    assert len(fake_messages.calls) == 1
    assert fake_messages.calls[0]["tool_choice"] == {"type": "tool", "name": "classify_document"}
    assert fake_messages.calls[0]["tools"][0]["input_schema"] == _DUMMY_SCHEMA


def test_anthropic_generate_structured_retries_when_no_tool_use_block(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_factory, "DEFAULT_USAGE_LOG_PATH", tmp_path / "usage_log.jsonl")
    client, fake_messages = _make_anthropic_client([
        _FakeToolResponse([_TextBlock("model tool cagirmadi")]),
        _FakeToolResponse([_ToolUseBlock("t", {"x": 1})]),
    ])

    result = client.generate_structured("sys", "user", schema=_DUMMY_SCHEMA, tool_name="t", retry_backoff_base=0.001)

    assert result == {"x": 1}
    assert len(fake_messages.calls) == 2


def test_anthropic_generate_structured_handles_deprecated_temperature(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_factory, "DEFAULT_USAGE_LOG_PATH", tmp_path / "usage_log.jsonl")
    error = _bad_request_error("`temperature` is deprecated for this model.")
    client, fake_messages = _make_anthropic_client([
        error, _FakeToolResponse([_ToolUseBlock("t", {"x": 1})]),
    ])

    result = client.generate_structured("sys", "user", schema=_DUMMY_SCHEMA, tool_name="t")

    assert result == {"x": 1}
    assert len(fake_messages.calls) == 2
    assert "temperature" in fake_messages.calls[0]
    assert "temperature" not in fake_messages.calls[1]


class _FakeFunctionCall:
    def __init__(self, name, arguments):
        self.function = type("F", (), {"name": name, "arguments": arguments})()


class _FakeToolCallMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeOpenAIResponse:
    def __init__(self, tool_calls, usage=None):
        self.choices = [_FakeChoice(_FakeToolCallMessage(tool_calls))]
        self.usage = usage


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _make_openai_client(responses):
    from llm_factory import OpenAIClient

    client = OpenAIClient.__new__(OpenAIClient)
    client.model_name = "gpt_4"
    fake_completions = _FakeCompletions(responses)
    client._client = type("FakeOpenAI", (), {"chat": type("Chat", (), {"completions": fake_completions})()})()
    return client, fake_completions


def test_openai_generate_structured_parses_function_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_factory, "DEFAULT_USAGE_LOG_PATH", tmp_path / "usage_log.jsonl")
    client, fake_completions = _make_openai_client([
        _FakeOpenAIResponse([_FakeFunctionCall("t", '{"x": 7}')]),
    ])

    result = client.generate_structured("sys", "user", schema=_DUMMY_SCHEMA, tool_name="t")

    assert result == {"x": 7}
    assert fake_completions.calls[0]["tool_choice"] == {"type": "function", "function": {"name": "t"}}


def test_anthropic_generate_structured_raises_after_exhausting_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_factory, "DEFAULT_USAGE_LOG_PATH", tmp_path / "usage_log.jsonl")
    client, fake_messages = _make_anthropic_client([RuntimeError("gecici hata"), RuntimeError("gecici hata")])

    with pytest.raises(RuntimeError):
        client.generate_structured("sys", "user", schema=_DUMMY_SCHEMA, tool_name="t", max_retries=1, retry_backoff_base=0.001)
    assert len(fake_messages.calls) == 2


def test_anthropic_generate_structured_raises_when_no_tool_use_after_all_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_factory, "DEFAULT_USAGE_LOG_PATH", tmp_path / "usage_log.jsonl")
    client, fake_messages = _make_anthropic_client([
        _FakeToolResponse([_TextBlock("hic tool cagirmadi")]),
        _FakeToolResponse([_TextBlock("yine cagirmadi")]),
    ])

    with pytest.raises(RuntimeError):
        client.generate_structured("sys", "user", schema=_DUMMY_SCHEMA, tool_name="t", max_retries=1, retry_backoff_base=0.001)
    assert len(fake_messages.calls) == 2


def test_openai_generate_structured_retries_when_no_matching_tool_call(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_factory, "DEFAULT_USAGE_LOG_PATH", tmp_path / "usage_log.jsonl")
    client, fake_completions = _make_openai_client([
        _FakeOpenAIResponse([_FakeFunctionCall("baska_fonksiyon", "{}")]),
        _FakeOpenAIResponse([_FakeFunctionCall("t", '{"x": 3}')]),
    ])

    result = client.generate_structured("sys", "user", schema=_DUMMY_SCHEMA, tool_name="t", retry_backoff_base=0.001)

    assert result == {"x": 3}
    assert len(fake_completions.calls) == 2


def test_openai_generate_structured_raises_after_exhausting_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_factory, "DEFAULT_USAGE_LOG_PATH", tmp_path / "usage_log.jsonl")
    client, fake_completions = _make_openai_client([RuntimeError("gecici hata"), RuntimeError("gecici hata")])

    with pytest.raises(RuntimeError):
        client.generate_structured("sys", "user", schema=_DUMMY_SCHEMA, tool_name="t", max_retries=1, retry_backoff_base=0.001)
    assert len(fake_completions.calls) == 2


def test_openai_generate_structured_retries_on_invalid_json_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_factory, "DEFAULT_USAGE_LOG_PATH", tmp_path / "usage_log.jsonl")
    client, fake_completions = _make_openai_client([
        _FakeOpenAIResponse([_FakeFunctionCall("t", "bozuk-json")]),
        _FakeOpenAIResponse([_FakeFunctionCall("t", '{"x": 9}')]),
    ])

    result = client.generate_structured("sys", "user", schema=_DUMMY_SCHEMA, tool_name="t", retry_backoff_base=0.001)
    assert result == {"x": 9}
    assert len(fake_completions.calls) == 2


def test_save_llm_settings_override_writes_only_override_file(tmp_path):
    config_path = tmp_path / "settings.yaml"
    base_content = "llm_settings:\n  active_mode: cloud\n  cloud_model:\n    provider: anthropic\n    model_name: claude-sonnet-5\n# elle yazilmis bir yorum\n"
    config_path.write_text(base_content, encoding="utf-8")

    llm_factory.save_llm_settings_override({"active_mode": "local"}, config_path=config_path)

    # base dosya HIC degismemis olmali (yorum dahil)
    assert config_path.read_text(encoding="utf-8") == base_content
    override_path = tmp_path / "settings.local.yaml"
    assert override_path.exists()

    result = llm_factory.load_llm_config(config_path)
    assert result["active_mode"] == "local"
