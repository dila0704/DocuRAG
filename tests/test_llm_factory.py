import anthropic
import httpx
import pytest

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
