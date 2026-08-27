import pytest

import ocr


class _TextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_TextBlock(text)]


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return _FakeResponse(result)


class _FakeAnthropicClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "sample.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nfake-bytes")
    return str(path)


def test_extract_text_from_image_happy_path(sample_image):
    client = _FakeAnthropicClient(["Talep Eden: Ahmet Yilmaz"])
    text = ocr.extract_text_from_image(sample_image, client=client)
    assert text == "Talep Eden: Ahmet Yilmaz"
    assert len(client.messages.calls) == 1


def test_extract_text_from_image_retries_transient_failure(sample_image):
    client = _FakeAnthropicClient([RuntimeError("gecici ag hatasi"), "OK sonunda basarili"])
    text = ocr.extract_text_from_image(sample_image, client=client, retry_backoff_base=0.001)
    assert text == "OK sonunda basarili"
    assert len(client.messages.calls) == 2


def test_extract_text_from_image_raises_after_exhausting_retries(sample_image):
    client = _FakeAnthropicClient([RuntimeError("hata1"), RuntimeError("hata2"), RuntimeError("hata3")])
    with pytest.raises(RuntimeError):
        ocr.extract_text_from_image(sample_image, client=client, max_retries=2, retry_backoff_base=0.001)
    assert len(client.messages.calls) == 3


def test_extract_text_from_image_missing_file_raises_file_not_found():
    client = _FakeAnthropicClient(["irrelevant"])
    with pytest.raises(FileNotFoundError):
        ocr.extract_text_from_image("does-not-exist.png", client=client)
