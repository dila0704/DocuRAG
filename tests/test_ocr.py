import io

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


@pytest.fixture
def real_png_image(tmp_path):
    """extract_word_boxes/render_highlighted_image PIL ile GERCEKTEN acilan
    bir goruntu bekler -- yukaridaki sample_image (bilerek bozuk PNG bytes)
    bunun icin yetersiz, gecerli kucuk bir PNG uretilir."""
    from PIL import Image

    path = tmp_path / "real.png"
    Image.new("RGB", (200, 100), color=(255, 255, 255)).save(path)
    return str(path)


def test_extract_word_boxes_returns_none_when_tesseract_binary_missing(real_png_image, monkeypatch):
    import pytesseract

    def _raise(*args, **kwargs):
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_data", _raise)
    assert ocr.extract_word_boxes(real_png_image) is None


def test_extract_word_boxes_parses_tesseract_output(real_png_image, monkeypatch):
    import pytesseract

    fake_data = {
        "text": ["", "Laptop", "talebi", ""],
        "left": [0, 10, 60, 0],
        "top": [0, 20, 20, 0],
        "width": [0, 45, 40, 0],
        "height": [0, 15, 15, 0],
    }
    monkeypatch.setattr(pytesseract, "image_to_data", lambda *a, **k: fake_data)

    boxes = ocr.extract_word_boxes(real_png_image)

    assert boxes == [
        {"text": "Laptop", "left": 10, "top": 20, "width": 45, "height": 15},
        {"text": "talebi", "left": 60, "top": 20, "width": 40, "height": 15},
    ]


def test_locate_chunk_bbox_none_word_boxes_returns_none():
    assert ocr.locate_chunk_bbox("laptop talebi", None) is None


def test_locate_chunk_bbox_empty_chunk_text_returns_none():
    assert ocr.locate_chunk_bbox("", [{"text": "x", "left": 0, "top": 0, "width": 1, "height": 1}]) is None


def test_locate_chunk_bbox_returns_union_of_matched_words():
    word_boxes = [
        {"text": "Laptop", "left": 10, "top": 20, "width": 40, "height": 15},
        {"text": "talebi", "left": 60, "top": 20, "width": 40, "height": 15},
        {"text": "ofis", "left": 200, "top": 200, "width": 30, "height": 15},
    ]
    bbox = ocr.locate_chunk_bbox("Laptop talebi", word_boxes)

    assert bbox is not None
    assert bbox["left"] == 10
    assert bbox["top"] == 20
    assert bbox["width"] == (60 + 40) - 10
    assert bbox["height"] == 15
    assert bbox["match_ratio"] == 1.0


def test_locate_chunk_bbox_returns_none_below_match_threshold():
    word_boxes = [{"text": "alakasiz", "left": 0, "top": 0, "width": 10, "height": 10}]
    bbox = ocr.locate_chunk_bbox("tamamen farkli uzun bir cumle burada", word_boxes)
    assert bbox is None


def test_render_highlighted_image_without_bbox_returns_valid_png(real_png_image):
    from PIL import Image

    result = ocr.render_highlighted_image(real_png_image, None)
    img = Image.open(io.BytesIO(result))
    assert img.format == "PNG"


def test_render_highlighted_image_with_bbox_returns_valid_png(real_png_image):
    from PIL import Image

    bbox = {"left": 10, "top": 10, "width": 50, "height": 20}
    result = ocr.render_highlighted_image(real_png_image, bbox)
    img = Image.open(io.BytesIO(result))
    assert img.format == "PNG"
