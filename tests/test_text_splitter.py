import re

import pytest

from text_splitter import count_tokens, split_text


def test_short_text_returns_single_chunk():
    text = "Merhaba dunya. Bu kisa bir test metni."
    result = split_text(text, chunk_size=50, chunk_overlap=10)
    assert len(result) == 1
    assert result[0]["text"] == text
    assert result[0]["chunk_id"] == 0


def test_invalid_overlap_raises_value_error():
    with pytest.raises(ValueError):
        split_text("herhangi bir metin", chunk_size=20, chunk_overlap=20)


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_empty_or_blank_text_returns_empty_list(text):
    assert split_text(text, chunk_size=50, chunk_overlap=10) == []


def test_no_word_is_broken_across_chunk_boundaries():
    corpus = (
        "Bu uzun bir test metnidir. Birden fazla cumle icerir. "
        "Amac, chunk sinirlarinin kelime ortasindan gecmedigini dogrulamaktir. "
        "Turkce karakterler de iceriyor: sozlesme, dilekce, fatura, talep formu."
    )
    chunks = split_text(corpus, chunk_size=15, chunk_overlap=4)
    assert len(chunks) > 1

    corpus_words = set(re.findall(r"\w+", corpus, flags=re.UNICODE))
    for chunk in chunks:
        for word in re.findall(r"\w+", chunk["text"], flags=re.UNICODE):
            assert word in corpus_words, f"kirik/olmayan kelime: {word!r}"


def test_no_data_loss_when_no_natural_separators_exist():
    # Bosluk/noktalama iceren hicbir ayrac bulunmuyor; splitter yine de
    # metnin tamamini kaybetmeden parcalamali.
    long_text = "a" * 3000
    chunks = split_text(long_text, chunk_size=50, chunk_overlap=0)
    assert len(chunks) > 1
    assert "".join(c["text"] for c in chunks) == long_text


def test_chunk_ids_are_sequential_starting_at_zero():
    corpus = "Cumle bir. Cumle iki. Cumle uc. Cumle dort."
    chunks = split_text(corpus, chunk_size=5, chunk_overlap=1)
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))


def test_token_count_field_matches_count_tokens():
    corpus = "Cumle bir. Cumle iki. Cumle uc. Cumle dort."
    for chunk in split_text(corpus, chunk_size=5, chunk_overlap=1):
        assert chunk["token_count"] == count_tokens(chunk["text"])


def test_count_tokens_empty_string_is_zero():
    assert count_tokens("") == 0


def test_count_tokens_nonempty_is_positive():
    assert count_tokens("Merhaba") > 0
