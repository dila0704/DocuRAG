"""
Ozel (framework'suz) recursive text splitter.

Hazir agir kutuphaneler (LangChain vb.) yerine, paragraf -> satir -> cumle ->
kelime siniri hiyerarsisini sirayla deneyerek metni anlam butunlugunu
korumaya calisarak parcalayan, token bazli boyut siniri kullanan bir
splitter. Ayraclardan hicbiri sinira sigdiramazsa son care olarak sabit
token uzunlugunda sert kesim yapilir.
"""
from __future__ import annotations

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _split_on_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    parts = text.split(separator)
    return [p + separator for p in parts[:-1]] + [parts[-1]]


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if count_tokens(text) <= chunk_size:
        return [text] if text else []

    if not separators:
        tokens = _ENCODING.encode(text)
        return [
            _ENCODING.decode(tokens[i : i + chunk_size])
            for i in range(0, len(tokens), chunk_size)
        ]

    separator, *rest = separators
    pieces = _split_on_separator(text, separator)

    result = []
    for piece in pieces:
        if count_tokens(piece) <= chunk_size:
            if piece:
                result.append(piece)
        else:
            result.extend(_recursive_split(piece, chunk_size, rest))
    return result


def _clean_overlap_start(text: str) -> str:
    """Token bazli kesim bir kelimenin ortasindan basliyorsa, o parcali kelimeyi at."""
    if not text or text[0].isspace():
        return text.lstrip()
    idx = text.find(" ")
    return text[idx + 1 :] if idx != -1 else text


def _merge_with_overlap(pieces: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""

    for piece in pieces:
        candidate = current + piece
        if count_tokens(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if chunk_overlap > 0 and current:
            overlap_tokens = _ENCODING.encode(current)[-chunk_overlap:]
            overlap_text = _clean_overlap_start(_ENCODING.decode(overlap_tokens))
            current = overlap_text + piece
        else:
            current = piece

    if current:
        chunks.append(current)
    return chunks


def split_text(
    text: str,
    chunk_size: int = 300,
    chunk_overlap: int = 50,
    separators: list[str] | None = None,
) -> list[dict]:
    """Metni anlam butunlugunu korumaya calisarak token-limitli parcalara boler.

    Returns:
        [{"chunk_id": int, "text": str, "token_count": int}, ...]
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap, chunk_size'dan kucuk olmali.")

    raw_pieces = _recursive_split(text.strip(), chunk_size, separators or DEFAULT_SEPARATORS)
    merged = _merge_with_overlap(raw_pieces, chunk_size, chunk_overlap)

    return [
        {"chunk_id": i, "text": chunk.strip(), "token_count": count_tokens(chunk)}
        for i, chunk in enumerate(merged)
        if chunk.strip()
    ]
