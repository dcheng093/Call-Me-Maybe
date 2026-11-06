"""Turns the model's byte-level BPE vocabulary into real text."""

import json

from pydantic import BaseModel


def bytes_to_unicode() -> dict[int, str]:
    """Builds the byte -> visible character table used by byte-level BPE.

    Bytes that are already printable characters map to themselves. The
    68 remaining bytes (spaces, control bytes, ...) map, in order, to
    chr(256), chr(257), ... so that every byte is visible in vocab.json.
    For example the space (byte 32) becomes "Ġ" and the newline "Ċ".

    Returns:
        A dict mapping each byte value (0-255) to its character.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    table: dict[int, str] = {byte: chr(byte) for byte in printable}
    extra = 0
    for byte in range(256):
        if byte not in table:
            table[byte] = chr(256 + extra)
            extra += 1
    return table


def decode_piece(piece: str, char_to_byte: dict[str, int]) -> str | None:
    """Converts one vocab.json entry to the text it really stands for.

    Args:
        piece: The token as written in vocab.json, e.g. "Ġthe".
        char_to_byte: Reverse of bytes_to_unicode().

    Returns:
        The decoded text (e.g. " the"), or None if the token is only part
        of a multi-byte character and cannot be decoded on its own.
    """
    try:
        raw = bytes(char_to_byte[char] for char in piece)
        text = raw.decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return None
    return text or None


class Vocabulary(BaseModel):
    """Every token of the model, decoded once at startup.

    Attributes:
        texts: texts[token_id] is the token's real text, or None when the
            token cannot be used on its own.
        by_first_char: Usable token IDs grouped by their first character.
        longest: Length of the longest token text.
    """

    texts: list[str | None]
    by_first_char: dict[str, list[int]]
    longest: int

    @classmethod
    def from_file(cls, path: str) -> "Vocabulary":
        """Reads vocab.json and decodes every token.

        Args:
            path: Path returned by Small_LLM_Model.get_path_to_vocab_file().

        Returns:
            The decoded vocabulary.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If the file is not a {token: id} JSON object.
        """
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict) or not raw:
            raise ValueError(f"{path} is not a token -> id mapping")

        table = bytes_to_unicode()
        char_to_byte = {char: byte for byte, char in table.items()}
        texts: list[str | None] = [None] * (max(raw.values()) + 1)
        by_first_char: dict[str, list[int]] = {}
        longest = 0
        for piece, token_id in raw.items():
            text = decode_piece(piece, char_to_byte)
            texts[token_id] = text
            if text is not None:
                by_first_char.setdefault(text[0], []).append(token_id)
                longest = max(longest, len(text))
        return cls(texts=texts, by_first_char=by_first_char, longest=longest)
