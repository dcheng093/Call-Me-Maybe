import json
from typing import Any, Dict, List

from pydantic import BaseModel, field_validator


class CustomTokenizer(BaseModel):
    """
    Tokenizer with grammar-constrained token filtering.

    Loads a BPE vocabulary from a JSON file and exposes methods to decode
    individual token IDs to strings and to compute, for a given grammar
    state, the subset of token IDs that are structurally valid.

    Results are memoized by grammar state string so that each unique state
    is only evaluated once per prompt, reducing compute time.

    Attributes:
        vocab_json_path: Path to the vocabulary JSON file
        vocab: Inverted vocabulary mapping {token_id: token_string}.
        cache_hits: Number of times the mask cache was hit.
        cache_misses: Number of times the mask cache was missed.
    """

    vocab_json_path: str
    vocab: Dict[int, str] = {}
    cache_hits: int = 0
    cache_misses: int = 0
    _mask_cache: Dict[str, List[int]] = {}

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("vocab_json_path")
    @classmethod
    def path_must_be_non_empty(cls, v: str) -> str:
        """
        Validate that the vocabulary path is not an empty string.

        Args:
            v: The file path to validate.

        Returns:
            The validated path string.

        Raises:
            ValueError: If the path is empty.
        """
        if not v.strip():
            raise ValueError("vocab_json_path must not be empty.")
        return v

    def model_post_init(self, __context: Any) -> None:
        """
        Load and invert the vocabulary file after Pydantic initialisation.

        Reads the JSON file at vocab_json_path, inverts the
        {token_string: token_id} mapping to {token_id: token_string},
        and stores it in vocab.  Also initialises the private mask cache.

        Args:
            __context: Pydantic post-init context (unused).

        Raises:
            FileNotFoundError: If the vocabulary file does not exist.
            json.JSONDecodeError: If the vocabulary file is not valid JSON.
        """
        with open(self.vocab_json_path, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        inverted: Dict[int, str] = {}
        for token_str, token_id in raw_vocab.items():
            inverted[int(token_id)] = token_str
        self.vocab = inverted
        self._mask_cache = {}

    def decode_single_token(self, token_id: int) -> str:
        """
        Decode a single token ID to its character string representation.

        Handles BPE-specific encoding by replacing the Ġ symbol with a
        regular space and Ċ with a newline character.

        Args:
            token_id: The integer token ID to decode.

        Returns:
            The decoded string for the token, or an empty string if the ID
            is not present in the vocabulary.
        """
        raw_piece = self.vocab.get(token_id, "")
        return raw_piece.replace("Ġ", " ").replace("Ċ", "\n")

    def get_allowed_token_ids(self, rulebook: Any) -> List[int]:
        """
        Returns the list of token IDs that are valid at the current grammar
        state.

        Results are cached by rulebook.text_so_far.  On a cache miss,
        every entry in the vocabulary is tested via rulebook.can_walk_token
        and only those that pass are returned.

        Args:
            rulebook: A TrieJSONRulebook instance representing the current
                generation state.

        Returns:
            A list of integer token IDs that the grammar allows at this step.
            Returns an empty list when no further tokens are permitted.
        """
        cache_key = rulebook.text_so_far

        if cache_key in self._mask_cache:
            self.cache_hits += 1
            return self._mask_cache[cache_key]

        self.cache_misses += 1

        allowed_chars = rulebook.get_allowed_characters()
        if not allowed_chars:
            self._mask_cache[cache_key] = []
            return []

        allowed_ids: List[int] = []
        for token_id, token_str in self.vocab.items():
            cleaned_str = token_str.replace("Ġ", " ").replace("Ċ", "\n")
            if rulebook.can_walk_token(cleaned_str):
                allowed_ids.append(token_id)

        self._mask_cache[cache_key] = allowed_ids
        return allowed_ids

    def print_telemetry(self) -> None:
        """
        Prints a cache performance summary to stdout.

        Displays the total number of cache hits, misses, and the resulting
        hit-rate percentage for the lifetime of this tokenizer instance.
        """
        total = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total * 100) if total > 0 else 0
        msg = (
            f"[Performance Profile] "
            f"Cache Hits: {self.cache_hits} | "
            f"Cache Misses: {self.cache_misses} | "
            f"Hit Rate: {hit_rate:.2f}%"
        )
        print(msg)
