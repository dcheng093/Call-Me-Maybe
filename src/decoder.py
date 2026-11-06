"""Constrained generation loop: the model chooses, the grammar filters."""

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, PrivateAttr

from src.grammar import CallGrammar, State
from src.schemas import CallResult, FunctionSpec
from src.vocab import Vocabulary

MAX_STEPS = 4096


def describe_function(function: FunctionSpec) -> str:
    """Writes one function as a short signature line for the prompt.

    Args:
        function: The function definition.

    Returns:
        A line such as "- fn_greet(name: string) -> string: Greets ...".
    """
    params = ", ".join(
        f"{name}: {spec.type}" for name, spec in function.parameters.items()
    )
    line = f"- {function.name}({params})"
    if function.returns is not None:
        line += f" -> {function.returns.type}"
    if function.description:
        line += f": {function.description}"
    return line


def build_prompt(functions: list[FunctionSpec], request: str) -> str:
    """Builds the text given to the model before it writes the call.

    Args:
        functions: Every available function.
        request: The user's natural-language request.

    Returns:
        The full prompt.
    """
    catalogue = "\n".join(describe_function(fn) for fn in functions)
    return (
        "You turn a user request into one function call written as JSON.\n"
        "Pick the function that matches the request and take the argument "
        "values from the request.\n\n"
        f"Available functions:\n{catalogue}\n\n"
        f"Request: {request}\n"
        "Function call:\n"
    )


class ConstrainedDecoder(BaseModel):
    """Generates one valid function call per request.

    Attributes:
        llm: The Small_LLM_Model from llm_sdk (only public methods used).
        vocab: The decoded vocabulary.
        grammar: The state machine built from the function definitions.
        functions: The function definitions, in file order.
    """

    llm: Any
    vocab: Vocabulary
    grammar: CallGrammar
    functions: list[FunctionSpec]

    _mask_cache: dict[State, NDArray[np.int64]] = PrivateAttr(
        default_factory=dict
    )
    _model_calls: int = PrivateAttr(default=0)
    _forced_chunks: int = PrivateAttr(default=0)
    _cache_hits: int = PrivateAttr(default=0)
    _cache_misses: int = PrivateAttr(default=0)

    def encode(self, text: str) -> list[int]:
        """Tokenizes text with the SDK and returns plain integers.

        Args:
            text: Any text.

        Returns:
            The token IDs as a flat Python list.
        """
        nested = self.llm.encode(text).tolist()
        return [int(token_id) for row in nested for token_id in row]

    def allowed_ids(self, state: State) -> NDArray[np.int64]:
        """Lists every token that the grammar accepts in this state.

        Results are cached by grammar state. On a miss, only tokens whose
        first character is accepted are walked through the grammar.

        Args:
            state: The current grammar state.

        Returns:
            The allowed token IDs.
        """
        key = self.grammar.cache_key(state, self.vocab.longest)
        cached = self._mask_cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached
        self._cache_misses += 1

        allowed: list[int] = []
        for first_char, token_ids in self.vocab.by_first_char.items():
            after_first = self.grammar.step(state, first_char)
            if after_first is None:
                continue
            for token_id in token_ids:
                text = self.vocab.texts[token_id]
                if text is None:
                    continue
                if self.grammar.feed(after_first, text[1:]) is not None:
                    allowed.append(token_id)
        result = np.array(allowed, dtype=np.int64)
        self._mask_cache[key] = result
        return result

    def choose_token(self, ids: list[int], state: State) -> int:
        """Asks the model for its scores and keeps the best legal token.

        Args:
            ids: Prompt tokens followed by everything generated so far.
            state: The current grammar state.

        Returns:
            The chosen token ID.

        Raises:
            RuntimeError: If no token is allowed (should never happen).
        """
        logits = np.asarray(
            self.llm.get_logits_from_input_ids(ids), dtype=np.float64
        )
        self._model_calls += 1
        allowed = self.allowed_ids(state)
        allowed = allowed[allowed < logits.size]
        if allowed.size == 0:
            raise RuntimeError("the grammar allows no token")
        mask = np.full(logits.size, -np.inf)
        mask[allowed] = 0.0
        return int(np.argmax(logits + mask))

    def generate(self, request: str) -> CallResult:
        """Produces the function call for one request.

        Args:
            request: The natural-language request.

        Returns:
            The validated call.

        Raises:
            RuntimeError: If generation stops without a complete call.
        """
        ids = self.encode(build_prompt(self.functions, request))
        state = self.grammar.start()
        pieces: list[str] = []
        for _ in range(MAX_STEPS):
            if self.grammar.is_done(state):
                break
            text = self.grammar.forced_text(state)
            if text:
                ids.extend(self.encode(text))
                self._forced_chunks += 1
            else:
                token_id = self.choose_token(ids, state)
                ids.append(token_id)
                text = self.vocab.texts[token_id] or ""
            following = self.grammar.feed(state, text)
            if following is None:
                raise RuntimeError(f"grammar rejected {text!r}")
            state = following
            pieces.append(text)
        if not self.grammar.is_done(state):
            raise RuntimeError("generation stopped before the call ended")
        return self.to_result(request, "".join(pieces))

    def to_result(self, request: str, text: str) -> CallResult:
        """Parses the generated JSON and applies the declared types.

        Args:
            request: The original request.
            text: The generated JSON (valid by construction).

        Returns:
            The validated call, with "number" values as floats.
        """
        call = json.loads(text)
        spec = next(fn for fn in self.functions if fn.name == call["name"])
        parameters: dict[str, bool | int | float | str] = {}
        for name, value in call["parameters"].items():
            declared = spec.parameters[name].type
            if declared == "number":
                parameters[name] = float(value)
            elif declared == "integer":
                parameters[name] = int(value)
            else:
                parameters[name] = value
        return CallResult(
            prompt=request, name=call["name"], parameters=parameters
        )

    def stats(self) -> str:
        """Summarizes how much work the optimizations saved.

        Returns:
            A one-line summary.
        """
        return (
            f"model calls: {self._model_calls}, "
            f"forced chunks: {self._forced_chunks}, "
            f"mask cache hits/misses: "
            f"{self._cache_hits}/{self._cache_misses}"
        )
