"""Character-level state machine that only accepts valid function calls.

Every function is compiled into a "plan": a list of segments that must be
written one after the other. Literal segments are fixed text (JSON
punctuation, keys and the function name); value segments are a string,
a number, an integer or a boolean written by the model.

For fn_add_numbers(a: number, b: number) the plan is:

    lit  '{"name":"fn_add_numbers","parameters":{"a":'
    num
    lit  ',"b":'
    num
    lit  '}}'

A state is a plain tuple (candidates, segment, position, extra):
    candidates: functions whose plan still matches what was written
    segment:    index of the current segment in the plan
    position:   characters already written inside that segment
    extra:      detail for value segments (number sub-state, "\\" while
                inside an escape sequence, letters of a boolean)
"""

from typing import Literal

from pydantic import BaseModel

from src.schemas import FunctionSpec

MAX_STRING_CHARS = 512
MAX_NUMBER_CHARS = 24
DIGITS = "0123456789"
STRING_ESCAPES = '"\\/bfnrt'
BOOLEAN_WORDS = ("true", "false")
NUMBER_ACCEPTING = ("zero", "int", "frac")
NAME_START = len('{"name":"')

SegmentKind = Literal["lit", "str", "num", "int", "bool"]
State = tuple[tuple[str, ...], int, int, str]

KIND_OF_TYPE: dict[str, SegmentKind] = {
    "string": "str",
    "number": "num",
    "integer": "int",
    "boolean": "bool",
}


def number_transition(
    current: str, char: str, allow_fraction: bool
) -> str | None:
    """Advances the JSON number sub-machine by one character.

    Accepted shape: -?(0|[1-9][0-9]*)(.[0-9]+)?  (no exponent, so a
    number can never overflow to infinity).

    Args:
        current: Sub-state ("start", "minus", "zero", "int", "dot",
            "frac").
        char: The next character.
        allow_fraction: False for "integer" parameters.

    Returns:
        The next sub-state, or None if the character cannot continue the
        number.
    """
    if current == "start" and char == "-":
        return "minus"
    if current in ("start", "minus"):
        if char == "0":
            return "zero"
        if char in DIGITS:
            return "int"
        return None
    if current == "int" and char in DIGITS:
        return "int"
    if current in ("zero", "int") and char == "." and allow_fraction:
        return "dot"
    if current in ("dot", "frac") and char in DIGITS:
        return "frac"
    return None


class Segment(BaseModel):
    """One piece of a function's plan.

    Attributes:
        kind: "lit" for fixed text, otherwise the kind of value.
        text: The fixed text (only used when kind is "lit").
    """

    kind: SegmentKind
    text: str = ""


def build_plan(function: FunctionSpec) -> list[Segment]:
    """Compiles a function definition into its list of segments.

    The opening quote of a string value is part of the literal before it
    and the closing quote starts the literal after it. Two literals are
    never next to each other.

    Args:
        function: The validated function definition.

    Returns:
        The plan, always starting and ending with a literal.
    """
    plan: list[Segment] = []
    literal = '{"name":"' + function.name + '","parameters":{'
    for index, (param_name, spec) in enumerate(function.parameters.items()):
        if index > 0:
            literal += ","
        literal += '"' + param_name + '":'
        kind = KIND_OF_TYPE[spec.type]
        if kind == "str":
            literal += '"'
        plan.append(Segment(kind="lit", text=literal))
        plan.append(Segment(kind=kind))
        literal = '"' if kind == "str" else ""
    plan.append(Segment(kind="lit", text=literal + "}}"))
    return plan


class CallGrammar(BaseModel):
    """Decides, character by character, what the model may write.

    Attributes:
        plans: Function name mapped to its compiled plan.
    """

    plans: dict[str, list[Segment]]

    @classmethod
    def from_functions(cls, functions: list[FunctionSpec]) -> "CallGrammar":
        """Builds the grammar for a list of function definitions.

        Args:
            functions: Validated definitions with unique names.

        Returns:
            The grammar.
        """
        return cls(plans={fn.name: build_plan(fn) for fn in functions})

    def start(self) -> State:
        """Returns the state before anything has been written.

        Returns:
            All functions are candidates, first segment, position 0.
        """
        return (tuple(self.plans), 0, 0, "")

    def is_done(self, state: State) -> bool:
        """Tells whether a complete function call has been written.

        Args:
            state: The current state.

        Returns:
            True once every segment of the chosen plan is consumed.
        """
        candidates, segment, _, _ = state
        return segment == len(self.plans[candidates[0]])

    def enter(self, candidates: tuple[str, ...], segment: int) -> State:
        """Returns the state at the very beginning of a segment.

        Args:
            candidates: Functions still possible.
            segment: Index of the segment being entered.

        Returns:
            A state at position 0 with the right starting extra.
        """
        plan = self.plans[candidates[0]]
        if segment < len(plan) and plan[segment].kind in ("num", "int"):
            return (candidates, segment, 0, "start")
        return (candidates, segment, 0, "")

    def step(self, state: State, char: str) -> State | None:
        """Feeds one character to the state machine.

        Args:
            state: The current state.
            char: A single character.

        Returns:
            The new state, or None if the character is not allowed here.
        """
        candidates, segment, _, _ = state
        plan = self.plans[candidates[0]]
        if segment >= len(plan):
            return None
        kind = plan[segment].kind
        if kind == "lit":
            return self.step_literal(state, char)
        if kind == "str":
            return self.step_string(state, char)
        if kind == "bool":
            return self.step_boolean(state, char)
        return self.step_number(state, char, kind == "num")

    def step_literal(self, state: State, char: str) -> State | None:
        """Matches one character of fixed text.

        While several functions are still possible, only those whose text
        has this character at this position survive. This is where the
        model's choice of function happens.

        Args:
            state: A state inside a literal segment.
            char: The next character.

        Returns:
            The new state, or None if no candidate accepts the character.
        """
        candidates, segment, position, _ = state
        survivors = tuple(
            name for name in candidates
            if self.plans[name][segment].text[position:position + 1] == char
        )
        if not survivors:
            return None
        if position + 1 == len(self.plans[survivors[0]][segment].text):
            return self.enter(survivors, segment + 1)
        return (survivors, segment, position + 1, "")

    def step_string(self, state: State, char: str) -> State | None:
        """Accepts one character inside a string value.

        Args:
            state: A state inside a string segment.
            char: The next character.

        Returns:
            The new state, or None if the character is not allowed.
        """
        candidates, segment, position, escaping = state
        if escaping:
            if char in STRING_ESCAPES:
                return (candidates, segment, position + 1, "")
            return None
        if char == '"':
            return self.step(self.enter(candidates, segment + 1), char)
        if position >= MAX_STRING_CHARS or ord(char) < 32:
            return None
        if char == "\\":
            return (candidates, segment, position + 1, "\\")
        return (candidates, segment, position + 1, "")

    def step_number(
        self, state: State, char: str, allow_fraction: bool
    ) -> State | None:
        """Accepts one character of a number or integer value.

        Args:
            state: A state inside a number segment.
            char: The next character.
            allow_fraction: False for "integer" parameters.

        Returns:
            The new state, or None if the character is not allowed.
        """
        candidates, segment, position, current = state
        finished = current in NUMBER_ACCEPTING
        following = number_transition(current, char, allow_fraction)
        if following is not None and not (
            finished and position >= MAX_NUMBER_CHARS
        ):
            return (candidates, segment, position + 1, following)
        if finished:
            return self.step(self.enter(candidates, segment + 1), char)
        return None

    def step_boolean(self, state: State, char: str) -> State | None:
        """Accepts one letter of "true" or "false".

        Args:
            state: A state inside a boolean segment.
            char: The next character.

        Returns:
            The new state, or None if the letters match neither word.
        """
        candidates, segment, position, typed = state
        typed += char
        if typed in BOOLEAN_WORDS:
            return self.enter(candidates, segment + 1)
        if any(word.startswith(typed) for word in BOOLEAN_WORDS):
            return (candidates, segment, position + 1, typed)
        return None

    def feed(self, state: State, text: str) -> State | None:
        """Feeds a whole piece of text (for example one token).

        Args:
            state: The current state.
            text: Characters to feed in order.

        Returns:
            The state after the last character, or None as soon as one
            character is rejected.
        """
        current: State | None = state
        for char in text:
            if current is None:
                return None
            current = self.step(current, char)
        return current

    def forced_text(self, state: State) -> str:
        """Returns the text that is the only possible continuation.

        Inside a literal of the chosen function, the rest of the literal
        is forced. While several functions are still possible, only the
        text before the function name is forced: the model must write the
        name itself, in its own tokens, so the choice is entirely its own
        (forcing a shared prefix such as "fn_" splits the name into
        unnatural tokens and makes the model pick the wrong function).
        Inside a value nothing is forced.

        Args:
            state: The current state.

        Returns:
            The forced text, or "" when the model has a real choice.
        """
        candidates, segment, position, _ = state
        plan = self.plans[candidates[0]]
        if segment >= len(plan) or plan[segment].kind != "lit":
            return ""
        rest = plan[segment].text[position:]
        if len(candidates) > 1:
            return rest[:max(0, NAME_START - position)]
        return rest

    def cache_key(self, state: State, longest_token: int) -> State:
        """Returns a key that identifies which tokens are allowed.

        Two states with the same key allow exactly the same tokens. Inside
        a string or number, the position only matters when the length cap
        is closer than the longest token, so it is clamped.

        Args:
            state: The current state.
            longest_token: Length of the longest token text.

        Returns:
            The cache key.
        """
        candidates, segment, position, extra = state
        plan = self.plans[candidates[0]]
        if segment < len(plan) and plan[segment].kind in ("str", "num", "int"):
            cap = (MAX_STRING_CHARS if plan[segment].kind == "str"
                   else MAX_NUMBER_CHARS)
            return (candidates, segment,
                    min(cap - position, longest_token), extra)
        return state
