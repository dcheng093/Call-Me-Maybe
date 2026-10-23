import json
from typing import Dict, List, Optional, Tuple

NUMBER_CHARS = "0123456789.-eE+"
NUMBER_ACCEPTING = {"zero", "int", "frac", "exp_digits"}


def _next_number_state(
    state: str, char: str, integer_only: bool
) -> Optional[str]:
    """
    Advances the JSON number state machine by one character.

    Follows the JSON number grammar: -?(0|[1-9][0-9]*)(.[0-9]+)?
    ([eE][+-]?[0-9]+)?  Integers stop after the whole part.

    Args:
        state: The current state name ("start" before any character).
        char: The next character to consume.
        integer_only: True if fractions and exponents are not allowed.

    Returns:
        The next state name, or None if the character is not allowed.
    """
    is_digit = char.isdigit()
    if state in ("start", "sign"):
        if state == "start" and char == "-":
            return "sign"
        if char == "0":
            return "zero"
        if is_digit:
            return "int"
        return None
    if state in ("zero", "int"):
        if state == "int" and is_digit:
            return "int"
        if integer_only:
            return None
        if char == ".":
            return "dot"
        if char in "eE":
            return "exp"
        return None
    if state in ("dot", "frac"):
        if is_digit:
            return "frac"
        if state == "frac" and char in "eE":
            return "exp"
        return None
    if state == "exp" and char in "+-":
        return "exp_sign"
    if state in ("exp", "exp_sign", "exp_digits") and is_digit:
        return "exp_digits"
    return None


class TrieJSONRulebook:
    """
    Enforces JSON schema constraints during text generation using a Trie.

    The class maintains the current state of a generated string and determines
    which characters are valid next steps based on a provided schema of
    function definitions and their required parameters.

    Attributes:
        schema: A dictionary mapping function names to a list of tuples,
            where each tuple contains a parameter name and its associated type.
        text_so_far: The sequence of characters generated/processed thus far.
    """
    def __init__(
        self,
        schema_metadata: Dict[str, List[Tuple[str, str]]]
    ) -> None:
        """
        Initializes the TrieJSONRulebook with a specific schema.

        Args:
            schema_metadata: A dictionary where keys are function names and
                values are lists of (parameter_name, parameter_type) tuples.
        """
        self.schema: Dict[str, List[Tuple[str, str]]] = schema_metadata
        self.text_so_far: str = ""

    def get_allowed_characters(self) -> List[str]:
        """
        Determines the valid next characters based on the current
        text state.

        Returns:
            A list of single-character strings that are valid next inputs
            given the current `text_so_far` and the enforced schema.
        """
        text = self.text_so_far.lstrip()
        if not text:
            return ["{", " "]

        prefix_target = '{"name":"'
        if len(text) < len(prefix_target):
            if prefix_target.startswith(text):
                return [prefix_target[len(text)]]
            return []

        if text.startswith(prefix_target):
            after_name_key = text[9:]
            quote_idx = after_name_key.find('"')

            if quote_idx == -1:
                prefix = after_name_key
                allowed = []
                for name in self.schema.keys():
                    if name.startswith(prefix):
                        idx = len(prefix)
                        if idx < len(name):
                            allowed.append(name[idx])
                        else:
                            allowed.append('"')
                return list(set(allowed))

            func_name = after_name_key[:quote_idx]
            if func_name not in self.schema:
                return []

            after_func = after_name_key[quote_idx + 1:]
            expected_mid = ',"parameters":{'
            if len(after_func) < len(expected_mid):
                if expected_mid.startswith(after_func):
                    return [expected_mid[len(after_func)]]
                return []

            if after_func.startswith(expected_mid):
                remaining = after_func[len(expected_mid):]
                params = self.schema[func_name]

                ptr = 0
                if not params:
                    # No parameters: the inner object is just "{}"
                    if ptr == len(remaining):
                        return ['}']
                    if remaining[ptr] != '}':
                        return []
                    ptr += 1

                for idx, (p_name, p_type) in enumerate(params):
                    expected_key = f'"{p_name}":'

                    if ptr == len(remaining):
                        return [expected_key[0]]

                    if remaining[ptr:].startswith(expected_key):
                        ptr += len(expected_key)
                    else:
                        sub = remaining[ptr:]
                        if expected_key.startswith(sub):
                            return [expected_key[len(sub)]]
                        return []

                    if p_type == "string":
                        if ptr == len(remaining):
                            return ['"']
                        if remaining[ptr] != '"':
                            return []
                        ptr += 1

                        found_close = False
                        close_idx = -1
                        for k in range(ptr, len(remaining)):
                            if remaining[k] == '"':
                                backslash_count = 0
                                back_k = k - 1
                                while (
                                    back_k >= ptr and
                                    remaining[back_k] == '\\'
                                ):
                                    backslash_count += 1
                                    back_k -= 1
                                if backslash_count % 2 == 0:
                                    found_close = True
                                    close_idx = k
                                    break
                        if found_close:
                            ptr = close_idx + 1
                        else:
                            trailing = len(remaining) - len(
                                remaining.rstrip('\\')
                            )
                            if trailing % 2 == 1:
                                # Inside an escape: only valid JSON escapes
                                return list('"\\/bfnrt')
                            text_chars = [
                                chr(c) for c in range(32, 127)
                                if chr(c) != '"'
                            ]
                            return text_chars + ['"']
                    elif p_type == "boolean":
                        # Consume the literal only; the separator after it
                        # is handled by the shared code below
                        written = remaining[ptr:]
                        literal_done = False
                        for literal in ("true", "false"):
                            if written.startswith(literal):
                                ptr += len(literal)
                                literal_done = True
                                break
                        if not literal_done:
                            allowed_bool = [
                                literal[len(written)]
                                for literal in ("true", "false")
                                if literal.startswith(written)
                            ]
                            return allowed_bool
                    else:
                        expected_end = ',' if idx < len(params) - 1 else '}'
                        state = "start"

                        # Stop on the separator without consuming it; the
                        # shared separator code below consumes it once
                        while ptr < len(remaining):
                            char = remaining[ptr]
                            if char == expected_end:
                                if state not in NUMBER_ACCEPTING:
                                    return []
                                break
                            next_state = _next_number_state(
                                state, char, p_type == "integer"
                            )
                            if next_state is None:
                                return []
                            state = next_state
                            ptr += 1

                        if ptr == len(remaining):
                            allowed = [
                                char for char in NUMBER_CHARS
                                if _next_number_state(
                                    state, char, p_type == "integer"
                                ) is not None
                            ]
                            if state in NUMBER_ACCEPTING:
                                allowed.append(expected_end)
                            return allowed

                    if idx < len(params) - 1:
                        if ptr == len(remaining):
                            return [',']
                        if remaining[ptr] == ',':
                            ptr += 1
                        else:
                            return []
                    else:
                        if ptr == len(remaining):
                            return ['}']
                        if remaining[ptr] == '}':
                            ptr += 1
                        else:
                            return []

                if ptr == len(remaining):
                    return ['}']
                if remaining[ptr:] == '}':
                    return []
                return []

        return []

    def is_complete(self) -> bool:
        """
        Checks if the generated text is a finished function call.

        The grammar allows nothing after the closing brace, and no proper
        prefix of a JSON object parses, so the call is complete exactly
        when no character is allowed and the text is valid JSON.  This
        avoids stopping early on a string value that contains "}}".

        Returns:
            True if the function call JSON is complete, False otherwise.
        """
        if self.get_allowed_characters():
            return False
        try:
            json.loads(self.text_so_far)
        except json.JSONDecodeError:
            return False
        return True

    def can_walk_token(self, token_string: str) -> bool:
        """
        Checks if a full token can be appended without violating the schema.

        This method simulates appending the token string character-by-character
        to verify if the entire sequence remains valid according to the
        rulebook. It does not permanently modify the state.

        Args:
            token_string: The sequence of characters to validate.

        Returns:
            True if the entire token is valid, False otherwise.
        """
        original_text = self.text_so_far

        for char in token_string:
            allowed = self.get_allowed_characters()
            if not allowed or char not in allowed:
                self.text_so_far = original_text
                return False
            self.text_so_far += char

        self.text_so_far = original_text
        return True

    def advance(self, token_string: str) -> None:
        """
        Permanently advances the internal state by appending a token.

        Args:
            token_string: The sequence of characters to append
            to `text_so_far`.
        """
        self.text_so_far += token_string
