from typing import Dict, List, Tuple


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
        normalized_text = "".join(self.text_so_far.split())

        if normalized_text.endswith("}}"):
            return []

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
                for idx, (p_name, p_type) in enumerate(params):
                    expected_key = f'"{p_name}":'

                    if ptr == len(remaining):
                        return [expected_key]

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
                            text_chars = [
                                chr(c) for c in range(32, 127)
                                if chr(c) != '"'
                            ]
                            return text_chars + ['"']
                    elif p_type == "boolean":
                        expected_end = ',' if idx < len(params) - 1 else '}'
                        bool_true = "true"
                        bool_false = "false"
                        written = remaining[ptr:]

                        if bool_true.startswith(written):
                            next_true = (
                                [bool_true[len(written)]]
                                if len(written) < len(bool_true)
                                else [expected_end]
                            )
                        else:
                            next_true = []

                        if bool_false.startswith(written):
                            next_false = (
                                [bool_false[len(written)]]
                                if len(written) < len(bool_false)
                                else [expected_end]
                            )
                        else:
                            next_false = []

                        allowed_bool = list(set(next_true + next_false))
                        if not allowed_bool:
                            return []

                        if written in (bool_true, bool_false):
                            ptr += len(written)
                        else:
                            return allowed_bool
                    else:
                        expected_end = ',' if idx < len(params) - 1 else '}'
                        num_chars = (
                            "0123456789-"
                            if p_type == "integer"
                            else "0123456789.-eE+"
                        )
                        end_found = False

                        for k in range(ptr, len(remaining)):
                            if remaining[k] == expected_end:
                                if k > ptr:
                                    end_found = True
                                    ptr = k
                                    break
                                return []
                            elif remaining[k] not in num_chars:
                                return []

                        if end_found:
                            ptr += 1
                        else:
                            allowed = list(num_chars)
                            if len(remaining) > ptr:
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
