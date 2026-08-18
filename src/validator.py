from typing import Any, Dict

from pydantic import BaseModel


class FunctionCallResult(BaseModel):
    """
    Validates result of a single LLM function-call generation.

    Used to enforce field presence and type correctness before results are
    serialised to the output JSON file.  Pydantic will raise a
    ValidationError at construction time if any field is missing or of
    the wrong type, surfacing schema violations early.

    Attributes:
        prompt: The original natural-language request that triggered this
            function call.
        name: The name of the function the model selected to call.
        parameters: A mapping of parameter names to their values, with
            numeric values normalised to float and all others to str.
    """

    prompt: str
    name: str
    parameters: Dict[str, Any]
