"""Pydantic models describing the data that enters and leaves the program."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ParamType = Literal["number", "integer", "string", "boolean"]


def check_identifier(value: str, what: str) -> str:
    """Rejects names that cannot be written safely inside a JSON string.

    Function and parameter names are pasted as-is into the JSON the model
    writes, so a quote, a backslash or a control character would break it.

    Args:
        value: The name to check.
        what: Human-readable label used in the error message.

    Returns:
        The unchanged name when it is acceptable.

    Raises:
        ValueError: If the name is empty or has a forbidden character.
    """
    if not value:
        raise ValueError(f"{what} must not be empty")
    for char in value:
        if char in '"\\' or ord(char) < 32:
            raise ValueError(
                f"{what} {value!r} contains the forbidden character "
                f"{char!r} (quotes, backslashes and control characters "
                f"are not allowed)"
            )
    return value


class ParamSpec(BaseModel):
    """Type of one function parameter, e.g. {"type": "number"}.

    Attributes:
        type: One of the JSON types the grammar knows how to generate.
    """

    type: ParamType


class ReturnSpec(BaseModel):
    """Declared return type of a function (only shown to the model).

    Attributes:
        type: Free-form type name, e.g. "number" or "string".
    """

    type: str = "any"


class FunctionSpec(BaseModel):
    """One entry of functions_definition.json.

    Attributes:
        name: Function name the model must write, e.g. "fn_add_numbers".
        description: Sentence explaining what the function does.
        parameters: Parameter names mapped to their types, in order.
        returns: Optional declared return type.
    """

    name: str
    description: str = ""
    parameters: dict[str, ParamSpec] = Field(default_factory=dict)
    returns: ReturnSpec | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Checks that the function name is JSON-safe.

        Args:
            value: The raw function name.

        Returns:
            The validated name.
        """
        return check_identifier(value, "function name")

    @field_validator("parameters")
    @classmethod
    def validate_parameter_names(
        cls, value: dict[str, ParamSpec]
    ) -> dict[str, ParamSpec]:
        """Checks that every parameter name is JSON-safe.

        Args:
            value: The parsed parameters mapping.

        Returns:
            The validated mapping.
        """
        for param_name in value:
            check_identifier(param_name, "parameter name")
        return value


class PromptItem(BaseModel):
    """One entry of function_calling_tests.json.

    Attributes:
        prompt: The natural-language request (may be an empty string).
    """

    prompt: str


class CallResult(BaseModel):
    """One object of the output file.

    Attributes:
        prompt: The original request.
        name: The function chosen by the model.
        parameters: Argument values, typed according to the definition.
    """

    prompt: str
    name: str
    parameters: dict[str, bool | int | float | str]
