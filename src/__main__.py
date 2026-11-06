"""Command-line entry point, run with: uv run python -m src."""

import argparse
import json
import os
import sys
import time
from typing import Any, NoReturn

from pydantic import TypeAdapter, ValidationError

from src.decoder import ConstrainedDecoder
from src.grammar import CallGrammar
from src.schemas import CallResult, FunctionSpec, PromptItem
from src.vocab import Vocabulary

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calls.json"


def fail(message: str) -> NoReturn:
    """Prints an error message and stops the program with exit code 1.

    Args:
        message: What went wrong, in plain words.
    """
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Reads the optional command-line flags.

    Returns:
        The parsed flags: functions_definition, input and output paths.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Turn natural-language requests into function calls "
                    "using constrained decoding on a small local LLM.",
    )
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS,
                        help="JSON file listing the available functions")
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="JSON file listing the prompts")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="where to write the function calls")
    return parser.parse_args()


def read_json(path: str, label: str) -> Any:
    """Loads a JSON file, turning every failure into a clear message.

    Args:
        path: File to read.
        label: Name of the file for error messages.

    Returns:
        The decoded JSON data.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        fail(f"{label} not found: {path}")
    except json.JSONDecodeError as err:
        fail(f"{label} is not valid JSON ({path}, line {err.lineno}, "
             f"column {err.colno}): {err.msg}")
    except UnicodeDecodeError:
        fail(f"{label} is not UTF-8 text: {path}")
    except OSError as err:
        fail(f"cannot read {label} {path}: {err.strerror or err}")


def describe_errors(err: ValidationError) -> str:
    """Formats pydantic errors as one readable line per problem.

    Args:
        err: The validation error.

    Returns:
        Lines like "  - entry 1 > parameters > a > type: Input should ...".
    """
    lines = []
    for problem in err.errors():
        where = " > ".join(str(part) for part in problem["loc"])
        place = f"entry {where}" if where else "whole file"
        lines.append(f"  - {place}: {problem['msg']}")
    return "\n".join(lines)


def load_functions(path: str) -> list[FunctionSpec]:
    """Reads and validates the function definitions file.

    Args:
        path: Path to functions_definition.json.

    Returns:
        The validated definitions (non-empty, unique names).
    """
    data = read_json(path, "functions definition file")
    try:
        functions = TypeAdapter(list[FunctionSpec]).validate_python(data)
    except ValidationError as err:
        fail(f"invalid functions definition file {path}:\n"
             f"{describe_errors(err)}")
    if not functions:
        fail(f"functions definition file {path} defines no function")
    names = [fn.name for fn in functions]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        fail(f"duplicate function names in {path}: {', '.join(duplicates)}")
    return functions


def load_prompts(path: str) -> list[PromptItem]:
    """Reads and validates the prompts file.

    Args:
        path: Path to function_calling_tests.json.

    Returns:
        The validated prompts (possibly an empty list).
    """
    data = read_json(path, "input file")
    try:
        return TypeAdapter(list[PromptItem]).validate_python(data)
    except ValidationError as err:
        fail(f"invalid input file {path}:\n{describe_errors(err)}")


def prepare_output(path: str) -> None:
    """Checks early that the output file can be created.

    Args:
        path: Where the results will be written.
    """
    if os.path.isdir(path):
        fail(f"output path is a directory: {path}")
    folder = os.path.dirname(path)
    try:
        if folder:
            os.makedirs(folder, exist_ok=True)
    except OSError as err:
        fail(f"cannot create output folder {folder}: {err.strerror or err}")


def build_decoder(functions: list[FunctionSpec]) -> ConstrainedDecoder:
    """Loads the model and vocabulary and assembles the decoder.

    Args:
        functions: The validated definitions.

    Returns:
        A ready-to-use decoder.
    """
    try:
        from llm_sdk import Small_LLM_Model
        llm = Small_LLM_Model()
        vocab = Vocabulary.from_file(llm.get_path_to_vocab_file())
    except Exception as err:
        fail(f"could not load the language model: {err}")
    return ConstrainedDecoder(
        llm=llm,
        vocab=vocab,
        grammar=CallGrammar.from_functions(functions),
        functions=functions,
    )


def write_results(path: str, results: list[CallResult]) -> None:
    """Writes the output file as a JSON array.

    Args:
        path: Output file path.
        results: One validated call per successful prompt.
    """
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([result.model_dump() for result in results], handle,
                      indent=2, ensure_ascii=False)
            handle.write("\n")
    except OSError as err:
        fail(f"cannot write output file {path}: {err.strerror or err}")


def run() -> None:
    """Runs the whole pipeline from command line to output file."""
    args = parse_args()
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)
    prepare_output(args.output)
    print(f"Loaded {len(functions)} functions and {len(prompts)} prompts.")

    print("Loading the model...", flush=True)
    decoder = build_decoder(functions)

    results: list[CallResult] = []
    started = time.perf_counter()
    for number, item in enumerate(prompts, start=1):
        print(f"[{number}/{len(prompts)}] {item.prompt!r}", flush=True)
        prompt_start = time.perf_counter()
        try:
            result = decoder.generate(item.prompt)
        except Exception as err:
            print(f"    failed: {err}", file=sys.stderr)
            continue
        elapsed = time.perf_counter() - prompt_start
        print(f"    -> {result.name} {json.dumps(result.parameters)} "
              f"({elapsed:.1f}s)")
        results.append(result)

    write_results(args.output, results)
    total = time.perf_counter() - started
    print(f"\nWrote {len(results)}/{len(prompts)} calls to {args.output} "
          f"in {total:.1f}s ({decoder.stats()}).")


def main() -> None:
    """Entry point: runs the pipeline and handles Ctrl-C cleanly."""
    try:
        run()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
