*This project has been created as part of the 42 curriculum by dcheng.*

# Call Me Maybe

## Description

Call Me Maybe is a small function-calling system built for the "Introduction to function calling in LLMs" project. The goal is to turn natural-language prompts into structured function calls that are valid JSON and match a schema defined in a JSON file. Instead of asking the model to produce a perfect answer directly, the system constrains generation so the model only chooses among valid tokens and valid function names, which greatly improves reliability for small models.

The project uses the provided local SDK, which exposes a lightweight causal LM wrapper around Qwen/Qwen3-0.6B. The core idea is to let the model choose the right function name and then validate the arguments against a schema before writing the final JSON output.

## Instructions

### 1. Project setup

```bash
cd /path/to/Call Me Maybe
uv sync
```

If you want to work inside the generated virtual environment explicitly, you can still activate it with:

```bash
source .venv/bin/activate
```

### 2. Run the program

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### 3. Common commands

```bash
make install
make run
make debug
make clean
make lint
make lint-strict
```

## Algorithm explanation

The constrained-decoding approach is intentionally simple and robust:

1. Read the available function definitions from JSON.
2. **For each prompt, use the LLM with constrained decoding to select the best matching function name** from the known set. The LLM generates tokens one-by-one, but only valid function name tokens are allowed at each step, guaranteeing the output is always a valid function name.
3. Use the prompt text itself to extract the relevant argument values (numbers, strings, regex patterns, square roots) via deterministic regex and string parsing.
4. Validate the result using Pydantic so the final output always matches the required schema.
5. Write the final JSON array to the output file with 100% valid structure and format.

**Key principle:** The LLM is used for semantic understanding (choosing the right function), while deterministic logic ensures structural validity and type correctness. This combination achieves near-perfect reliability even with a small 0.6B parameter model.

**Constrained decoding process:**
- The LLM produces logits (probability scores) for all possible next tokens.
- Only tokens that would produce valid function names are allowed.
- Invalid tokens are masked to negative infinity, preventing them from being selected.
- This process repeats token-by-token until a complete valid function name is generated.
- Result: 100% valid function selection without relying on spontaneous prompt compliance.

## Design decisions

- **SDK as single model interface:** The provided `llm_sdk` package is used as the exclusive interface to the LLM. It encapsulates all model operations (encoding, logits extraction, decoding) and eliminates the need for external dependencies like PyTorch or Hugging Face transformers.
- **Minimal dependencies:** Only `numpy` and `pydantic` are required. The `llm_sdk` handles all model operations internally.
- **Use Pydantic models for validation:** Function metadata and final outputs are validated against Pydantic schemas to guarantee type correctness and prevent malformed JSON.
- **Keep the implementation modular:** The CLI entrypoint, business logic (pipeline), and validation models (schemas) are clearly separated.
- **Handle errors gracefully:** Malformed inputs, missing files, and edge cases produce clear, user-friendly error messages without crashing.
- **Support flexible paths:** The CLI accepts custom `--functions_definition`, `--input`, and `--output` arguments while providing sensible defaults.

## Performance analysis

The solution is designed for correctness and reliability rather than raw benchmark speed. On a small model such as Qwen/Qwen3-0.6B, the best strategy is not to ask for arbitrary JSON generation, but to constrain the candidate space:

- Function selection is restricted to the known names from the function definition file.
- Argument extraction is deterministic from the prompt text instead of relying on fragile free-form generation.
- The output remains valid and parseable because the final structure is validated before writing.

This gives strong reliability even for small models, while keeping execution simple and fast enough for local experimentation.

## Challenges faced

The main challenge was balancing the assignment requirements with the realistic behavior of a small LLM. Pure unconstrained generation is too brittle, especially for small models. The solution avoids this by reducing the problem to a structured choice followed by typed extraction. Another challenge was the requirement to use the provided SDK without modifying it, while still making the project runnable from the repository root.

## Testing strategy

The program is validated with small end-to-end tests that check:

- The selected function name matches the prompt intent.
- The output object always contains the keys prompt, name, and parameters.
- The numeric arguments keep the correct types and values.
- The CLI can load the default JSON files and produce a valid result file.

## Example usage

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Example output:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

## Resources

- Qwen documentation for model usage and local inference.
- Hugging Face documentation on tokenization and causal language models.
- Pydantic documentation for data validation.
- The project references in the references/ directory for architecture and implementation ideas.

### AI usage

AI was used as a learning aid to understand the project requirements, review the model-usage patterns, and compare different implementation strategies. The final project was checked and adapted so it stays faithful to the assignment requirements and remains understandable and maintainable.
