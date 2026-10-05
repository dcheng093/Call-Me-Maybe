*This project has been created as part of the 42 curriculum by dcheng.*

# Call Me Maybe

## Description

This project is a lightweight function-calling system built around a local language model. The goal is to convert a natural-language prompt such as “What is the sum of 2 and 3?” into a machine-readable JSON object representing a function call, like:

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

The important idea is that the model is not allowed to “free-form” generate any text it wants. Instead, the program constrains generation so that the output must respect a fixed JSON schema. In practice, the system reads the available functions from a JSON file, builds a grammar that knows which keys and parameter types are allowed, and then masks out invalid next tokens while the model generates text one token at a time.

This project is deliberately small and educational: it is not trying to replace a large production agent framework. It demonstrates the core principle behind function calling in LLMs: combine model inference with structural constraints so the output remains valid and machine-readable.

## Instructions

### 1. Install dependencies

```bash
uv sync
```

### 2. Run the program

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

This is the default flow. If you do not pass the optional arguments, the program uses these defaults automatically:

- `--functions_definition`: `data/input/functions_definition.json`
- `--input`: `data/input/function_calling_tests.json`
- `--output`: `data/output/function_calling_results.json`

### 3. Common project commands

```bash
make install
make run
make debug
make clean
make lint
make lint-strict
```

## How the project works

The implementation is split into a few Python modules, each with a clear role:

- [src/__main__.py](src/__main__.py): command-line entry point and orchestration loop
- [src/parser.py](src/parser.py): converts the raw JSON schema into a simpler format for the grammar checker
- [src/grammar.py](src/grammar.py): state machine that enforces the valid JSON structure for the allowed functions
- [src/tokenizer.py](src/tokenizer.py): reads the model vocabulary and filters valid token IDs based on the grammar state
- [src/validator.py](src/validator.py): validates the final function-call result with Pydantic

The project uses the local model wrapper in `llm_sdk` to access `Qwen/Qwen3-0.6B` and retrieve token logits.

## Algorithm explanation

The actual code follows this flow:

1. Read the command-line arguments.
2. Check that the required input files exist.
3. Load both JSON files.
4. Convert the function definitions into a schema mapping that the grammar can reason about.
5. Instantiate the LLM and tokenizer.
6. For each prompt:
   - build a prompt string with the function schema and the user query
   - encode the text into token IDs
   - repeatedly ask the model for logits for the next token
   - compute the set of valid next token IDs using the grammar
   - apply a mask so invalid tokens are almost completely ignored
   - choose the best remaining token
   - append it to the generation state
   - stop once the generated text closes the JSON object
7. Parse the resulting JSON text as Python data
8. Validate the output with Pydantic
9. Save the results to the output file

This is a grammar-aware generation loop. The model still chooses the next symbol, but the program prevents it from choosing symbols that would make the output invalid.

### Why this is called constrained decoding

A normal language model does this:

- It sees the current text
- It predicts the next token
- It picks the token with the highest score

That works for normal text generation, but it is unreliable for structured output because the model might choose a token that breaks the JSON or violates the function schema.

In this project, we do this instead:

- The model produces logits for all possible next tokens
- The grammar decides which tokens are valid for the current state
- Invalid tokens are masked, meaning their score is set to `-inf` (negative infinity) so they can never be selected
- The model picks only among the allowed tokens

This is the heart of the constrained decoding idea in this repository.

### What the rulebook is doing

The class `TrieJSONRulebook` in [src/grammar.py](src/grammar.py) tracks the current generated text and allows only a small set of legal next characters. For example, it enforces these rules:

- the output must begin with `{`
- the output must contain a JSON object with a `name` field and a `parameters` field
- the selected function name must be one of the names defined in `functions_definition.json`
- each parameter name must exist in the selected function definition
- string values must be quoted strings
- number values must remain numeric
- boolean values must be `true` or `false`

The grammar is therefore not just checking syntax; it is also checking schema consistency.

## Design decisions

### 1. Separate the model from the grammar

The model decides what the next token should be, but the grammar decides whether that token is legal. This separation is the key to the project. The model is still doing language understanding, but the output is steered by strict structure.

### 2. Work with token-level constraints instead of prompt-only instructions

The project does not rely on a prompt like “respond with valid JSON only.” That approach is fragile, especially for small models. Instead, it modifies the generation process directly, token by token.

### 3. Use real vocabulary mapping to filter valid tokens

The tokenizer reads the model’s vocabulary file and converts token IDs into their string form. It then tests whether a candidate token can be appended without violating the grammar. Only valid token IDs remain eligible.

### 4. Validate after generation

Even with all the constraints, the code still performs a final validation pass with Pydantic. This catches shape mismatches or unexpected conversions before writing the output.

### 5. Graceful failure handling

The CLI checks for missing files and malformed JSON. If either input file is broken, the program prints a clear error message and exits instead of crashing unexpectedly.

## Performance analysis

This project prioritizes reliability over raw speed.

- The LLM is small: `Qwen/Qwen3-0.6B`
- Generation is constrained, so the valid choice space is much smaller than unrestricted free-form output
- The vocabulary filter has a cache in `CustomTokenizer`, which reduces repeated work for similar grammar states
- Each prompt is processed with a maximum of 300 generation steps

This does not guarantee perfect model understanding, but it significantly improves reliability when compared with generating unstructured text and hoping it matches the schema.

## Challenges faced

### Challenge 1: small models are not naturally good at valid JSON

A model can easily produce text that is almost correct but slightly malformed. Without structural constraints, this becomes a problem because you need exact JSON before parsing it.

### Challenge 2: token-level constraints are harder than word-level constraints

The model works on subword tokens, not on whole words. That means the grammar must reason about tokens like `"`, `,`, `true`, digits, and spaces, not just “one complete word.”

### Challenge 3: output must follow a function schema, not just any JSON

The generated text must match the allowed function names and parameter names from the JSON schema. This requires a more precise grammar than a generic JSON checker.

### Challenge 4: local inference can be expensive

The model must run directly on the machine, so the implementation tries to stay lightweight and efficient by reusing the current generation state and caching token-mask decisions.

## Testing strategy

The repository includes example prompts in [data/input/function_calling_tests.json](data/input/function_calling_tests.json), and the function definitions in [data/input/functions_definition.json](data/input/functions_definition.json). A typical validation workflow is:

1. Run the script with the default files
2. Inspect whether the output JSON file is created
3. Confirm that every result has:
   - `prompt`
   - `name`
   - `parameters`
4. Ensure the function name exists in the function definitions
5. Ensure every parameter matches the expected type
6. Check that the file is parseable JSON

This is a practical end-to-end test of the whole pipeline.

## Example usage

### Example command

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Example output

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

## Important note about the implementation

This repository does not implement a general-purpose agent that calls external code automatically. Instead, it demonstrates an essential pattern used in real LLM systems: constrain generation to a known schema so output remains valid and predictable.

The code is purposely educational and intentionally narrow in scope. It is focused on one central idea: the model can help choose a function and generate its arguments, but the system must guide it so the final JSON is valid.

## Resources

- [bpe tokenization](https://huggingface.co/learn/nlp-course/en/chapter6/5)
- [qwen tool calling format](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [qwen3 documentation](https://qwen.readthedocs.io/en/latest/)
- [pydantic documentation](https://docs.pydantic.dev/)

### AI usage

AI was used to understand the project requirements, compare implementation strategies, and reason about constrained decoding and vocabulary masking. The code was then reviewed and adapted to fit the actual repository design rather than relying on a purely idealized version of the assignment.

## Glossary

- `argparse`: Python library used to parse command-line arguments.
- `json`: Python package for reading and writing JSON files.
- `os`: Python package for file and directory operations.
- `sys`: Python package for exiting the program and printing errors.
- `numpy`: library for numeric arrays, used here for logits and masking.
- `logits`: raw scores the model assigns to each possible next token.
- `token`: a subword piece of text used by the tokenizer.
- `vocabulary`: the list of tokens known by the model.
- `constrained decoding`: restricting model choices to valid outputs.
- `logit masking`: setting invalid token scores to an extreme negative number.
- `Pydantic`: validation library ensuring the output has the required structure.

This project is best understood as a small demonstration of “LLM + schema + constraints = reliable structured output.”
