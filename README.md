*This project has been created as part of the 42 curriculum by dcheng.*

# Call Me Maybe

## Description

Call Me Maybe turns natural-language requests into **function calls**. Given the request
"What is the sum of 2 and 3?" and a list of available functions, it does not answer "5".
Instead it outputs which function to call and with which typed arguments:

```json
{"prompt": "What is the sum of 2 and 3?", "name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

The model used is **Qwen3-0.6B**, a small local LLM that on its own rarely produces valid
JSON. The project makes it reliable with **constrained decoding**: at every generation step,
every token that would break the JSON structure or the function schema has its logit set to
negative infinity, so the model can only ever produce a valid call. The model still makes
every decision (which function, which values); the program only removes illegal options.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
make install        # uv sync: installs numpy, pydantic, llm_sdk, flake8, mypy
make run            # uv run python -m src (default paths)
make debug          # run under pdb
make lint           # flake8 + mypy with the subject's flags
make lint-strict    # flake8 + mypy --strict
make clean          # remove caches and data/output
```

Run with explicit paths (all three flags are optional):

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

The first run downloads the model (~1.5 GB) into the Hugging Face cache.

## Algorithm explanation

### 1. Compiling the functions into a grammar
Each function definition is compiled into a *plan*: a list of segments that must be written
in order. Literal segments are fixed text; value segments are written by the model.
For `fn_add_numbers(a: number, b: number)`:

```
lit  {"name":"fn_add_numbers","parameters":{"a":
num
lit  ,"b":
num
lit  }}
```

The output is compact JSON (no whitespace outside strings), which keeps the grammar small.

### 2. A character-level state machine
The state is a small tuple `(candidates, segment, position, extra)`:
- `candidates`: functions whose plan still matches what has been written. At the start,
  every function is a candidate; each character of the name removes the ones that no longer
  match. This is how the model's choice of function is enforced without any heuristic.
- `segment` / `position`: where we are in the plan.
- `extra`: sub-state for values (number automaton state, escape flag, boolean letters).

Value rules:
- **number**: `-?(0|[1-9][0-9]*)(\.[0-9]+)?`, at most 24 characters, no exponent (so a value
  can never overflow to `Infinity`, which is not valid JSON); **integer**: same without a
  fraction.
- **string**: any character except raw control characters; `\"`, `\\`, `\/`, `\b`, `\f`,
  `\n`, `\r` and `\t` escapes are allowed; at most 512 characters, then only the closing quote is.
- **boolean**: `true` or `false`.

A value ends when the next character belongs to the following literal (`,`, `}` or `"`).

### 3. From characters to tokens
The model works with tokens (pieces of text), not characters. The vocabulary file maps each
token to an ID using byte-level BPE notation (`Ġ` = space, `Ċ` = newline, …). At startup
every token is decoded once into its real text using the GPT-2 `bytes_to_unicode` table.
A token is **allowed** in a state if feeding all of its characters to the state machine
never fails.

### 4. The generation loop
For each prompt:
1. Encode the prompt (function signatures + descriptions + request) with the SDK.
2. Repeat until the state machine reports a complete call:
   - If only one continuation is possible (for example `","parameters":{"a":` once the
     function is known), encode it and append it **without calling the model**.
   - Otherwise get the logits, build a mask (`-inf` everywhere, `0` on allowed IDs), add it
     to the logits and take the `argmax` (greedy decoding).
3. Parse the text with `json.loads` (guaranteed to succeed), convert `number` values to
   `float`, and validate the result with pydantic.

### Why the output is always valid
Every reachable state is a prefix of at least one valid call, every state has at least one
allowed token (all 256 single-byte tokens exist), and strings and numbers have a maximum
length. So generation always ends, and it can only end on a complete, schema-compliant call.

## Design decisions

- **Plans of segments instead of a generic JSON parser**: the output shape is fully known
  from the definitions, so only the values need real parsing. Function names and keys are
  just literals.
- **Incremental state + pure `step` function**: checking a token only feeds its own
  characters, instead of re-parsing everything generated so far.
- **Function choice left entirely to the model**: while several functions are possible,
  nothing past `{"name":"` is forced, so the model writes the whole name in its own tokens.
  (Forcing the common prefix `fn_` split names into unnatural tokens and dropped accuracy
  from 11/11 to 7/11.)
- **Pydantic everywhere**: input files are validated with `TypeAdapter`, so a malformed file
  produces a precise message (`entry 0 > parameters > a > type: Input should be ...`).
- **No hardcoding**: the prompt and the grammar are built from whatever definitions file is
  given.
- **Errors**: every failure (missing file, unreadable file, invalid JSON, bad schema, empty
  or duplicate functions, model loading, unwritable output, Ctrl-C) ends with a one-line
  message and a non-zero exit code, never a traceback.

## Performance analysis

Measured on the 11 provided prompts:

| | GPU (CUDA) | CPU only |
|---|---|---|
| Generation time (all prompts) | ~4.5 s | ~70 s |
| Total run time, including model loading | ~50 s | ~2 min |
| Correct function | 11/11 | 11/11 |
| Valid, schema-compliant JSON | 11/11 | 11/11 |

Speed-ups:
- **Forced literals**: 32 chunks of fixed text are appended without a model call; only
  100 model calls are needed for all 11 prompts.
- **Mask cache keyed by grammar state**: inside a string the allowed set does not depend on
  what was written, so it is computed once and reused (80 hits / 20 misses).
- **First-character index**: on a cache miss only tokens whose first character is legal are
  checked.
- **Compact prompt**: signatures instead of the raw JSON definitions, so every forward pass
  is shorter.

## Challenges faced

- **Tokens mix structure and content**: tokens like `",` or `"}}` close a string and continue
  the JSON in one go. Solved by checking every character of a token against the state
  machine instead of classifying tokens.
- **Byte-level BPE**: tokens are stored as remapped bytes. Replacing only `Ġ`/`Ċ` gives wrong
  text for accented characters; the full `bytes_to_unicode` table fixes it.
- **Wrong function chosen**: forcing the shared prefix `fn_` made the model choose
  `fn_get_square_root` for "Greet shrek". Letting the model write the full name fixed it.
- **Termination and overflow**: without limits, a string could grow forever and a number like
  `1e400` would become `Infinity`. Length caps and a number grammar without exponents solve
  both.
- **Speed on CPU**: one forward pass re-reads the whole sequence, so the number of model calls
  was reduced (forced literals) and masks are cached.

## Testing strategy

- **Grammar unit checks (no model)**: valid calls are accepted (negative and decimal numbers,
  empty strings, escaped quotes, booleans, integers, a function with no parameters); invalid
  ones are rejected (leading zeros, exponents, `2.`, a string where a number is expected,
  unknown function, wrong parameter order, raw newline in a string, invalid escape, trailing
  text, missing brace); length caps hold; forced text is correct.
- **End-to-end**: the provided files, plus an unseen function set with integer, boolean and
  zero-parameter functions and edge-case prompts (empty prompt, quotes, backslashes,
  accents, a 20-digit number).
- **Output checker**: the output is valid JSON, each entry has exactly `prompt`, `name`,
  `parameters`, the name exists, and every argument is present with the declared type.
- **Error paths**: missing file, directory instead of file, invalid JSON, top-level object
  instead of list, wrong prompt type, unknown parameter type, `parameters` as a list, name
  containing a quote, duplicate names, empty function list, output path that is a directory.
- **Lint**: `make lint` and `make lint-strict` pass.

## Example usage

```bash
$ make run
Loaded 5 functions and 11 prompts.
Loading the model...
[1/11] 'What is the sum of 2 and 3?'
    -> fn_add_numbers {"a": 2.0, "b": 3.0} (0.7s)
[3/11] 'Greet shrek'
    -> fn_greet {"name": "shrek"} (0.4s)
[9/11] 'Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS'
    -> fn_substitute_string_with_regex {"source_string": "Hello 34 I'm 233 years old", "regex": "\\d+", "replacement": "NUMBERS"} (1.2s)
...
Wrote 11/11 calls to data/output/function_calls.json in 4.6s (model calls: 100, forced chunks: 32, mask cache hits/misses: 80/20).

$ uv run python -m src --input missing.json
Error: input file not found: missing.json
```

## Resources

- [Hugging Face NLP course: Byte-Pair Encoding tokenization](https://huggingface.co/learn/nlp-course/chapter6/5)
- [OpenAI GPT-2 encoder (origin of `bytes_to_unicode`)](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- [JSON specification (RFC 8259)](https://www.rfc-editor.org/rfc/rfc8259)
- [Willard & Louf, "Efficient Guided Generation for Large Language Models" (2023)](https://arxiv.org/abs/2307.09702)
- [Qwen function calling documentation](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
- [Pydantic documentation](https://docs.pydantic.dev/)
- [NumPy documentation](https://numpy.org/doc/)

### How AI was used
- explain the subject and the underlying concepts (tokens, BPE, logits, constrained
  decoding, finite-state machines);
- structure the README

I then studied the code line by line, so that I can explain and modify every part of it.
