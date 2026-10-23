# Call Me Maybe: Evaluation Prep Guide

Everything you need to explain, defend, and demo this project in a peer evaluation.
Every example output in this file comes from a real run on this repo (2026-10-22, RTX 3060, `Qwen/Qwen3-0.6B`), not from guesswork. The grammar bugs this guide originally described have been fixed, and the logs and examples here come from the fixed code (see §3 and §9.1).

---

## Table of contents

1. [The 30-second pitch](#1-the-30-second-pitch)
2. [Pre-evaluation checklist](#2-pre-evaluation-checklist)
3. [Reading the log (and the bugs that used to be in it)](#3-reading-the-log-and-the-bugs-that-used-to-be-in-it)
4. [The pipeline end to end](#4-the-pipeline-end-to-end)
5. [Module by module, with examples](#5-module-by-module-with-examples)
6. [Core concepts you must be able to explain](#6-core-concepts-you-must-be-able-to-explain)
7. [Live tests the evaluator might run](#7-live-tests-the-evaluator-might-run)
8. [Likely evaluator questions and answers](#8-likely-evaluator-questions-and-answers)
9. [Known weaknesses and how to fix or defend them](#9-known-weaknesses-and-how-to-fix-or-defend-them)
10. [Command cheat sheet](#10-command-cheat-sheet)

---

## 1. The 30-second pitch

> "A small LLM (Qwen3-0.6B) reads a natural-language request like *'What is the sum of 2 and 3?'* and has to output a JSON function call like `{"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}`.
> Small models are bad at producing exact JSON, so I don't just *ask* for JSON. I **constrain decoding**: at every step the model scores all ~151k tokens (logits), my grammar works out which tokens are legal next, and I set every illegal token's score to `-inf`. The model can only pick from tokens that keep the output valid. At the end, Pydantic validates the result and it's written to a JSON file."

Three key ideas to name:
- **Logits**: the raw scores the model gives each possible next token.
- **Logit masking**: adding `-inf` to illegal tokens so `argmax` can never pick them.
- **Grammar / state machine**: code that knows which characters may come next, given the text generated so far and the function schemas.

---

## 2. Pre-evaluation checklist

| Check | Command | Expected |
|---|---|---|
| Dependencies install | `make install` (or `uv sync`) | no errors |
| flake8 clean | `make lint` | no output from flake8 |
| mypy clean | `make lint-strict` | `Success: no issues found in 6 source files` ✅ (verified) |
| Model cached (avoids a slow download during the eval) | `ls ~/.cache/huggingface/hub` | `models--Qwen--Qwen3-0.6B` ✅ (present) |
| Program runs | `make run` | creates `data/output/function_calling_results.json`, 11/11 prompts succeed ✅ (verified) |
| Clean state | `make clean` | removes `__pycache__`, `.mypy_cache`, `data/output` |
| README has the required sections | open `README.md` | Description, Instructions, Resources, AI usage ✅ |
| `data/output/` not committed | `.gitignore` | ✅ already ignored |
| Uncommitted changes | `git status` | `README.md` (the extra BPE link), `src/grammar.py` and `src/__main__.py` (the bug fixes), and this guide. Commit them if you want them graded. |

**Timing:** the full 11-prompt file took 4m52s end to end on this machine, mostly model load plus vocabulary scanning (every cache miss tests all 151 643 vocab tokens against the grammar). Start `make run` early in the eval and talk while it runs.

---

## 3. Reading the log (and the bugs that used to be in it)

This is the real log for `"What is the sum of 2 and 3?"` after the fixes. `Allowed` is the number of tokens the grammar allowed when that step's token was picked, and `State` is the text after the step.

```
[Step 000] Allowed: 90    | Masked: 151553 | State: ' {"'
[Step 001] Allowed: 4     | Masked: 151639 | State: ' {"name'
[Step 002] Allowed: 3     | Masked: 151640 | State: ' {"name":'
[Step 003] Allowed: 2     | Masked: 151641 | State: ' {"name":"'
[Step 004] Allowed: 2     | Masked: 151641 | State: ' {"name":"fn'
[Step 005] Allowed: 16    | Masked: 151627 | State: ' {"name":"fn_add'
[Step 006] Allowed: 5     | Masked: 151638 | State: ' {"name":"fn_add_numbers'
[Step 007] Allowed: 3     | Masked: 151640 | State: ' {"name":"fn_add_numbers",'
[Step 008] Allowed: 1     | Masked: 151642 | State: ' {"name":"fn_add_numbers","'
[Step 009] Allowed: 7     | Masked: 151636 | State: ' {"name":"fn_add_numbers","parameters'
[Step 010] Allowed: 3     | Masked: 151640 | State: ' {"name":"fn_add_numbers","parameters":{"'
[Step 011] Allowed: 1     | Masked: 151642 | State: ' {"name":"fn_add_numbers","parameters":{"a'
[Step 012] Allowed: 3     | Masked: 151640 | State: ' {"name":"fn_add_numbers","parameters":{"a":'
[Step 013] Allowed: 11    | Masked: 151632 | State: ' {"name":"fn_add_numbers","parameters":{"a":2'
[Step 014] Allowed: 15    | Masked: 151628 | State: ' {"name":"fn_add_numbers","parameters":{"a":2,"'
[Step 015] Allowed: 1     | Masked: 151642 | State: ' {"name":"fn_add_numbers","parameters":{"a":2,"b'
[Step 016] Allowed: 3     | Masked: 151640 | State: ' {"name":"fn_add_numbers","parameters":{"a":2,"b":'
[Step 017] Allowed: 11    | Masked: 151632 | State: ' {"name":"fn_add_numbers","parameters":{"a":2,"b":3'
[Step 018] Allowed: 15    | Masked: 151628 | State: ' {"name":"fn_add_numbers","parameters":{"a":2,"b":3}}'

Final Structural Closure Confirmed: ' {"name":"fn_add_numbers","parameters":{"a":2,"b":3}}'
✅ Success: Generated fn_add_numbers
```

**Every step is constrained.** The parameter keys (`"a":`, `"b":`) are forced, the values can only be valid numbers (11 tokens allowed at step 013: the digits `0-9` and `-`), and the model closes the JSON with a single `}}` token. In the full 11-prompt run, `Allowed:` never showed `0` and all 11 prompts succeeded.

Inside a string value the count jumps to about 90 000 (e.g. `Allowed: 89975` while writing `"shrek"`), because any printable ASCII token that doesn't close the string early is legal there. That's expected: the grammar can't know what the name is, only that it must be a string.

### What the log looked like before the fixes

```
[Step 011] Allowed: 1     | Masked: 151642 | State: ' {"name":"fn_add_numbers","parameters":{'
[Step 012] Allowed: 0     | Masked: 151643 | State: ' {"name":"fn_add_numbers","parameters":{ "'
[Step 013] Allowed: 0     | Masked: 151643 | State: ' {"name":"fn_add_numbers","parameters":{ "a'
 ...
[Step 021] Allowed: 0     | Masked: 151643 | State: '... "a":2, "b":3 }}\n\n'
```

From step 012 on, nothing was allowed, and the old fallback (`custom_mask.fill(0.0)`) **turned the constraint off completely**. Every parameter was generated free-form, and the output was only right because the model happened to write good JSON. Three bugs were behind it:

- **Bug A, key start** (`grammar.py`, was line 91): at the start of a key the grammar returned the whole key `'"a":'` as if it were one character. `can_walk_token` compares one character at a time, so `'"'` never matched and no token passed. Fix: `return [expected_key[0]]`.
- **Bug B, double separator**: the number branch stepped past the `,`/`}`, and then the shared separator code asked for *another* one (`'…"a":2.0,'` → `[',']`, so the grammar's happy path was the invalid `2.0,,"b"`). Booleans had the same problem in a different form. Fix: the value branches stop *on* the separator and the shared code consumes it once.
- **Bug C, whitespace**: once the mask was off, the model picked `' "'` (leading space), which the grammar has no rule for. With A and B fixed and the fallback changed to "stop", the mask is never off, so the model can't pick that token.

See §9.1 for the full list of fixes, including a few the original log didn't show.

> **Tip for the eval:** run `make run`, point at the `Allowed:` column, and say "it never hits 0. If the grammar ever dead-ends, generation stops instead of silently running unconstrained." Then show the before log above and explain the bug. Evaluators respect "I know exactly where the bug was and why."

---

## 4. The pipeline end to end

```
functions_definition.json ──► parser.py ──► schema_metadata {fn: [(param, type)]}
                                                     │
function_calling_tests.json ──► for each prompt:     ▼
                                  build prompt ─► model.encode ─► input_ids
                                         │
                       ┌─────────────────┴──────────────────┐
                       │  loop (max 300 steps):             │
                       │   logits = model(input_ids)        │  151 936 scores
                       │   allowed = tokenizer(rulebook)    │  grammar filter
                       │   logits += mask (-inf if illegal) │
                       │   tok = argmax(logits)             │  greedy pick
                       │   rulebook.advance(tok_text)       │
                       │   input_ids.append(tok)            │
                       │   stop when rulebook.is_complete() │
                       │   (or if no token is allowed)      │
                       └─────────────────┬──────────────────┘
                                         ▼
                        json.loads ─► type-normalize ─► Pydantic ─► results[]
                                                                       │
                                     data/output/function_calling_results.json
```

### Worked trace for "Greet shrek"

1. Prompt sent to the model (`__main__.py:131-135`):
   ```
   Functions schemas:
   [{"name": "fn_add_numbers", ...}, {"name": "fn_greet", ...}, ...]

   Query: Greet shrek

   Correct JSON output call:
   ```
2. Grammar forces `{"name":"`, then allows only `f`, then `n`, then `_`.
3. After `{"name":"fn_` the grammar allows `['r','s','g','a']`. These are the next letters of `fn_reverse_string`, `fn_substitute…`, `fn_greet`, `fn_add_numbers`, `fn_get_square_root` (`g` covers two names). The model picks `g` (actually the token `_g`).
4. After `fn_g` the allowed letters are `['r','e']` (`fn_g`**r**`eet` vs `fn_g`**e**`t_square_root`). The model picks the token `reet`.
5. `,"parameters":{` is forced.
6. `"name":` is forced, then the value can be any string. The model writes `sh`, `rek`, and the closing `"}` as separate tokens (`Allowed: 89975` at each of those steps, see §3).
7. The model picks the single token `}`. Now `rulebook.is_complete()` is true, so the loop stops after 17 steps.
8. `json.loads` → `{"name": "fn_greet", "parameters": {"name": "shrek"}}`, then Pydantic, then output.

Real output:
```json
{ "prompt": "Greet shrek", "name": "fn_greet", "parameters": { "name": "shrek" } }
```

---

## 5. Module by module, with examples

### 5.1 `src/parser.py`: `parse_function_schemas`

Turns the verbose JSON definition into the minimal form the grammar needs. It keeps **parameter order**, because the grammar forces keys in definition order.

**Example 1**: the real input file:
```python
{'fn_add_numbers': [('a', 'number'), ('b', 'number')],
 'fn_greet': [('name', 'string')],
 'fn_reverse_string': [('s', 'string')],
 'fn_get_square_root': [('a', 'number')],
 'fn_substitute_string_with_regex': [('source_string', 'string'),
                                     ('regex', 'string'),
                                     ('replacement', 'string')]}
```

**Example 2**: a parameter with no `"type"` defaults to `"string"`:
```python
parse_function_schemas([{"name": "f", "parameters": {"x": {}}}])
# {'f': [('x', 'string')]}
```

**Example 3**: a function with no parameters:
```python
parse_function_schemas([{"name": "fn_ping"}])
# {'fn_ping': []}       → grammar then expects {"name":"fn_ping","parameters":{}}
```
(Before the fix, the grammar dead-ended at `…"parameters":{}`, because it never consumed the inner `}` when there were no parameters. `grammar.py:135-141` now handles that case.)

Why a list of tuples rather than a dict? Order matters for the grammar, and a tuple is an obvious, immutable pair. (Since Python 3.7, dicts also keep insertion order. The list just makes the intent explicit.)

### 5.2 `src/grammar.py`: `TrieJSONRulebook`

A **state machine**: given `text_so_far`, `get_allowed_characters()` returns the legal next characters. It is called a "Trie" because function-name matching works like walking a prefix tree. At each character, only names that share the prefix typed so far remain.

The output must have exactly this shape:
```
{"name":"<one of the schema names>","parameters":{"<p1>":<v1>,"<p2>":<v2>}}
```

**Real outputs from `get_allowed_characters()`:**

| `text_so_far` | Allowed next | Why |
|---|---|---|
| `''` | `['{', ' ']` | must open the object (leading space tolerated) |
| `'{'` | `['"']` | fixed prefix `{"name":"` |
| `'{"na'` | `['m']` | still inside the fixed prefix |
| `'{"name":"'` | `['f']` | every function starts with `f` |
| `'{"name":"fn_'` | `['r','s','g','a']` | trie branch: reverse / substitute / greet+get / add |
| `'{"name":"fn_g'` | `['r','e']` | greet vs get_square_root |
| `'{"name":"fn_add_numbers"'` | `[',']` | name finished, start `,"parameters":{` |
| `'{"name":"fn_hack"'` | `[]` | unknown function, dead end |
| `'…"parameters":{'` | `['"']` | first character of the first key (`"a":` is then forced one char at a time) |
| `'…"a":'` (number) | `-` and `0-9` | a number starts with a sign or a digit |
| `'…"a":2'` | `0-9 . e E ,` | more digits, a fraction, an exponent, or the separator |
| `'…"a":0'` | `. e E ,` | JSON forbids leading zeros, so no digit after `0` |
| `'…"a":2.'` | `0-9` | a `.` must be followed by a digit |
| `'…"a":1-'` | `[]` | a `-` in the middle is rejected |
| `'…"a":2,'` | `['"']` | separator consumed once, next key starts |
| `'…{"name":"shr'` (string) | 95 chars | any printable ASCII, including closing `"` |
| `'…{"name":"shr\'` | `" \ / b f n r t` | only valid JSON escapes after a backslash |
| `'…{"name":"a}}'` | 95 chars | `}}` inside a string does **not** end generation |
| `'…{"name":"shrek"'` | `['}']` | last parameter, so close the parameters object |
| `'…"b":3}'` | `['}']` | close the outer object |
| `'…"b":3}}'` | `[]` | finished (`is_complete()` is `True`) |

The number branch is a small state machine (`_next_number_state`, `grammar.py:8-51`) that follows the JSON number grammar `-?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?`. For `integer` parameters it stops after the whole part, so `.` and `e` are never allowed.

**Boolean example** (`{"on": boolean}`, the only parameter):
```
'…"on":'      -> ['f', 't']
'…"on":tr'    -> ['u']      # only "true" fits
'…"on":true'  -> ['}']      # separator (would be ',' if more parameters followed)
'…"on":true}' -> ['}']      # close the outer object
```

**String escape handling** (`grammar.py:157-192`): a `"` closes the string only if an *even* number of backslashes come right before it. So `sh\"` stays open (an escaped quote) and `sh\\"` closes. While the string is open and ends in an odd number of backslashes, only the valid JSON escape characters `" \ / b f n r t` are allowed, so the model can't write `\d` (invalid JSON) and has to write `\\d`.

**`is_complete()`** (`grammar.py:265-283`): the call is finished when no character is allowed *and* the text parses as JSON. A proper prefix of a JSON object never parses, so a dead end and a finished call can't be confused.

**`can_walk_token(token)`**: simulates appending a multi-character token one character at a time and then **restores** the state. Example: at `'{"name":"fn_'`, the token `greet` passes (g → r → e → e → t are each allowed in turn), and `gx` fails at `x`.

**`advance(token)`**: permanently appends the token's text.

### 5.3 `src/tokenizer.py`: `CustomTokenizer`

1. Loads Qwen's `vocab.json` (`{"token_string": id}`, 151 643 entries) and **inverts** it to `{id: "token_string"}`.
2. `decode_single_token(id)`: converts BPE byte-level markers.
   - `Ġ` → space (example: `"Ġhello"` → `" hello"`)
   - `Ċ` → newline (example: `"ĊĊ"` → `"\n\n"`, which is the `\n\n` at the end of the real log)
3. `get_allowed_token_ids(rulebook)`: tests **every** vocab token with `can_walk_token` and returns the ids that pass.
4. **Cache**: the result is memoized by `text_so_far`. The same state gives the same allowed set, so computing it again is pointless.

**Cache example from the real run:**
```
prompt 1: Cache Hits: 0  | Misses: 19 | Hit Rate: 0.00%
prompt 2: Cache Hits: 15 | Misses: 27 | Hit Rate: 35.71%
...
prompt 11: Cache Hits: 119 | Misses: 137 | Hit Rate: 46.48%
```
The counters are cumulative. Prompt 2 is also `fn_add_numbers`, so it gets 15 hits: the shared prefix states (`' {"'`, `' {"name'`, … `' {"name":"fn'`, which *every* prompt reuses) plus the key states up to `"a":`. Only its number values (`265`, `345`) are new states. Over the whole file the hit rate ends at about 46%. The cache lives on the tokenizer, which is created once in `main`, so it persists across prompts.

Why is it a Pydantic `BaseModel`? It validates `vocab_json_path` (it must not be empty) using `@field_validator`, and `model_post_init` runs the loading logic after validation.

### 5.4 `src/__main__.py`: orchestration

Key points:
- `argparse` with 3 optional flags and defaults (`__main__.py:35-49`).
- File-existence check that lists **all** missing files before exiting (`76-88`).
- `os.makedirs(output_dir, exist_ok=True)` creates `data/output/` when it's missing.
- `json.JSONDecodeError` is caught, with a clean message and `exit(1)`.
- Shape check (`104-113`): both files must be a list of objects, otherwise a clean error and `exit(1)`, before the model is loaded.
- A **new `TrieJSONRulebook` for each prompt** (fresh state), but a **single model and tokenizer** (expensive, so loaded once).
- Generation loop of at most **300 steps**, which guarantees termination even if the model never closes the JSON. It stops when `rulebook.is_complete()` is true.
- **Masking** (`147-165`): see §6.3 for a numeric example. If the grammar allows no token at all (a dead end), generation **stops** instead of continuing unconstrained.
- **Post-processing** (`194-235`): slice from the first `{` to the last `}`, `json.loads`, then cast each parameter to its schema type (`boolean`→`bool`, `integer`→`int`, `number`→`float`, else `str`). That's why `"a": 2` in the raw text becomes `2.0` in the output.
- `try/except Exception` for each prompt: one failed prompt doesn't kill the run. It prints `❌ Failed…` and moves on.

### 5.5 `src/validator.py`: `FunctionCallResult`

```python
class FunctionCallResult(BaseModel):
    prompt: str
    name: str
    parameters: Dict[str, Any]
```

**Example 1, valid:**
```python
FunctionCallResult(prompt="Greet john", name="fn_greet", parameters={"name": "john"})
```
**Example 2, invalid** (raises `ValidationError`, caught in main as `❌ Failed`):
```python
FunctionCallResult(prompt="x", name=None, parameters={})
# name: Input should be a valid string
```
`.model_dump()` turns it into a plain dict for `json.dump`.

---

## 6. Core concepts you must be able to explain

### 6.1 Token and BPE
A token is a sub-word piece. BPE (Byte-Pair Encoding) starts from bytes and repeatedly merges the most frequent adjacent pair into a new token.
- Example: `"fn_add_numbers"` is several tokens, not one. In the real log you can see `fn`, then `_add`, then `_numbers`.
- That's why the grammar is character-based: one token can cover several grammar positions at once (`can_walk_token`).

### 6.2 Logits vs probabilities
The model outputs one **logit** (raw score) per vocab entry. `softmax` would turn them into probabilities, but since we only take `argmax`, softmax isn't needed: it preserves order.
- Example: logits `[2.0, 1.0]` → softmax `[0.73, 0.27]` → the argmax is index 0 either way.

### 6.3 Masking: a numeric example
```
vocab:       0:'{'   1:'hello'   2:'"'   3:' {'
logits:      [ 2.1,    5.3,       -0.4,    3.8 ]
allowed:     {0, 3}
mask:        [ 0,     -inf,      -inf,     0   ]
logits+mask: [ 2.1,   -inf,      -inf,     3.8 ]
argmax  →  3  (' {')
```
Without the mask the model would have picked `'hello'` (5.3). Why `-inf` instead of a big negative number? `x + -inf = -inf` for any finite `x`, so an illegal token can **never** win. With e.g. `-1000`, a sufficiently large logit could in theory still win.

### 6.4 Greedy decoding
`np.argmax` always takes the single best allowed token. There is no sampling and no temperature, so the result is **deterministic**: same input, same output. That's good for function calling, where creativity isn't wanted.

### 6.5 Why not just prompt "answer in JSON"?
A 0.6B model will often add text ("Sure! Here's the JSON: …"), make up function names, forget quotes, or not stop. A prompt is a *request*. A mask is a *guarantee*. Here the prompt still helps: it shows the model the schemas so it picks the *right* function and arguments. The mask keeps it *legal*.

### 6.6 Autoregressive generation
Each chosen token is appended to `input_ids`, and the whole sequence is fed back in to get the next logits (`__main__.py:171`). The model "sees" its own output.

### 6.7 Two kinds of correctness
- **Syntactic**: valid JSON (braces, quotes, commas).
- **Semantic / schema**: the function exists, the parameter names match, the types match.
The grammar handles both. Pydantic plus type casting is a second line of defence.

---

## 7. Live tests the evaluator might run

| Test | Command | Actual behaviour |
|---|---|---|
| Default run | `make run` | writes `data/output/function_calling_results.json` |
| Custom paths | `uv run python -m src --input X --output Y --functions_definition Z` | uses those paths |
| Missing files | `uv run python -m src --input nope.json --functions_definition nope2.json` | prints `Error: File not found:` then `- nope2.json` and `- nope.json`, exit 1 ✅ (verified) |
| Malformed JSON | input file `[{"prompt": "x"` | prints `Error: One of the input JSON configuration files is malformed!` and `Details: Expecting ',' delimiter: line 2 column 1 (char 16)`, exit 1 ✅ (verified) |
| Output dir missing | `--output new_dir/out.json` | the directory is created automatically |
| Empty prompt list | input `[]` | writes `[]` (after loading the model) |
| Prompts file is an object, not a list | input `{"a": 1}` | prints `Error: The input prompts must be a list of objects!`, exit 1 ✅ (verified) |
| Prompt with no `"prompt"` key | `[{}]` | treated as an empty prompt `""` |
| New function added | add e.g. `fn_multiply` with `a`, `b` numbers | the grammar picks it up automatically, nothing is hardcoded |
| Function with no parameters | `{"name": "fn_ping", "parameters": {}}` | grammar forces `{"name":"fn_ping","parameters":{}}` ✅ (grammar verified) |
| Integer type | `{"type": "integer"}` | a sign then digits only (no `.` or `e`), cast with `int()` |
| Boolean param | `{"type": "boolean"}` | only `true` or `false`, kept as a JSON boolean in the output ✅ (grammar verified) |
| Help | `uv run python -m src --help` | shows the argparse usage and description |

**Prepare a custom test file in advance** so you can show you understand your own code:
```json
[
  {"name": "fn_multiply", "description": "Multiply two numbers.",
   "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
   "returns": {"type": "number"}}
]
```
with prompts `[{"prompt": "What is 6 times 7?"}]`.

---

## 8. Likely evaluator questions and answers

**Q: What does the project do, in one sentence?**
A: It converts natural-language requests into schema-valid JSON function calls by masking a local LLM's logits with a grammar at every generation step.

**Q: Which model, and why so small?**
A: `Qwen/Qwen3-0.6B`, which the subject's `llm_sdk` provides. Being small makes it fast locally, and it shows the point: constraints make even a weak model produce valid structure.

**Q: Where exactly does the constraint happen?**
A: `__main__.py:147-165`. Get the allowed ids, get the logits, build a mask full of `-inf`, set allowed ids to `0`, add, `argmax`.

**Q: How do you know which tokens are allowed?**
A: `CustomTokenizer.get_allowed_token_ids` walks every vocab token through `rulebook.can_walk_token`, which simulates appending it one character at a time against `get_allowed_characters()`.

**Q: Isn't checking 151k tokens every step slow?**
A: Yes. A cache miss costs roughly (vocab size × token length × grammar check). That's why results are memoized by grammar state. The fixed prefix states are shared by every prompt, so they hit the cache from prompt 2 on. Possible improvements: a precomputed token trie, or caching by grammar *position* instead of the full text.

**Q: What happens if the model never closes the JSON?**
A: The loop is capped at 300 steps. Parsing then fails, and the `except` prints `❌ Failed to parse for query: …` and continues with the next prompt.

**Q: How does generation stop?**
A: When `rulebook.is_complete()` is true: the grammar allows no further character and the text parses as JSON. It used to stop when the text ended with `}}`, which cut off a string value like `"a}}b"` early. If the grammar ever reaches a dead end (no token allowed but not complete), generation also stops, instead of turning the mask off.

**Q: What if the model wants to write a space, e.g. `"a": 2`?**
A: It can't. The grammar only allows the compact form (`"a":2`), and because the mask is never turned off, tokens with a leading space are masked. Only whitespace *before* the opening `{` is tolerated.

**Q: Why does `2` become `2.0`?**
A: The post-processing step casts by schema type: `number` → `float()`. The schema says `number`, which in JSON Schema terms includes non-integers, so a float is the safe representation.

**Q: What if a user asks for a function that doesn't exist, e.g. "Book me a flight"?**
A: The grammar only allows the 5 defined names, so the model *must* pick one of them, probably whichever is closest. There is no "none" option. That's a limitation (see §9).

**Q: Why Pydantic if the grammar already constrains output?**
A: Defence in depth. It guarantees the final record has `prompt: str`, `name: str`, and `parameters: dict` before it's written, and catches anything the grammar or casting missed.

**Q: What are `Ġ` and `Ċ`?**
A: GPT-2-style byte-level BPE maps bytes to printable Unicode characters. `Ġ` represents a space and `Ċ` a newline. `vocab.json` stores tokens in that form, so they must be translated back before comparing with the grammar.

**Q: Why does each prompt get a new `TrieJSONRulebook` but share the tokenizer?**
A: The rulebook holds per-prompt state (`text_so_far`). The tokenizer is stateless apart from its cache, and the cache stays valid across prompts because the schema is the same.

**Q: Why `-inf` and not removing tokens from the list?**
A: It keeps the logits array aligned with the token ids (index = id), so `argmax` gives a token id directly. It's also a single vectorized numpy add.

**Q: What does `if token_id < len(custom_mask)` protect against?**
A: The logits size (model embedding size, 151 936) is larger than `vocab.json` (151 643). Special tokens such as `<|im_end|>` aren't in `vocab.json`. The check guards against an index error if the sizes ever disagree.

**Q: What is the difference between `make lint` and `make lint-strict`?**
A: `lint` runs flake8 plus mypy with specific flags. `lint-strict` runs flake8 plus `mypy --strict` (every strictness flag). Both pass with no issues.

**Q: How did you use AI?**
A: Answer from the README's *AI usage* section: to understand the requirements, compare strategies, and reason about constrained decoding and masking. You reviewed and adapted the code yourself. Be ready to explain any line.

**Q: How would you support nested objects or arrays?**
A: Add grammar branches for `[`…`]` and nested `{`…`}` with recursion (or an explicit stack) in `get_allowed_characters`. Right now any unknown type falls into the number branch (`grammar.py:211`).

---

## 9. Known weaknesses and how to fix or defend them

### 9.1 Fixed (2026-10-22)

These were real bugs. Each one was reproduced with the grammar alone before the fix and checked again after it. If the evaluator asks "did you find any bugs?", this list is a good answer.

| # | Bug | Was | Fix | Now |
|---|---|---|---|---|
| 1 | First character of each parameter key was never allowed, so the mask turned off | `return [expected_key]` | `return [expected_key[0]]` (`grammar.py:147`) | `'…"parameters":{'` → `['"']` |
| 2 | Numbers consumed the `,`/`}` separator twice | `'"a":2.0,'` → `[',']` | number scan stops *on* the separator; the shared code consumes it once | `'"a":2.0,'` → `['"']` |
| 3 | Booleans followed by a separator dead-ended | `'"on":true,'` → `[]` | match `true`/`false` as a prefix of the remainder, not the whole remainder (`grammar.py:194-210`) | `'"on":true,'` → `['"']` |
| 4 | When nothing was allowed, the mask was disabled (`fill(0.0)`), which hid bugs 1–3 | free-form generation | stop generating instead (`__main__.py:147-151`) | mask is never turned off |
| 5 | Number characters weren't ordered: `1-2-`, `1.2.`, `007` were accepted | any of `0-9.-eE+` in any order | JSON number state machine (`grammar.py:8-51`) | `'"a":1-'` → `[]` |
| 6 | A string value containing `}}` ended generation early | stop on text suffix `}}` | stop on `rulebook.is_complete()` (`grammar.py:265-283`) | `'"name":"a}}'` keeps the string open |
| 7 | A function with no parameters dead-ended at `"parameters":{}` | `[]` | consume the inner `}` when the parameter list is empty (`grammar.py:135-141`) | `'…{}'` → `['}']` |
| 8 | Any character was allowed after `\`, so `\d` gave invalid JSON | 95 chars | only `" \ / b f n r t` after an odd run of backslashes (`grammar.py:183-188`) | `'"shr\'` → 8 chars |
| 9 | Booleans were written to the output as the string `"True"` | `str(v)` | `bool(v)` for `boolean` parameters (`__main__.py:212-213`) | `true` in the output JSON |
| 10 | A prompts file that is an object instead of a list crashed | `AttributeError` on `item.get` | `isinstance` check on both input files (`__main__.py:104-113`) | clean error, exit 1 |

The old whitespace problem (the free-running model picked `' "'` and every later state returned `[]`) is gone as a side effect of 1 and 4: the mask stays on, so a token with a leading space can't be picked.

### 9.2 Still open (design limits you can defend)

| # | Issue | Where | Fix if asked |
|---|---|---|---|
| 1 | Strings are ASCII 32–126 only, so `é` and emoji are blocked | `grammar.py:189-192` | `vocab.json` stores bytes as GPT-2 byte-level characters (e.g. `é` is `Ã©`), and `decode_single_token` only translates `Ġ` and `Ċ`. Allowing non-ASCII would need a full byte-level decode first, otherwise mojibake ends up in the output. Raw control characters such as a newline are invalid inside a JSON string anyway, so blocking them is correct. |
| 2 | Always forces a function call, with no "none" option | design | Add a sentinel function, or let the grammar allow `null` |
| 3 | Cache keyed on the full text, so most parameter states miss | `tokenizer.py:110` | Key on the grammar position (e.g. "inside string") instead |
| 4 | Only the compact JSON form is allowed (no spaces after `:` or `,`) | grammar | Not a bug, since the mask enforces it. Allowing optional whitespace would be a small grammar change but adds more states to scan. |
| 5 | Only flat objects: no arrays or nested objects | grammar | See the last question in §8 |

**How to talk about the fixes:** "The first version only constrained the function name. The key-start state returned the whole key instead of its first character, so `can_walk_token` rejected every token, and my fallback then disabled the mask. That hid the bug because the model usually writes good JSON anyway. I fixed the grammar, and now a dead end stops generation instead of turning the mask off, so a grammar bug can't hide again." Knowing exactly where and why scores well.

---

## 10. Command cheat sheet

```bash
make install        # uv sync + flake8/mypy/pydantic/numpy
make run            # default paths
make debug          # run under pdb (n = next, s = step, c = continue, p var = print)
make lint           # flake8 + mypy (selected flags)
make lint-strict    # flake8 + mypy --strict
make clean          # remove caches and data/output

uv run python -m src --help
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json

# Test the grammar on its own (no model, instant):
uv run python -c "
from src.parser import parse_function_schemas; from src.grammar import TrieJSONRulebook; import json
r = TrieJSONRulebook(parse_function_schemas(json.load(open('data/input/functions_definition.json'))))
r.text_so_far = '{\"name\":\"fn_'; print(r.get_allowed_characters())"
# -> ['r', 's', 'g', 'a']   (order may vary, it comes from a set)
```

### File map

| File | Role | Key lines |
|---|---|---|
| `src/__main__.py` | CLI, loop, masking, post-processing | 104-113 input shape, 143-192 loop, 147-165 mask, 194-235 parse |
| `src/parser.py` | schema → `{fn: [(param, type)]}` | 30-41 |
| `src/grammar.py` | state machine / allowed characters | 8-51 number states, 97-101 prefix, 103-117 name trie, 135-141 no params, 157-255 values, 265-283 `is_complete` |
| `src/tokenizer.py` | vocab inversion, token filtering, cache | 70-73, 110-130 |
| `src/validator.py` | Pydantic output model | 23-25 |
| `llm_sdk/` | provided wrapper: `encode`, `get_logits_from_input_ids`, `get_path_to_vocab_file` | not your code |
