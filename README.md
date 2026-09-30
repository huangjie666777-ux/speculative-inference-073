# Speculative Inference (SmolLM2 360M / 135M, CPU)

CPU-only streaming web application that compares plain autoregressive
continuation with a **hand-written speculative decoding loop**:

* target model: SmolLM2-360M (`models/target`)
* draft model: SmolLM2-135M (`models/draft`)
* one shared tokenizer (the two models ship identical tokenizer files)
* real PyTorch CPU inference; no GPU and no cloud credentials
* the speculative accept/reject core is implemented explicitly
  (`src/specserve/decoding.py`); it does **not** call a built-in assisted
  generation API

No speedup is promised: the page reports raw counters and wall time only.

## Layout

| Path | Responsibility |
| --- | --- |
| `src/specserve/models.py` | lazy, process-wide CPU loading of both models and the shared tokenizer |
| `src/specserve/sampling.py` | temperature-normalized probabilities, seeded RNG, `min(1, p/q)` acceptance and the normalized `max(0, p-q)` residual distribution |
| `src/specserve/decoding.py` | per-request `DynamicCache` KV states, autoregressive baseline and the speculative proposal/verification loop with cache cropping |
| `src/specserve/service.py` | request validation, the single active-request slot and generation lifecycle |
| `src/specserve/app.py` | FastAPI SSE streaming endpoint, busy/validation responses, disconnect cleanup |
| `web/index.html`, `web/app.js`, `web/style.css` | page, controls and stop button (cross-file front end) |
| `tests/` | probability, decoding, cache and HTTP lifecycle tests using tiny Llama models |

## Algorithm

Each speculative round:

1. The draft proposes `draft_steps` (gamma) candidates, one autoregressive
   step per proposal, reusing its own KV cache.
2. The target verifies the carried token plus **all** candidates in a single
   forward (the first round uses prompt-prefill logits for candidate zero).
3. Acceptance:
   * temperature = 0: accept the longest prefix matching target greedy
     tokens; at the first divergence the target token replaces the proposal;
   * temperature > 0: accept token `x` with probability `min(1, p(x)/q(x))`
     using the same temperature for both models; on rejection the correction
     token is drawn from the normalized positive part of `p - q`;
   * when every candidate is accepted, an extra target "bonus" token is
     emitted so the output marginal stays the target distribution.
4. Both caches are cropped to the confirmed prefix length (positions and
   attention masks are rebuilt from the cache length), so prompt history is
   never recomputed. Unconfirmed tokens never leave the server.
5. EOS and the length cap stop emission immediately. Each request gets fresh
   caches and a seeded `torch.Generator`; the same mode and seed reproduce the
   same stream. Caches are never shared across requests.

Only one generation runs at a time. Invalid parameters are rejected before
the slot is acquired. On client disconnect/stop the cancellation flag is
observed at the next token/round boundary after the forward currently running
finishes, then per-request state is released.

## Install

```sh
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
```

Models are already present under `models/`. A fresh checkout can restore the
pinned files with `.venv/bin/python scripts/fetch_models.py`.

## Run

```sh
PYTHONPATH=src .venv/bin/python -m uvicorn specserve.app:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>, type an English prefix, pick the mode and
parameters, and use **Stop** to cancel. The page shows candidate count,
accepted count, target forward count, elapsed seconds and the final status.

### Streaming HTTP API

`POST /api/generate` with JSON body:

| Field | Range | Meaning |
| --- | --- | --- |
| `text` | non-empty string | English prompt prefix |
| `mode` | `autoregressive` / `speculative` | decoding mode |
| `max_new_tokens` | 1–512 | generation length cap |
| `draft_steps` | 1–16 | gamma, ignored by autoregressive mode |
| `temperature` | 0.0–2.0 | 0 selects greedy decoding |
| `seed` | 0–2147483647 | CPU RNG seed |

Response is `text/event-stream`:

* `event: token` with `{"text": ..., "token_id": ...}` per confirmed token;
* `event: done` with `candidates`, `accepted`, `target_forwards`,
  `elapsed_seconds` and `stopped`;
* `event: error` on inference errors.

Status codes: `409` while another request is active, `422` for invalid
parameters, `400` for unusable prompts.

Example:

```sh
curl -N -X POST http://127.0.0.1:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The best way to learn programming is","mode":"speculative",
       "max_new_tokens":20,"draft_steps":4,"temperature":0,"seed":1234}'
```

Abort the request (close the connection, e.g. `--max-time 3`) to exercise the
stop path; `/health` reports the current `busy` flag.

## Tests

```sh
.venv/bin/python -m pytest
```

Tests build tiny random Llama models (no multi-hundred-MB downloads) and
cover: temperature/residual probability math, seeded reproducibility,
greedy equivalence between speculative and autoregressive output at every
gamma, KV-cache cropping correctness, EOS/length behavior, cancellation after
the current forward, and the 409/422/disconnect HTTP lifecycle.
