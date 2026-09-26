"""The OpenRouter door and the ride-out under the paper's `gpt_request` (003-IP DS-D, DS-E).

D2: one OpenAI client pointed at OpenRouter serves chat and embeddings. Every chat request
names the provider pin with `allow_fallbacks: false`, so OpenRouter never answers from a
provider the run did not declare; the paper's model names get OpenRouter's `openai/`
prefix. Temperature, `max_tokens`, the prompt and the parser are the caller's, unchanged.

D3: `ride_out` sends one request and waits out whatever a later attempt can cure: any
status but 400–403, a timeout, a dropped connection, a 200 with no choices. Backoff is
deterministic (2, 4, 8… s, capped at 120 s); a provider's `Retry-After` replaces the
backoff, under the same cap; 40 attempts. Each wait is a `rate_limit` event, the answer a
`call` event, and a request that is given up on an `error` event followed by `Failed`,
which the paper's `gpt_request` lets through (its `except` would otherwise turn it into a
"GENERATION ERROR" string that the parser reads as an empty answer). The client's own
retries are off, so every wait the run makes is one this module recorded.

Events go to the sinks in `SINKS`; 003-IP step 3 plugs the run's trace in there. With no
sink an event is dropped.
"""
import time

import openai

from simulation_engine import settings

BASE_URL = "https://openrouter.ai/api/v1"
HEADERS = {
  "HTTP-Referer": "https://github.com/ericvon07/genagents",
  "X-OpenRouter-Title": "System One Lab - Predicting People",
}
EMBEDDING_MODEL = "openai/text-embedding-3-small"

ATTEMPTS = 40
FIRST_WAIT_S = 2
MAX_WAIT_S = 120
NOT_RETRIED = (400, 401, 402, 403)

# The provider that serves chat. Step 5's launch query (DS-I) sets it; `openai` until then.
pin = "openai"
# DS-D's fallback: True when the health gate found OpenRouter's embeddings differ from
# the stored ones; embeddings then go straight to OpenAI and the manifest says so.
embeddings_direct = False

SINKS = []
sleep = time.sleep
monotonic = time.monotonic


class Failed(Exception):
  """A request the ride-out gave up on: not retryable, or its budget spent."""

  def __init__(self, reason, cause):
    super().__init__(f"{reason}: {cause}")
    self.reason = reason
    self.cause = cause


class EmptyAnswer(Exception):
  """A 200 whose body carries no choices (OpenRouter reports upstream errors this way)."""


def emit(event, **fields):
  record = {"event": event, "ts": time.time(), **fields}
  for sink in SINKS:
    sink(record)
  return record


def openrouter_model(model):
  return model if "/" in model else f"openai/{model}"


def make_client():
  return openai.OpenAI(base_url=BASE_URL, api_key=settings.OPENROUTER_API_KEY,
                       default_headers=HEADERS, max_retries=0)


def make_direct_client():
  return openai.OpenAI(api_key=settings.OPENAI_API_KEY, max_retries=0)


def _status(exc):
  return getattr(exc, "status_code", None)


def _retry_after(exc):
  response = getattr(exc, "response", None)
  value = response.headers.get("retry-after") if response is not None else None
  try:
    return float(value)
  except (TypeError, ValueError):
    return None


def _retryable(exc):
  if isinstance(exc, (EmptyAnswer, openai.APIConnectionError)):
    return True  # APITimeoutError is an APIConnectionError
  if isinstance(exc, openai.APIStatusError):
    return exc.status_code not in NOT_RETRIED
  return False


def ride_out(send, kind, model, provider):
  """Call `send()` until it answers, waiting out what a later attempt can cure."""
  for attempt in range(1, ATTEMPTS + 1):
    started = monotonic()
    try:
      response = send()
      if kind == "chat" and not getattr(response, "choices", None):
        raise EmptyAnswer("200 with no choices")
    except Exception as exc:
      last = attempt == ATTEMPTS
      if not _retryable(exc) or last:
        reason = "budget_spent" if _retryable(exc) else "not_retryable"
        emit("error", kind=kind, model=model, provider=provider, attempt=attempt,
             status=_status(exc), reason=reason, error=f"{type(exc).__name__}: {exc}")
        raise Failed(reason, exc) from exc
      backoff = FIRST_WAIT_S * 2 ** (attempt - 1)
      wait = min(_retry_after(exc) or backoff, MAX_WAIT_S)
      emit("rate_limit", kind=kind, model=model, provider=provider, attempt=attempt,
           status=_status(exc), wait_s=wait, error=f"{type(exc).__name__}: {exc}")
      sleep(wait)
      continue
    usage = getattr(response, "usage", None)
    emit("call", kind=kind, model=getattr(response, "model", None) or model,
         provider=getattr(response, "provider", None) or provider, attempt=attempt,
         latency_ms=round((monotonic() - started) * 1000),
         tokens_in=getattr(usage, "prompt_tokens", None),
         tokens_out=getattr(usage, "completion_tokens", None))
    return response


def chat(prompt, model, max_tokens, temperature):
  """D2 + D3: the paper's chat request, through OpenRouter, pinned, ridden out."""
  client = make_client()
  model = openrouter_model(model)
  provider = pin
  return ride_out(lambda: client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=max_tokens,
    temperature=temperature,
    extra_body={"provider": {"order": [provider], "allow_fallbacks": False}},
  ), "chat", model, provider)


def embed(text, model):
  """D2 + D3: the paper's embedding request, through OpenRouter unless DS-D's fallback is on."""
  if embeddings_direct:
    client, model, provider = make_direct_client(), model.split("/")[-1], "openai-direct"
  else:
    client, model, provider = make_client(), openrouter_model(model), "openrouter"
  response = ride_out(lambda: client.embeddings.create(input=[text], model=model),
                      "embedding", model, provider)
  return response.data[0].embedding


def cosine(a, b):
  dot = sum(x * y for x, y in zip(a, b))
  na = sum(x * x for x in a) ** 0.5
  nb = sum(y * y for y in b) ** 0.5
  return dot / (na * nb)


def embedding_identity(text, stored, embed_fn=None, threshold=0.999):
  """DS-D's check: does `text` re-embed to the vector the agent bank stored for it?

  Returns (cosine, passed). The health gate (DS-J) runs it on a node of the example
  agent; a failure turns `embeddings_direct` on for the run and the manifest says so.
  """
  from simulation_engine.gpt_structure import get_text_embedding
  vector = (embed_fn or get_text_embedding)(text)
  c = cosine(vector, stored)
  return c, c >= threshold
