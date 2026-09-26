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

DS-I, the provider policy: `choose` asks OpenRouter which endpoints serve the model and
ranks the declared candidates (`openai`, `azure`, the two that serve `gpt-4o-mini` with the
same weights) by `uptime_last_30m`; the best is the pin unless one is forced, the rest are
the fallbacks, and every endpoint's snapshot goes into `run.json`. `policy` then names the
provider of each chat request. It moves to the next declared provider when a request spent
its whole ride-out budget on the current one (`fallback`), or when answered calls over the
last five minutes fall under `min_calls_per_minute` (`throughput`); fifteen minutes after a
move away from the pin, the next request tries the pin again (`retry_pin`). Every move is a
`provider_switch` event, emitted before the request that names the new provider.

Events go through `lab.trace.emit`, to whatever sinks the run's trace opened.
"""
import collections
import json
import threading
import time
import urllib.request

import openai

from lab import trace
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

DECLARED = ("openai", "azure")  # DS-I: the providers of openai/gpt-4o-mini, same weights
ENDPOINTS_URL = BASE_URL + "/models/%s/endpoints"
WINDOW_S = 300           # the throughput floor looks at the last five minutes
RETRY_PIN_AFTER_S = 900  # fifteen minutes after a move, the pin is tried again
MATERIAL_GAP = 1.0       # uptime points another provider needs to take the pin from the first declared

# DS-D's fallback: True when the health gate found OpenRouter's embeddings differ from
# the stored ones; embeddings then go straight to OpenAI and the manifest says so.
embeddings_direct = False

sleep = time.sleep
monotonic = time.monotonic


class Failed(Exception):
  """A request the ride-out gave up on: not retryable, or its budget spent."""

  def __init__(self, reason, cause):
    super().__init__(f"{reason}: {cause}")
    self.reason = reason
    self.cause = cause


class NoProvider(Exception):
  """None of the declared providers serves the model now."""


class Policy:
  """Which provider the next chat request names (DS-I). Thread-safe: the runner's agents
  share one policy."""

  def __init__(self, pin="openai", fallbacks=(), min_calls_per_minute=None):
    self.pin, self.fallbacks = pin, list(fallbacks)
    self.floor = min_calls_per_minute
    self.current, self.since = pin, None
    self.answers = collections.deque()
    self._lock = threading.Lock()

  @property
  def declared(self):
    return [self.pin] + self.fallbacks

  def _next(self):
    declared = self.declared
    if len(declared) < 2:
      return None
    return declared[(declared.index(self.current) + 1) % len(declared)]

  def _move(self, to, reason, now, **extra):
    trace.emit("provider_switch", kind="chat", reason=reason, from_provider=self.current,
               to_provider=to, **extra)
    self.current, self.since = to, now
    self.answers.clear()

  def _rate(self, now):
    while self.answers and now - self.answers[0] > WINDOW_S:
      self.answers.popleft()
    return len(self.answers) / (WINDOW_S / 60)

  def provider(self):
    """The provider the next request names, after any move the clock or the floor calls for."""
    with self._lock:
      now = monotonic()
      if self.since is None:
        self.since = now
      if self.current != self.pin and now - self.since >= RETRY_PIN_AFTER_S:
        self._move(self.pin, "retry_pin", now)
      elif self.floor is not None and self._next() and now - self.since >= WINDOW_S:
        rate = self._rate(now)
        if rate < self.floor:
          self._move(self._next(), "throughput", now, calls_per_minute=round(rate, 2),
                     min_calls_per_minute=self.floor)
      return self.current

  def answered(self, provider):
    with self._lock:
      if provider == self.current:
        self.answers.append(monotonic())

  def failed(self, provider):
    """A request spent its ride-out budget on `provider`: the next one names the next provider."""
    with self._lock:
      if provider == self.current and self._next():
        self._move(self._next(), "fallback", monotonic())


# The run's policy; the runner replaces it at launch. Until then: the pin `openai`, no fallback.
policy = Policy()


def fetch_endpoints(model):
  request = urllib.request.Request(ENDPOINTS_URL % model, headers={
    "Authorization": "Bearer %s" % settings.OPENROUTER_API_KEY, **HEADERS})
  with urllib.request.urlopen(request, timeout=30) as response:
    return json.load(response)


def rank(body, declared=DECLARED):
  """The declared providers that serve the model, and a snapshot of every endpoint listed.
  The first declared provider leads unless another beats its `uptime_last_30m` by
  `MATERIAL_GAP` points or more (the author's rule at the dry run: a gap of hundredths is
  noise, and the first declared is the faster); the rest follow by uptime, ties in declared
  order, an unknown uptime last."""
  snapshot = [{"tag": e.get("tag"), "provider_name": e.get("provider_name"),
               "status": e.get("status"), "uptime_last_5m": e.get("uptime_last_5m"),
               "uptime_last_30m": e.get("uptime_last_30m"),
               "uptime_last_1d": e.get("uptime_last_1d"),
               "latency_p50_ms_30m": (e.get("latency_last_30m") or {}).get("p50")}
              for e in body["data"]["endpoints"]]
  served = {e["tag"]: e["uptime_last_30m"] for e in snapshot if e["tag"] in declared}
  uptime = lambda t: served[t] if served[t] is not None else -1
  order = sorted((t for t in declared if t in served), key=lambda t: -uptime(t))
  first = next((t for t in declared if t in served), None)
  if order and uptime(order[0]) - uptime(first) < MATERIAL_GAP:
    order = [first] + [t for t in order if t != first]
  return order, snapshot


def choose(model, force_pin=None, declared=DECLARED, fetch=None):
  """DS-I at launch: the pin, the fallbacks and the snapshot `run.json` records."""
  model = openrouter_model(model)
  order, snapshot = rank((fetch or fetch_endpoints)(model), declared)
  if force_pin:
    if force_pin not in declared:
      raise NoProvider("--pin %r is not declared (%s)" % (force_pin, ", ".join(declared)))
    order = [force_pin] + [t for t in order if t != force_pin]
  if not order:
    raise NoProvider("no declared provider (%s) serves %s" % (", ".join(declared), model))
  return {"model": model, "declared": list(declared), "pin": order[0],
          "fallbacks": order[1:], "forced": bool(force_pin),
          "queried_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "endpoints": snapshot}


class EmptyAnswer(Exception):
  """A 200 whose body carries no choices (OpenRouter reports upstream errors this way)."""


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
        trace.emit("error", kind=kind, model=model, provider=provider, attempt=attempt,
                   status=_status(exc), reason=reason, error=f"{type(exc).__name__}: {exc}")
        raise Failed(reason, exc) from exc
      backoff = FIRST_WAIT_S * 2 ** (attempt - 1)
      wait = min(_retry_after(exc) or backoff, MAX_WAIT_S)
      trace.emit("rate_limit", kind=kind, model=model, provider=provider, attempt=attempt,
                 status=_status(exc), wait_s=wait, error=f"{type(exc).__name__}: {exc}")
      sleep(wait)
      continue
    usage = getattr(response, "usage", None)
    trace.emit("call", kind=kind, model=getattr(response, "model", None) or model,
               provider=getattr(response, "provider", None) or provider, attempt=attempt,
               latency_ms=round((monotonic() - started) * 1000),
               tokens_in=getattr(usage, "prompt_tokens", None),
               tokens_out=getattr(usage, "completion_tokens", None))
    return response


def chat(prompt, model, max_tokens, temperature):
  """D2 + D3: the paper's chat request, through OpenRouter, pinned, ridden out; DS-I picks the
  provider it names."""
  client = make_client()
  model = openrouter_model(model)
  provider = policy.provider()
  try:
    response = ride_out(lambda: client.chat.completions.create(
      model=model,
      messages=[{"role": "user", "content": prompt}],
      max_tokens=max_tokens,
      temperature=temperature,
      extra_body={"provider": {"order": [provider], "allow_fallbacks": False}},
    ), "chat", model, provider)
  except Failed as exc:
    if exc.reason == "budget_spent":
      policy.failed(provider)
    raise
  policy.answered(provider)
  return response


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
