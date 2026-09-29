"""The typed backend: Jev answers the forced choice over the paper's `agent_desc` (003-IP DS-G).

One `system_one` request per batch. The state is the very `agent_desc` string the
chain-of-thought prompt is built from; each item of the batch is one question, keyed by its
position (the key is for code and is not sent to the model), whose instructions are the
paper's task sentence and the item's text:

- a categorical item is a `Choice` whose criteria are the item's options, verbatim and in
  the bank's order, undescribed; the pick is the answer's `choice`, the distribution its
  probabilities put back in the bank's order (the API returns them in its own);
- an integer range with at most 255 values is a `Choice` over its integers; the pick is the
  chosen integer;
- a float range is a `Score` over ten evenly spaced levels, each a sentence naming its
  value; the pick is the expected value mapped back onto the range, the distribution the
  probability of each level (JE-K: numeric items are the typed model's weak ground).

Each item comes back as a dict the seam writes into its `decision` event: `predicted`,
`primitive`, `distribution`, `confidence` and `model`, the version the response reported,
never the `jev-latest` alias it was asked for. The request itself is one `call` event
(`kind: typed`) with its latency and tokens, which are per batch and so not repeated on
each item.

The client is the SDK's, which identifies itself on every request (GT-2), with a 60 s
timeout for a state of up to ~30k tokens. The SDK's own retries are off; `_send` retries 408, 429, 5xx, timeouts and dropped
connections with deterministic backoff (2, 4, 8… s, `Retry-After` wins, 60 s cap, 8
attempts), each wait a `rate_limit` event, and never retries another 4xx. A request given
up on is an `error` event and `transport.Failed`; a 402 is also a `credits_out` event and
`transport.CreditsOut`, which stops the run (005-IP PL-H). After ten consecutive failed
batches the next one is refused with a `backend_down` event and `BackendDown`, without being
sent; one answered batch resets the count.

`jev-steps` (005-IP PL-F, `variant = "steps"`): each question's `instructions` carries the
paper's reasoning steps for its kind, read verbatim from the chain-of-thought template
(`categorical_resp/batch_v1.txt`, `numerical_resp/batch_v1.txt`), between the task sentence
and the question. The state and the criteria are those of `jev`, byte for byte; `instructions`
is the field TypeSafe's docs give a question (docs.typesafe.ai/primitives/choice, read
2026-09-28).
"""
import os
import re
import threading
import time

from lab import trace
from lab.transport import CreditsOut, Failed
from simulation_engine import settings

MODEL = "jev-latest"
TIMEOUT_S = 60.0
ATTEMPTS = 8
FIRST_WAIT_S = 2
MAX_WAIT_S = 60
RETRIED = {408, 429}  # and every 5xx
MAX_CHOICES = 255
SCORE_LEVELS = 10
DOWN_AFTER = 10
USD_PER_MTOK = (0.042, 0.0)  # list price in, out (002-ST §6, 2026-09-26): output is free
VARIANTS = (None, "steps")
TEMPLATES = os.path.join(settings.LLM_PROMPT_DIR, "generative_agent", "interaction")

# The paper's task sentence (numerical_resp/batch_v1.txt, categorical_resp/batch_v1.txt),
# with "the interview transcript" named as what the state holds.
TASK = ("The state is a participant's self description followed by excerpts of their "
        "interview. Based on it, predict the participant's survey response.")

sleep = time.sleep
monotonic = time.monotonic

_lock = threading.Lock()
_client = None
_consecutive_errors = 0
models_seen = set()  # every version a response reported; the runner writes it into run.json
variant = None       # PL-F: "steps" for jev-steps; the runner sets it after `reset`


class BackendDown(Exception):
  """Ten consecutive batches failed; the run stops instead of sending an eleventh."""


def make_client(transport=None):
  """The SDK's client, its retries off. It names itself (`User-Agent: typesafe-sdk/<v>`) and
  overwrites any User-Agent passed to it, which is what keeps Cloudflare's error 1010 away
  (GT-2)."""
  from typesafe_sdk import RetryPolicy, TypeSafeClient
  return TypeSafeClient(api_key=settings.TYPESAFE_API_KEY, model=MODEL, timeout=TIMEOUT_S,
                        retry=RetryPolicy(max_retries=0), transport=transport)


def client():
  global _client
  with _lock:
    if _client is None:
      _client = make_client()
    return _client


def reset():
  """Forget the client, the error count, the models seen and the variant (a new run)."""
  global _client, _consecutive_errors, variant
  with _lock:
    _client, _consecutive_errors, variant = None, 0, None
    models_seen.clear()


# --- the questions --------------------------------------------------------------

def _levels(low, high):
  step = (high - low) / (SCORE_LEVELS - 1)
  return [round(low + k * step, 6) for k in range(SCORE_LEVELS)]


def shape(kind, options, float_resp):
  """Which primitive an item becomes: `choice` over labels, or `score` over levels."""
  if kind == "categorical":
    return "choice"
  if float_resp:
    return "score"
  low, high = options[0], options[-1]
  if high - low + 1 > MAX_CHOICES:
    raise ValueError(f"an integer range of {high - low + 1} values exceeds a Choice's {MAX_CHOICES}")
  return "choice"


def labels(kind, options, float_resp):
  """The criteria's keys (choice) or the levels' values (score), in order."""
  if kind == "categorical":
    if len(set(options)) != len(options):
      raise ValueError(f"duplicate options {options!r}")
    return list(options)
  low, high = options[0], options[-1]
  if float_resp:
    return _levels(low, high)
  return [str(v) for v in range(int(low), int(high) + 1)]


def steps(kind):
  """The paper's reasoning steps for `kind`, verbatim: from "As you answer" to the last step."""
  with open(os.path.join(TEMPLATES, f"{kind}_resp", "batch_v1.txt"), encoding="utf-8") as f:
    text = f.read()
  found = re.search(r"As you answer.*\n(?:Step \d\).*\n)*Step \d\)[^\n]*", text)
  if not found:
    raise ValueError(f"no reasoning steps in the {kind} template")
  return found.group(0)


def build_question(kind, item, options, float_resp):
  from typesafe_sdk import Choice, Score
  primitive = shape(kind, options, float_resp)
  if variant not in VARIANTS:
    raise ValueError(f"variant {variant!r}; known: {VARIANTS}")
  task = f"{TASK}\n\n{steps(kind)}" if variant == "steps" else TASK
  instructions = f"{task}\n\nQuestion: {item}"
  if kind == "numerical":
    low, high = options[0], options[-1]
    number = "a number" if float_resp else "an integer"
    instructions += f"\nThe answer is {number} from {low:g} to {high:g}."
  if primitive == "choice":
    return Choice(instructions=instructions,
                  criteria={label: None for label in labels(kind, options, float_resp)})
  low, high = options[0], options[-1]
  return Score(instructions=instructions,
               criteria=[f"The participant answers {v:.2f}, on a range from {low:g} to {high:g}."
                         for v in labels(kind, options, float_resp)])


def build_questions(kind, questions, float_resp=False):
  return {f"q{i}": build_question(kind, item, options, float_resp)
          for i, (item, options) in enumerate(questions.items())}


# --- the request ----------------------------------------------------------------

def _retryable(exc):
  from typesafe_sdk import TypeSafeAPIConnectionError, TypeSafeAPIError
  if isinstance(exc, TypeSafeAPIConnectionError):  # a timeout is one
    return True
  if isinstance(exc, TypeSafeAPIError):
    return exc.status in RETRIED or exc.status >= 500
  return False


def _retry_after(exc):
  ms = getattr(exc, "retry_after_ms", None)
  return ms / 1000 if ms is not None else None


def _send(state, questions):
  for attempt in range(1, ATTEMPTS + 1):
    started = monotonic()
    try:
      response = client().system_one(state=state, questions=questions)
    except Exception as exc:
      status = getattr(exc, "status", None)
      if not _retryable(exc) or attempt == ATTEMPTS:
        reason = "budget_spent" if _retryable(exc) else "not_retryable"
        trace.emit("error", kind="typed", model=MODEL, provider="typesafe", attempt=attempt,
                   status=status, reason=reason, error=f"{type(exc).__name__}: {exc}")
        if status == 402:  # PL-H: out of credits
          trace.emit("credits_out", kind="typed", model=MODEL, provider="typesafe", status=402,
                     error=f"{type(exc).__name__}: {exc}")
          raise CreditsOut(reason, exc) from exc
        raise Failed(reason, exc) from exc
      wait = min(_retry_after(exc) or FIRST_WAIT_S * 2 ** (attempt - 1), MAX_WAIT_S)
      trace.emit("rate_limit", kind="typed", model=MODEL, provider="typesafe", attempt=attempt,
                 status=status, wait_s=wait, error=f"{type(exc).__name__}: {exc}")
      sleep(wait)
      continue
    trace.emit("call", kind="typed", model=response.model, provider="typesafe",
               attempt=attempt, latency_ms=round((monotonic() - started) * 1000),
               tokens_in=response.usage.input_tokens, tokens_out=response.usage.output_tokens)
    return response


def _item(kind, options, float_resp, answer, model):
  primitive = shape(kind, options, float_resp)
  if answer is None:  # the response left this question out
    return {"predicted": None, "primitive": primitive, "distribution": None,
            "confidence": None, "model": model}
  if primitive == "choice":
    predicted = answer.choice if kind == "categorical" else int(answer.choice)
    probabilities = answer.probabilities  # the API's order is not the bank's: put it back
    return {"predicted": predicted, "primitive": primitive,
            "distribution": {label: probabilities.get(label)
                             for label in labels(kind, options, float_resp)},
            "confidence": answer.confidence, "model": model}
  levels = labels(kind, options, float_resp)
  low, high = options[0], options[-1]
  step = (high - low) / (SCORE_LEVELS - 1)
  return {"predicted": round(low + answer.score * step, 6), "primitive": primitive,
          "distribution": {f"{v:g}": answer.probabilities.get(k) for k, v in enumerate(levels)},
          "confidence": answer.confidence, "model": model}


def answer(kind, agent_desc, questions, float_resp=False):
  """One typed request for the batch; one result per item, in the batch's order."""
  global _consecutive_errors
  with _lock:
    down = _consecutive_errors >= DOWN_AFTER
    errors = _consecutive_errors
  if down:
    trace.emit("backend_down", kind="typed", provider="typesafe", consecutive_errors=errors)
    raise BackendDown(f"{errors} consecutive typed-backend errors")
  typed = build_questions(kind, questions, float_resp)  # a bad item raises here, before any count
  try:
    response = _send(agent_desc, typed)
  except Failed:
    with _lock:
      _consecutive_errors += 1
    raise
  with _lock:
    _consecutive_errors = 0
    models_seen.add(response.model)
  answers = response.answers
  return [_item(kind, options, float_resp, answers.get(f"q{i}"), response.model)
          for i, options in enumerate(questions.values())]
