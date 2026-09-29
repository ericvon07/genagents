"""The run's trace: every event a run makes, one JSONL line each (003-IP step 3).

The event shape is the sister lab's (System-1-Agentic-World, trace-and-manifest, 2026-09-22),
cut to what a prediction run makes: `manifest`, `call`, `rate_limit`, `error`, `decision`,
`provider_switch`, `backend_down`, plus `health`, the verdict of the gate before launch
(003-IP DS-J), and the two stops of 005-IP PL-H: `spend_cap` (the next batch could pass the
run's cap) and `credits_out` (a backend answered 402). Smallville's `reject` and `outcome` describe actions in a world and have no
counterpart here. A name outside `EVENTS` is refused, so a typo cannot
write an event no reader looks for.

`emit` sends a record to every sink in `SINKS`; a `Trace` opened on a path is one such sink
and writes each record as a line, flushed, under a lock (the runner answers agents in
threads). `bound(**fields)` adds fields to every event emitted inside it in the same thread
or task — the runner binds the agent and the batch there, so the transport and the seam
never need to know them. With no sink an event is dropped.
"""
import contextlib
import contextvars
import json
import threading
import time

EVENTS = {"manifest", "call", "rate_limit", "error", "decision", "provider_switch",
          "backend_down", "health", "spend_cap", "credits_out"}

SINKS = []
_bound = contextvars.ContextVar("lab_trace_bound", default={})


def emit(event, **fields):
  if event not in EVENTS:
    raise ValueError(f"unknown trace event {event!r}; known: {sorted(EVENTS)}")
  record = {"event": event, "ts": time.time(), **_bound.get(), **fields}
  for sink in list(SINKS):
    sink(record)
  return record


@contextlib.contextmanager
def bound(**fields):
  token = _bound.set({**_bound.get(), **fields})
  try:
    yield
  finally:
    _bound.reset(token)


class Trace:
  """A JSONL file that receives every event while it is open."""

  def __init__(self, path):
    self.path = path
    self._lock = threading.Lock()
    self._file = None

  def __enter__(self):
    self._file = open(self.path, "a", encoding="utf-8")
    SINKS.append(self.write)
    return self

  def __exit__(self, *exc):
    SINKS.remove(self.write)
    self._file.close()

  def write(self, record):
    line = json.dumps(record, ensure_ascii=False, default=str)
    with self._lock:
      self._file.write(line + "\n")
      self._file.flush()


def read(path):
  with open(path, encoding="utf-8") as f:
    return [json.loads(line) for line in f if line.strip()]
