"""The runner: one arm over a population and the item bank, into runs-local/<name>/ (003-IP DS-H).

  cd <genagents>
  python -m lab.runner smoke-cot-a --predictor cot --llm gpt-4o-mini --population example+demographic:20 \
    --seed 20260926
  python -m lab.runner dry-jev-a --predictor jev --population example+demographic:1 --items 8
  python -m lab.runner gss22-cot-4o-p --predictor cot --llm gpt-4o --population built:gss22-a:10 \
    --max-usd 5

A launch, in order:

1. `<lab>/.env` is loaded and `check_ready.require` refuses a predictor whose keys are
   missing, before anything exists under runs-local/ (S8.1). An existing run folder is
   refused too: there is no resumption; a crashed run is registered and relaunched under a
   new name. So is a `cot` run whose model is not named (`--llm` or `LLM_VERS`): the
   published default is `gpt-4o-mini`, the paper's model `gpt-4o`, and a run that fell to
   the default would say so only in its model field.
2. The population is drawn (`example`, the authors' interview agent; `demographic:N`, N
   folders of `gss_agents/` drawn by the seed from the sorted list; `built:<name>[:N]`, both
   agents of the first N respondents of a population built under `LAB_DATA`, each with its
   own items, 005-IP PL-D) and hashed; the items are loaded and cut into batches
   (`lab.items`), per agent; the knobs are fingerprinted.
3. The credit gate (005-IP PL-H): the OpenRouter balance must cover the spend cap plus
   US$0.50, or the launch is refused having written nothing. For the chain-of-thought arm,
   the provider query (DS-I) then names the pin and the fallbacks, and `transport.policy`
   is set from them.
4. The health gate (DS-J): three answers in a row from the arm's backend, none more than a
   minute after the one before, and, when an agent has memories, the embedding identity
   check (DS-D) — its failure turns the direct-to-OpenAI route on, and the run says so. The
   gate's events are held in memory: a failed gate writes nothing.
5. The run folder is created, `run.json` written, and the gate's events become the
   trace's first lines — all before the first request of the run (S5.1, S8.2).
6. Agents are answered in threads, each its batches in bank order through the paper's
   `categorical_resp` / `numerical_resp`, so the state is built by the paper's code and the
   seam (D4) records each item. As a batch ends its items become lines of
   `predictions.jsonl`, taken from the seam's `decision` events. A failed batch is recorded
   `ok: false` and the agent goes on. Three things stop every agent: `BackendDown`, a 402
   (`CreditsOut`), and the meter refusing a batch that could pass `--max-usd` (a
   `spend_cap` event).
7. `run.json` gains `outcome` (`finished`, `models_seen`, `stopped`, `spent_usd`). A run
   without `outcome` crashed, and the registrar invalidates it.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time

from lab import check_ready, items, trace

FORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def newest(models):
  """The latest version among `models` ("jev-1.13.0" after "jev-1.9.0"), or None (011-F7)."""
  def key(model):
    return [(0, int(part)) if part.isdigit() else (1, part)
            for part in re.split(r"[.\-]", model)]
  return max(models, key=key) if models else None


def paper_max_tokens():
  """The cap the paper's `gpt_request` sends, read from its signature rather than retyped, so
  `run.json` and the readout's cut count follow the code (011-F7)."""
  import inspect
  from simulation_engine.gpt_structure import gpt_request
  return inspect.signature(gpt_request).parameters["max_tokens"].default
LAB = os.path.dirname(FORK)
RUNS_LOCAL = os.path.join(LAB, "runs-local")
POPULATIONS = os.path.join(FORK, "agent_bank", "populations")
EXAMPLE = os.path.join("single_agent", "01fd7d2a-0357-4c1b-9f3e-8eade2d537ae")
DEMOGRAPHIC = "gss_agents"
BUILT = "built:"
BUILT_TYPES = ("demographic", "survey")  # both agents of a respondent, in this order (GP-C)
VARIANTS = {"cot": (None,), "jev": (None, "steps")}
UNNAMED_MODEL = (
  "a cot run must name its chain-of-thought model: pass --llm gpt-4o (the paper's model) or "
  "--llm gpt-4o-mini, or set LLM_VERS. Unnamed, the published code's default, gpt-4o-mini, "
  "would answer, and run.json would look as clean as a gpt-4o run's")
MARGIN_USD = 0.50  # PL-H: the balance must cover the cap and this much more

GATE_ANSWERS = 3
GATE_MAX_GAP_S = 60
GATE_ATTEMPTS = 3  # the gate asks for a healthy backend, not one that answers after an hour
GATE_STATE = "Self description: {'first_name': 'Test', 'age': '40'}\n==\nOther observations about the subject:\n\n"
GATE_QUESTION = {"Is this a test of the connection?": ["Yes", "No"]}
GATE_PROMPT = "Reply with the single word: ready."

monotonic = time.monotonic


def now_iso():
  return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha1_file(path, h=None):
  h = h or hashlib.sha1()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h


# --- the population -------------------------------------------------------------

def population(spec, seed, root=POPULATIONS, built=None):
  """[(agent_id, agent_type, folder, own)] for `example+demographic:N` or `built:<name>[:N]`.

  `own` is None for an agent that answers the run's items, and the item ids `items.json`
  holds for an agent of a built population (PL-D). A built population is not mixed with
  other parts.
  """
  if spec.startswith(BUILT):
    return built_agents(spec, built or built_root())
  agents = []
  for part in spec.split("+"):
    if part == "example":
      agents.append(("interview", os.path.join(root, EXAMPLE)))
    elif part.startswith("demographic:"):
      folders = sorted(f for f in os.listdir(os.path.join(root, DEMOGRAPHIC))
                       if not f.startswith("."))
      drawn = random.Random(seed).sample(folders, int(part.split(":", 1)[1]))
      agents += [("demographic", os.path.join(root, DEMOGRAPHIC, f)) for f in drawn]
    else:
      raise ValueError("population part %r; known: example, demographic:N, or alone "
                       "built:<name>[:N]" % part)
  return [(os.path.basename(folder), kind, folder, None) for kind, folder in agents]


def built_root(environ=None):
  """`LAB_DATA/populations`, where the lab's builder writes (004-ST GP-J)."""
  environ = os.environ if environ is None else environ
  if not environ.get("LAB_DATA"):
    raise ValueError("a built population needs LAB_DATA in the environment (<lab>/.env)")
  return os.path.join(os.path.expanduser(environ["LAB_DATA"]), "populations")


def built_spec(spec):
  """(name, N or None) of `built:<name>[:N]`."""
  name, _, n = spec[len(BUILT):].partition(":")
  if not name or (n and not n.isdigit()):
    raise ValueError("population %r; expected built:<name>[:N]" % spec)
  return name, int(n) if n else None


def built_agents(spec, root):
  """Both agents of the first N respondents in draw order, each with its held-out items."""
  name, n = built_spec(spec)
  folder = os.path.join(root, name)
  meta, own = read_json(folder, "population.json"), read_json(folder, "items.json")
  respondents = sorted(meta["respondents"], key=lambda r: r["position"])
  if n is not None:
    if not 0 < n <= len(respondents):
      raise ValueError("%s holds %d respondents, not %d" % (name, len(respondents), n))
    respondents = respondents[:n]
  return [(r["agents"][kind], kind, os.path.join(folder, "agents", r["agents"][kind]),
           own[r["agents"][kind]])
          for r in respondents for kind in BUILT_TYPES]


def built_data(spec, root):
  """What `data` says of a built population: its files by sha256, checked against the
  build's own record, so a population changed after its build is refused."""
  name, n = built_spec(spec)
  folder = os.path.join(root, name)
  meta = read_json(folder, "population.json")
  files = {f: sha256_file(os.path.join(folder, f)) for f in ("items.json", "truth.jsonl")}
  for f, digest in files.items():
    if meta["files"].get(f) != digest:
      raise ValueError("%s/%s is not the file its build recorded; rebuild the population"
                       % (name, f))
  return {"population_path": "<lab-data>/populations/%s" % name,
          "population_json_sha256": sha256_file(os.path.join(folder, "population.json")),
          "items_sha256": files["items.json"], "truth_sha256": files["truth.jsonl"],
          "respondents": n if n is not None else len(meta["respondents"]),
          "held_out": "%d scorable items per respondent, 004-ST GP-E" % meta["slice"]["k"]}


def read_json(folder, name):
  with open(os.path.join(folder, name), encoding="utf-8") as f:
    return json.load(f)


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def population_sha1(agents):
  h = hashlib.sha1()
  for agent_id, _, folder, own in agents:
    h.update(agent_id.encode() + b"\0")
    for name in ("scratch.json", "memory_stream/nodes.json", "memory_stream/embeddings.json"):
      sha1_file(os.path.join(folder, name), h)
    if own is not None:
      h.update(json.dumps(own).encode())
  return h.hexdigest()


def agent_batches(agents, bank, n_items, size):
  """{agent_id: batches}: the run's items, or the agent's own in bank order (PL-D); `n_items`
  keeps the first N of either. An own item the bank does not hold is refused."""
  index = {item.id: k for k, item in enumerate(bank)}
  out = {}
  for agent_id, _, _, own in agents:
    if own is None:
      chosen = bank
    else:
      missing = [q for q in own if q not in index]
      if missing:
        raise ValueError("%s is asked %s, which the run's bank does not hold"
                         % (agent_id, ", ".join(missing)))
      chosen = [bank[k] for k in sorted(index[q] for q in own)]
    out[agent_id] = items.batches(chosen[:n_items], size)
  return out


def per_agent(counts):
  """One count when every agent has it, else the sorted counts seen."""
  seen = sorted(set(counts))
  return seen[0] if len(seen) == 1 else seen


def first_memory(agents):
  """(text, stored vector) of the first node of the first agent with memories, or None."""
  for _, _, folder, _ in agents:
    with open(os.path.join(folder, "memory_stream", "nodes.json")) as f:
      nodes = json.load(f)
    if nodes:
      with open(os.path.join(folder, "memory_stream", "embeddings.json")) as f:
        embeddings = json.load(f)
      return nodes[0]["content"], embeddings[nodes[0]["content"]]
  return None


def code_state(repo=FORK):
  def git(*args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                          check=True).stdout.strip()
  return {"sha": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
          "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
          "dirty": bool(git("status", "--porcelain"))}


# --- the health gate (DS-J) -----------------------------------------------------

def health_gate(predictor, agents):
  """Three answers from the arm's backend and the embedding identity check.

  Returns (verdict, events): the events are the gate's, held in memory, and the verdict is
  also the last of them, a `health` event.
  """
  from lab import jev_backend, transport
  from simulation_engine import settings
  events = []
  trace.SINKS.append(events.append)
  saved = transport.ATTEMPTS, jev_backend.ATTEMPTS
  transport.ATTEMPTS = jev_backend.ATTEMPTS = GATE_ATTEMPTS
  backend = "typed" if predictor == "jev" else "chat"
  verdict = {"passed": False, "backend": backend, "gaps_ms": [], "embedding": None}
  try:
    with trace.bound(phase="health"):
      last = monotonic()
      for _ in range(GATE_ANSWERS):
        try:
          if backend == "typed":
            jev_backend.answer("categorical", GATE_STATE, GATE_QUESTION)
          else:
            transport.chat(GATE_PROMPT, settings.LLM_VERS, 16, 0.7)
        except Exception as exc:
          verdict["error"] = "%s: %s" % (type(exc).__name__, exc)
          break
        gap = monotonic() - last
        last += gap
        verdict["gaps_ms"].append(round(gap * 1000))
        if gap > GATE_MAX_GAP_S:
          verdict["error"] = "an answer came %.0f s after the one before" % gap
          break
      answered = len(verdict["gaps_ms"]) == GATE_ANSWERS and "error" not in verdict
      memory = first_memory(agents) if answered else None
      if memory:
        verdict["embedding"] = check_embeddings(*memory)
        if not verdict["embedding"]["passed"]:
          verdict["error"] = "embeddings differ from the stored ones on both routes"
      verdict["passed"] = answered and (memory is None or verdict["embedding"]["passed"])
      if backend == "typed":
        verdict["models"] = sorted(jev_backend.models_seen)
      trace.emit("health", **verdict)
  finally:
    transport.ATTEMPTS, jev_backend.ATTEMPTS = saved
    trace.SINKS.remove(events.append)
  return verdict, events


def check_embeddings(text, stored):
  """DS-D: OpenRouter's embedding of a stored node, and the direct route if it differs."""
  from lab import transport
  from simulation_engine import settings
  try:
    cosine, passed = transport.embedding_identity(text, stored)
  except Exception as exc:
    cosine, passed = None, False
    error = "%s: %s" % (type(exc).__name__, exc)
  else:
    error = None
  result = {"route": "openrouter", "cosine": cosine, "passed": passed}
  if error:
    result["error"] = error
  if not passed and settings.OPENAI_API_KEY:
    transport.embeddings_direct = True
    try:
      direct, passed = transport.embedding_identity(text, stored)
    except Exception as exc:
      direct, passed = None, False
      result["direct_error"] = "%s: %s" % (type(exc).__name__, exc)
    result.update(route="openai-direct", openrouter_cosine=cosine, cosine=direct,
                  passed=passed)
  return result


# --- the spend cap and the stop (PL-H) ------------------------------------------

def prices(llm):
  """US$ per million tokens, by `call` kind: the run's chat model, the embeddings, Jev."""
  from lab import jev_backend, transport
  chat, embedding = transport.PRICES[llm], transport.PRICES[transport.EMBEDDING_MODEL]
  return {"chat": {"model": llm, "in": chat[0], "out": chat[1]},
          "embedding": {"model": transport.EMBEDDING_MODEL, "in": embedding[0],
                        "out": embedding[1]},
          "typed": {"model": jev_backend.MODEL, "in": jev_backend.USD_PER_MTOK[0],
                    "out": jev_backend.USD_PER_MTOK[1]}}


class Meter:
  """The run's spend at list price, and the cap's judgement (PL-H).

  A trace sink: every `call` event is priced by its kind at `prices`. Before a batch starts,
  `admit` refuses it when what was spent, plus the dearest batch so far for it and for every
  batch still in flight, would pass `max_usd`; it then returns the figures the `spend_cap`
  event carries. Without a cap every batch is admitted and the spend is still counted.
  """

  def __init__(self, prices, max_usd=None):
    self.prices, self.max_usd = prices, max_usd
    self.spent, self.dearest, self.open = 0.0, 0.0, {}
    self._lock = threading.Lock()

  def cost(self, record):
    price = self.prices.get(record.get("kind"))
    if price is None:
      return 0.0
    return ((record.get("tokens_in") or 0) * price["in"]
            + (record.get("tokens_out") or 0) * price["out"]) / 1e6

  def __call__(self, record):
    if record["event"] != "call":
      return
    cost, key = self.cost(record), (record.get("agent"), record.get("batch"))
    with self._lock:
      self.spent += cost
      if key in self.open:
        self.open[key] += cost

  def admit(self, agent_id, batch):
    with self._lock:
      if self.max_usd is not None:
        projected = self.spent + self.dearest * (len(self.open) + 1)
        if projected > self.max_usd:
          return {"spent_usd": round(self.spent, 6), "projected_usd": round(projected, 6),
                  "dearest_batch_usd": round(self.dearest, 6), "in_flight": len(self.open),
                  "max_usd": self.max_usd}
      self.open[(agent_id, batch)] = 0.0
      return None

  def close(self, agent_id, batch):
    with self._lock:
      self.dearest = max(self.dearest, self.open.pop((agent_id, batch), 0.0))


class Stop:
  """Why the run stopped early, set once: `backend_down`, `credits_out` or `spend_cap`."""

  def __init__(self):
    self.reason, self._lock = None, threading.Lock()

  def set(self, reason):
    """True for the first reason given; a later one is ignored."""
    with self._lock:
      if self.reason is None:
        self.reason = reason
        return True
      return False

  def is_set(self):
    return self.reason is not None


class CreditsShort(RuntimeError):
  """The balance does not cover the cap plus the margin, or could not be read."""


def credit_gate(max_usd, read):
  """PL-H: the balance must be at least the cap plus `MARGIN_USD`; returns what run.json keeps."""
  try:
    credits = read()
  except Exception as exc:
    raise CreditsShort("could not read the OpenRouter balance: %s: %s"
                       % (type(exc).__name__, exc)) from exc
  balance = credits["total_credits"] - credits["total_usage"]
  needed = (max_usd or 0) + MARGIN_USD
  if balance < needed:
    raise CreditsShort("the OpenRouter balance is US$%.2f; this run needs US$%.2f (a cap of "
                       "US$%s plus US$%.2f)" % (balance, needed, max_usd, MARGIN_USD))
  return {**credits, "balance": round(balance, 6), "needed": needed, "read_at": now_iso()}


# --- one agent ------------------------------------------------------------------

class Collector:
  """A trace sink that keeps each batch's `decision` events until the runner takes them."""

  def __init__(self):
    self.held, self._lock = {}, threading.Lock()

  def __call__(self, record):
    if record["event"] == "decision" and "batch" in record:
      with self._lock:
        self.held.setdefault((record.get("agent"), record["batch"]), []).append(record)

  def take(self, agent_id, batch):
    with self._lock:
      return {d["position"]: d for d in self.held.pop((agent_id, batch), [])}


KEPT = ("ok", "error", "predicted", "in_domain", "primitive", "confidence", "model",
        "reasoning", "aligned", "state_sha1", "predictor")


def records(agent_id, agent_type, b, batch, decided, error, latency_ms):
  out = []
  for position, item in enumerate(batch):
    d = decided.get(position)
    r = {"agent": agent_id, "agent_type": agent_type, "item": item.id,
         "instrument": item.instrument, "kind": item.kind, "options": item.options,
         "flagged": item.flagged, "batch": b, "position": position}
    if d is None:  # the batch failed before the seam recorded it (e.g. retrieval's embedding)
      r.update(ok=False, predicted=None, in_domain=False,
               error=error or "no decision was recorded")
    else:
      r.update({k: d[k] for k in KEPT if k in d})
    r["p"] = d.get("distribution") if d else None
    r.update(fail_safe=False, latency_ms=latency_ms, ts=now_iso())
    out.append(r)
  return out


def answer_agent(agent_id, agent_type, folder, batches, collector, write, stop, meter):
  from genagents.genagents import GenerativeAgent
  from lab.jev_backend import BackendDown
  from lab.transport import CreditsOut
  agent = GenerativeAgent(folder)
  for b, batch in enumerate(batches):
    if stop.is_set():
      return
    with trace.bound(agent=agent_id, batch=b):
      refused = meter.admit(agent_id, b)
      if refused:
        if stop.set("spend_cap"):
          trace.emit("spend_cap", **refused)
        return
    questions = items.questions(batch)
    error, started = None, monotonic()
    with trace.bound(agent=agent_id, batch=b):
      try:
        if batch[0].kind == "categorical":
          agent.categorical_resp(questions)
        else:
          agent.numerical_resp(questions, batch[0].float_resp)
      except BackendDown as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
        stop.set("backend_down")
      except CreditsOut as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
        stop.set("credits_out")
      except Exception as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
    meter.close(agent_id, b)
    latency_ms = round((monotonic() - started) * 1000)
    write(records(agent_id, agent_type, b, batch, collector.take(agent_id, b), error,
                  latency_ms))


# --- the launch -----------------------------------------------------------------

class GateFailed(RuntimeError):
  pass


def launch(name, predictor, population_spec="example+demographic:20", seed=20260926,
           instruments=items.INSTRUMENTS, n_items=None, threads=4, pin=None,
           min_calls_per_minute=None, out=RUNS_LOCAL, env_file=None, gate=health_gate,
           choose=None, llm=None, variant=None, max_usd=None, credits=None):
  """Run one arm; return the run's folder. Raises NotReady, CreditsShort, GateFailed or
  ValueError having written nothing."""
  check_ready.load_env(env_file)
  check_ready.require(predictor)
  folder = os.path.join(out, name)
  if os.path.exists(folder):
    raise FileExistsError("%s exists; a run is never resumed, relaunch under a new name" % folder)
  if variant not in VARIANTS[predictor]:
    raise ValueError("variant %r is not one of the %s arm's: %s"
                     % (variant, predictor, VARIANTS[predictor]))
  if predictor == "cot" and not (llm or os.environ.get("LLM_VERS")):
    raise ValueError(UNNAMED_MODEL)

  from lab import jev_backend, transport
  from simulation_engine import settings
  settings.PREDICTOR = predictor
  if llm:
    settings.LLM_VERS = llm  # PL-E; the seam reads it at call time
  chat_model = transport.openrouter_model(settings.LLM_VERS)
  declared = transport.declared_for(chat_model)
  jev_backend.reset()
  jev_backend.variant = variant
  transport.embeddings_direct = False
  transport.policy = transport.Policy()

  agents = population(population_spec, seed)
  bank = items.load(instruments)
  batches = agent_batches(agents, bank, n_items, settings.MAX_CHUNK_SIZE)
  asked = [item for agent_id in batches for batch in batches[agent_id] for item in batch]
  used = [i for i in instruments if any(item.instrument == i for item in asked)]
  built = built_data(population_spec, built_root()) if population_spec.startswith(BUILT) else None
  price = prices(chat_model)
  config = {"predictor": predictor, "llm": chat_model, "variant": variant,
            "typed_model": jev_backend.MODEL, "population": population_spec, "seed": seed,
            "instruments": list(instruments), "items": n_items,
            "batch_size": settings.MAX_CHUNK_SIZE, "threads": threads,
            "declared": list(declared), "pin": pin,
            "min_calls_per_minute": min_calls_per_minute, "max_usd": max_usd,
            "prices": price}

  balance = credit_gate(max_usd, credits or transport.fetch_credits)
  log("credits: balance US$%.2f; cap %s" % (balance["balance"],
      "US$%.2f" % max_usd if max_usd is not None else "none"))

  providers = None
  if predictor == "cot":
    providers = (choose or transport.choose)(settings.LLM_VERS, force_pin=pin)
    providers["min_calls_per_minute"] = min_calls_per_minute
    transport.policy = transport.Policy(providers["pin"], providers["fallbacks"],
                                        min_calls_per_minute)

  verdict, gate_events = gate(predictor, agents)
  if not verdict["passed"]:
    raise GateFailed("health gate: %s" % verdict.get("error", verdict))

  if predictor == "jev":
    model = newest(jev_backend.models_seen)
    request = {"asked": jev_backend.MODEL, "timeout_s": jev_backend.TIMEOUT_S,
               "attempts": jev_backend.ATTEMPTS, "score_levels": jev_backend.SCORE_LEVELS}
  else:
    model = transport.openrouter_model(settings.LLM_VERS)
    request = {"temperature": 0.7, "max_tokens": paper_max_tokens(),
               "attempts": transport.ATTEMPTS}
  data = {"population": population_spec,
          "population_path": "<genagents>/agent_bank/populations",
          "population_sha1": population_sha1(agents), "agents": len(agents),
          "agent_ids": [a for a, _, _, _ in agents],
          "items": {"path": "<osf>/figure2/data/question_master",
                    "count": len({item.id for item in asked}),
                    "sha1": items.bank_sha1(instruments),
                    "batches": per_agent(len(b) for b in batches.values()),
                    "batch_size": settings.MAX_CHUNK_SIZE},
          "held_out": "nothing is scored in this run; the 27 items the paper excludes for "
                      "overlapping its interview are not asked"}
  if built:
    data.update(built)
    data["items"]["per_agent"] = per_agent(sum(map(len, b)) for b in batches.values())
  run = {
    "name": name,
    "condition": {"predictor": predictor, "model": model, "variant": variant,
                  "agent_type": "+".join(dict.fromkeys(t for _, t, _, _ in agents)),
                  "instrument": "+".join(used), "prompt_version": "1",
                  "request": request},
    "code": code_state(),
    "data": data,
    "seed": seed,
    "config": config,
    "config_sha1": hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest(),
    "providers": providers,
    "embeddings": (verdict.get("embedding") or {}).get("route", "not used"),
    "health": verdict,
    "credits": balance,
    "started": now_iso(),
  }
  os.makedirs(folder)
  run_path = os.path.join(folder, "run.json")
  write_json(run_path, run)

  stop, collector, lock = Stop(), Collector(), threading.Lock()
  meter = Meter(price, max_usd)
  for event in gate_events:  # the gate's requests are the run's spend too
    meter(event)
  predictions = open(os.path.join(folder, "predictions.jsonl"), "a", encoding="utf-8")

  def write(lines):
    with lock:
      for line in lines:
        predictions.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
      predictions.flush()

  with trace.Trace(os.path.join(folder, "trace.jsonl")) as t:
    for event in gate_events:
      t.write(event)
    trace.SINKS.extend((collector, meter))
    try:
      with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(answer_agent, a, kind, f, batches[a], collector, write, stop,
                               meter)
                   for a, kind, f, _ in agents]
        for future in futures:
          future.result()
    finally:
      trace.SINKS.remove(collector)
      trace.SINKS.remove(meter)
      predictions.close()

  run["outcome"] = {"finished": now_iso(), "models_seen": sorted(jev_backend.models_seen),
                    "stopped": stop.reason, "spent_usd": round(meter.spent, 6)}
  write_json(run_path, run)
  return folder


def log(message):
  print(message, file=sys.stderr, flush=True)


def write_json(path, data):
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
    f.write("\n")


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("name")
  ap.add_argument("--predictor", choices=("cot", "jev"), required=True)
  ap.add_argument("--population", default="example+demographic:20")
  ap.add_argument("--seed", type=int, default=20260926)
  ap.add_argument("--instruments", default=",".join(items.INSTRUMENTS))
  ap.add_argument("--items", type=int, default=None, help="only the first N items (a dry run)")
  ap.add_argument("--threads", type=int, default=4)
  ap.add_argument("--pin", default=None, help="force the chat provider instead of the query's best")
  ap.add_argument("--min-calls-per-minute", type=float, default=None)
  ap.add_argument("--llm", default=None, choices=("gpt-4o-mini", "gpt-4o"),
                  help="the chain-of-thought model; a cot run must name it here or in LLM_VERS")
  ap.add_argument("--variant", default=None, choices=("steps",),
                  help="jev only: steps is jev-steps, the paper's reasoning steps in the instruction")
  ap.add_argument("--max-usd", type=float, default=None,
                  help="the run's spend cap at list price; the balance must cover it plus US$0.50")
  ap.add_argument("--out", default=RUNS_LOCAL)
  args = ap.parse_args(argv)
  try:
    folder = launch(args.name, args.predictor, args.population, args.seed,
                    tuple(args.instruments.split(",")), args.items, args.threads, args.pin,
                    args.min_calls_per_minute, args.out, llm=args.llm, variant=args.variant,
                    max_usd=args.max_usd)
  except (check_ready.NotReady, CreditsShort, GateFailed, FileExistsError, ValueError) as exc:
    sys.exit("not launched: %s" % exc)
  with open(os.path.join(folder, "run.json")) as f:
    outcome = json.load(f)["outcome"]
  print("%s: finished %s, US$%.4f at list price%s"
        % (folder, outcome["finished"], outcome["spent_usd"],
           ", STOPPED: %s" % outcome["stopped"] if outcome["stopped"] else ""))


if __name__ == "__main__":
  main()
