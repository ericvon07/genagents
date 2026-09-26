"""The runner: one arm over a population and the item bank, into runs-local/<name>/ (003-IP DS-H).

  cd <genagents>
  python -m lab.runner smoke-cot-a --predictor cot --population example+demographic:20 --seed 20260926
  python -m lab.runner dry-jev-a --predictor jev --population example+demographic:1 --items 8

A launch, in order:

1. `<lab>/.env` is loaded and `check_ready.require` refuses a predictor whose keys are
   missing, before anything exists under runs-local/ (S8.1). An existing run folder is
   refused too: there is no resumption; a crashed run is registered and relaunched under a
   new name.
2. The population is drawn (`example`, the authors' interview agent; `demographic:N`, N
   folders of `gss_agents/` drawn by the seed from the sorted list) and hashed; the items
   are loaded and cut into batches (`lab.items`); the knobs are fingerprinted.
3. For the chain-of-thought arm, the provider query (DS-I) names the pin and the
   fallbacks, and `transport.policy` is set from them.
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
   `ok: false` and the agent goes on; `BackendDown` stops every agent.
7. `run.json` gains `outcome` (`finished`, `models_seen`, `stopped`). A run without
   `outcome` crashed, and the registrar invalidates it.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time

from lab import check_ready, items, trace

FORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.dirname(FORK)
RUNS_LOCAL = os.path.join(LAB, "runs-local")
POPULATIONS = os.path.join(FORK, "agent_bank", "populations")
EXAMPLE = os.path.join("single_agent", "01fd7d2a-0357-4c1b-9f3e-8eade2d537ae")
DEMOGRAPHIC = "gss_agents"

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

def population(spec, seed, root=POPULATIONS):
  """[(agent_id, agent_type, folder)] for `example+demographic:N`, in that order."""
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
      raise ValueError("population part %r; known: example, demographic:N" % part)
  return [(os.path.basename(folder), kind, folder) for kind, folder in agents]


def population_sha1(agents):
  h = hashlib.sha1()
  for agent_id, _, folder in agents:
    h.update(agent_id.encode() + b"\0")
    for name in ("scratch.json", "memory_stream/nodes.json", "memory_stream/embeddings.json"):
      sha1_file(os.path.join(folder, name), h)
  return h.hexdigest()


def first_memory(agents):
  """(text, stored vector) of the first node of the first agent with memories, or None."""
  for _, _, folder in agents:
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


def answer_agent(agent_id, agent_type, folder, batches, collector, write, stop):
  from genagents.genagents import GenerativeAgent
  from lab.jev_backend import BackendDown
  agent = GenerativeAgent(folder)
  for b, batch in enumerate(batches):
    if stop.is_set():
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
        stop.set()
      except Exception as exc:
        error = "%s: %s" % (type(exc).__name__, exc)
    latency_ms = round((monotonic() - started) * 1000)
    write(records(agent_id, agent_type, b, batch, collector.take(agent_id, b), error,
                  latency_ms))


# --- the launch -----------------------------------------------------------------

class GateFailed(RuntimeError):
  pass


def launch(name, predictor, population_spec="example+demographic:20", seed=20260926,
           instruments=items.INSTRUMENTS, n_items=None, threads=4, pin=None,
           min_calls_per_minute=None, out=RUNS_LOCAL, env_file=None, gate=health_gate,
           choose=None):
  """Run one arm; return the run's folder. Raises NotReady or GateFailed having written nothing."""
  check_ready.load_env(env_file)
  check_ready.require(predictor)
  folder = os.path.join(out, name)
  if os.path.exists(folder):
    raise FileExistsError("%s exists; a run is never resumed, relaunch under a new name" % folder)

  from lab import jev_backend, transport
  from simulation_engine import settings
  settings.PREDICTOR = predictor
  jev_backend.reset()
  transport.embeddings_direct = False
  transport.policy = transport.Policy()

  agents = population(population_spec, seed)
  bank = items.load(instruments)[:n_items]
  batches = items.batches(bank, settings.MAX_CHUNK_SIZE)
  config = {"predictor": predictor, "llm": transport.openrouter_model(settings.LLM_VERS),
            "typed_model": jev_backend.MODEL, "population": population_spec, "seed": seed,
            "instruments": list(instruments), "items": n_items,
            "batch_size": settings.MAX_CHUNK_SIZE, "threads": threads,
            "declared": list(transport.DECLARED), "pin": pin,
            "min_calls_per_minute": min_calls_per_minute}

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
    model = sorted(jev_backend.models_seen)[-1] if jev_backend.models_seen else None
    request = {"asked": jev_backend.MODEL, "timeout_s": jev_backend.TIMEOUT_S,
               "attempts": jev_backend.ATTEMPTS, "score_levels": jev_backend.SCORE_LEVELS}
  else:
    model = transport.openrouter_model(settings.LLM_VERS)
    request = {"temperature": 0.7, "max_tokens": 1500, "attempts": transport.ATTEMPTS}
  run = {
    "name": name,
    "condition": {"predictor": predictor, "model": model,
                  "agent_type": "+".join(dict.fromkeys(t for _, t, _ in agents)),
                  "instrument": "+".join(instruments), "prompt_version": "1",
                  "request": request},
    "code": code_state(),
    "data": {"population": population_spec,
             "population_path": "<genagents>/agent_bank/populations",
             "population_sha1": population_sha1(agents), "agents": len(agents),
             "agent_ids": [a for a, _, _ in agents],
             "items": {"path": "<osf>/figure2/data/question_master", "count": len(bank),
                       "sha1": items.bank_sha1(instruments), "batches": len(batches),
                       "batch_size": settings.MAX_CHUNK_SIZE},
             "held_out": "nothing is scored in this run; the demographic items the scratch "
                         "carries are the 27 the paper excludes"},
    "seed": seed,
    "config": config,
    "config_sha1": hashlib.sha1(json.dumps(config, sort_keys=True).encode()).hexdigest(),
    "providers": providers,
    "embeddings": (verdict.get("embedding") or {}).get("route", "not used"),
    "health": verdict,
    "started": now_iso(),
  }
  os.makedirs(folder)
  run_path = os.path.join(folder, "run.json")
  write_json(run_path, run)

  stop, collector, lock = threading.Event(), Collector(), threading.Lock()
  predictions = open(os.path.join(folder, "predictions.jsonl"), "a", encoding="utf-8")

  def write(lines):
    with lock:
      for line in lines:
        predictions.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
      predictions.flush()

  with trace.Trace(os.path.join(folder, "trace.jsonl")) as t:
    for event in gate_events:
      t.write(event)
    trace.SINKS.append(collector)
    try:
      with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(answer_agent, a, kind, f, batches, collector, write, stop)
                   for a, kind, f in agents]
        for future in futures:
          future.result()
    finally:
      trace.SINKS.remove(collector)
      predictions.close()

  run["outcome"] = {"finished": now_iso(), "models_seen": sorted(jev_backend.models_seen),
                    "stopped": "backend_down" if stop.is_set() else None}
  write_json(run_path, run)
  return folder


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
  ap.add_argument("--out", default=RUNS_LOCAL)
  args = ap.parse_args(argv)
  try:
    folder = launch(args.name, args.predictor, args.population, args.seed,
                    tuple(args.instruments.split(",")), args.items, args.threads, args.pin,
                    args.min_calls_per_minute, args.out)
  except (check_ready.NotReady, GateFailed, FileExistsError) as exc:
    sys.exit("not launched: %s" % exc)
  with open(os.path.join(folder, "run.json")) as f:
    outcome = json.load(f)["outcome"]
  print("%s: finished %s%s" % (folder, outcome["finished"],
                               ", STOPPED: %s" % outcome["stopped"] if outcome["stopped"] else ""))


if __name__ == "__main__":
  main()
