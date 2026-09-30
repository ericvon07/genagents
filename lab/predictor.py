"""The decision seam: who answers the forced choice (003-IP DS-F, deviation D4).

The paper's `categorical_resp` and `numerical_resp` build `agent_desc` as published and
then call `answer` here instead of their `run_gpt_generate_*`. `PREDICTOR` (settings, D1)
picks the backend at call time:

- `cot` calls the paper's `run_gpt_generate_categorical_resp` / `_numerical_resp`
  unchanged, with the paper's arguments, and returns their output unchanged: with `cot`
  the fork behaves as the published code (law 2).
- `jev` hands the same `agent_desc` and the same question dict to `lab.jev_backend`
  (003-IP step 4) and returns its picks in the paper's output shape.

Either way the seam writes one `decision` event per item: the predictor, the kind, the
`state_sha1` of the very `agent_desc` both arms receive, the item's position, text and
options, `ok`, the raw `predicted`, and `in_domain`. It records; it never corrects — an
answer outside the options is kept as the model wrote it and flagged, and a failed request
is recorded `ok: false` with `predicted: null` and then raised again, never replaced by a
fail-safe.
"""
import hashlib

from lab import trace
from simulation_engine import settings

KINDS = ("categorical", "numerical")


def state_sha1(agent_desc):
  return hashlib.sha1(agent_desc.encode("utf-8")).hexdigest()


def in_domain(kind, predicted, options, float_resp=False):
  """Is `predicted` one of the item's options (categorical) or inside its range (numerical)?"""
  if predicted is None:
    return False
  if kind == "categorical":
    return predicted in options
  if isinstance(predicted, bool) or not isinstance(predicted, (int, float)):
    return False
  if not float_resp and predicted != int(predicted):
    return False
  low, high = options[0], options[-1]
  return low <= predicted <= high


def _cot(kind, agent_desc, questions, float_resp):
  """The paper's predictor: its own function, its own arguments, its own output.

  The model is `settings.LLM_VERS` read at call time, as `PREDICTOR` is: the paper's
  `interaction` copies it at import, and the runner's `--llm` (005-IP PL-E) sets it later.
  With nothing set the two are the same value.
  """
  from genagents.modules import interaction  # here, not above: interaction imports this module
  if kind == "categorical":
    output = interaction.run_gpt_generate_categorical_resp(
               agent_desc, questions, "1", settings.LLM_VERS)[0]
  else:
    output = interaction.run_gpt_generate_numerical_resp(
               agent_desc, questions, float_resp, "1", settings.LLM_VERS)[0]
  responses, reasonings = output["responses"], output["reasonings"]
  # The paper's parser finds answers by regex, so a skipped question shifts every later
  # answer onto the wrong item; the record says when the count does not match.
  aligned = len(responses) == len(questions)
  if not aligned:
    # 011-F7: the reply itself, so a run can say why it parsed short (the cap, a refusal,
    # another shape); the records are written as before
    from lab import transport
    trace.emit("unaligned", questions=len(questions), answers=len(responses),
               reply=transport.last_reply())
  items = [{"predicted": responses[i] if i < len(responses) else None,
            "reasoning": reasonings[i] if i < len(reasonings) else None,
            "aligned": aligned}
           for i in range(len(questions))]
  return output, items


def _jev(kind, agent_desc, questions, float_resp):
  """The typed backend: one request per batch, the pick and its distribution per item."""
  from lab import jev_backend
  items = jev_backend.answer(kind, agent_desc, questions, float_resp)
  output = {"responses": [item["predicted"] for item in items], "reasonings": []}
  return output, items


BACKENDS = {"cot": _cot, "jev": _jev}


def backend(predictor):
  if predictor not in BACKENDS:
    raise ValueError(f"PREDICTOR is {predictor!r}; known: {sorted(BACKENDS)}")
  if predictor == "jev":
    try:
      from lab import jev_backend  # noqa: F401
    except ModuleNotFoundError as exc:
      raise NotImplementedError("the typed backend lands in 003-IP step 4") from exc
  return BACKENDS[predictor]


def answer(kind, agent_desc, questions, float_resp=False):
  """D4: answer one batch with the selected predictor, recording one decision per item."""
  if kind not in KINDS:
    raise ValueError(f"kind is {kind!r}; known: {KINDS}")
  predictor = settings.PREDICTOR
  run = backend(predictor)
  base = {"predictor": predictor, "kind": kind, "state_sha1": state_sha1(agent_desc)}
  if kind == "numerical":
    base["float_resp"] = bool(float_resp)
  batch = list(questions.items())
  try:
    output, items = run(kind, agent_desc, questions, float_resp)
  except Exception as exc:
    for position, (item, options) in enumerate(batch):
      trace.emit("decision", **base, position=position, item=item, options=options,
                 ok=False, predicted=None, in_domain=False,
                 error=f"{type(exc).__name__}: {exc}")
    raise
  for position, ((item, options), result) in enumerate(zip(batch, items)):
    trace.emit("decision", **base, position=position, item=item, options=options, ok=True,
               in_domain=in_domain(kind, result["predicted"], options, float_resp),
               **result)
  return output
