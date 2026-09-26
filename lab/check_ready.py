"""Refuse a launch that lacks the keys its predictor needs (003-IP DS-C, Rule 8).

`load_env` reads <lab>/.env into the environment without overriding what is already set;
`require` raises before anything is written under runs-local/. Both predictors need
OpenRouter (chat for `cot`, embeddings for the retrieval both arms share); `jev` also needs
TypeSafe. OPENAI_API_KEY is optional: only the embedding fallback of DS-D uses it.
"""
import os

LAB_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NEEDS = {
  "cot": ("OPENROUTER_API_KEY",),
  "jev": ("OPENROUTER_API_KEY", "TYPESAFE_API_KEY"),
}


class NotReady(RuntimeError):
  pass


def load_env(path=None, environ=None):
  """KEY=VALUE lines; blanks, comments and an `export ` prefix are tolerated."""
  path = path or os.path.join(LAB_ROOT, ".env")
  environ = os.environ if environ is None else environ
  if not os.path.isfile(path):
    return environ
  with open(path) as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#") or "=" not in line:
        continue
      if line.startswith("export "):
        line = line[len("export "):]
      key, value = line.split("=", 1)
      key, value = key.strip(), value.strip().strip('"').strip("'")
      environ.setdefault(key, value)
  return environ


def require(predictor, environ=None):
  """Raise NotReady naming every missing key; return the keys checked."""
  environ = os.environ if environ is None else environ
  if predictor not in NEEDS:
    raise NotReady("unknown predictor %r; expected one of %s"
                   % (predictor, ", ".join(sorted(NEEDS))))
  missing = [k for k in NEEDS[predictor] if not environ.get(k)]
  if missing:
    raise NotReady("predictor %r needs %s in the environment (put them in <lab>/.env)"
                   % (predictor, ", ".join(missing)))
  return NEEDS[predictor]


if __name__ == "__main__":
  import sys
  load_env()
  predictor = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PREDICTOR", "cot")
  try:
    print("ready: %s (%s)" % (predictor, ", ".join(require(predictor))))
  except NotReady as e:
    sys.exit("not ready: %s" % e)
