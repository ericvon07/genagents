"""Side readings of the memory curve and the direct answers: numbers `readout.py` does not print.

Usage (from the repository root):

  python analysis/memory_curve_extras.py

It reads six runs (`runs/<name>/`, never `runs-local/`) under the
readout's `--first 600`, and prints, aggregates only:

- **Against the majority guess**: a run's raw accuracy
  minus the floor's on one agent type, with the bootstrap interval. `--against RUN` refuses when no
  other run of RUN's group shares its states (`gss22-d-jev-every`), so the interval is read
  here with the readout's own `bootstrap`, `outcome`, `same_state` and floor over the same
  records: the same seed (20260926), resamples (10,000) and respondents.
- **Unrounded pairs** whose intervals end close to zero.
- **Picks that change**: the share of records whose answer differs between two
  points, read without the truth, over the records both answered.

The records are the readout's (`expected` under `--first 600`): the trap records of
`universe.json` out, 17,852 per agent type. Raw accuracy only. Its pairs agree with
the readout's `--against` tables to the second decimal.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import readout as R  # noqa: E402

RUNS = ["gss22-c-jev", "gss22-c-cot-gemma", "gss22-c-direct-gemma",
        "gss22-d-jev-k10", "gss22-d-jev-k30", "gss22-d-jev-every"]
FIRST = 600


def load():
  """The six runs from the registry, their population's head, and each run's records."""
  root = os.path.dirname(HERE)
  runs = [R.load_run(os.path.join(root, "runs", name)) for name in RUNS]
  by = {run.name: run for run in runs}
  pop = R.population_of(runs, R.lab_data(), FIRST)
  keys = {run.name: R.expected(run, pop, FIRST) for run in runs}
  return by, pop, keys


def pair(by, pop, keys, a, b, agent_type):
  """a minus b (b None: the floor) in points of raw accuracy over a's records of one agent
  type, left out where the two sides hold another state, with the bootstrap interval."""
  base = by[a]
  ks = [k for k in keys[a] if pop.kind(k[0]) == agent_type]
  if b is not None:
    ks = [k for k in ks if R.same_state([base, by[b]], k)]
  order = {}
  for agent, _ in ks:
    order.setdefault(pop.agents[agent][0], len(order))
  columns = {}

  def column(name, value):
    sums = [0.0] * len(order)
    for k in ks:
      sums[order[pop.agents[k[0]][0]]] += value(k)
    columns[name] = sums

  def truth(k):
    return pop.row(k)["truth"]

  column("n", lambda k: 1)
  column("a", lambda k: R.outcome(base, k, truth(k)) == R.RIGHT)
  if b is None:
    column("b", lambda k: pop.row(k)[R.FLOOR] == truth(k))
  else:
    column("b", lambda k: R.outcome(by[b], k, truth(k)) == R.RIGHT)
  (value, low, high), = R.bootstrap(columns, [lambda t: 100 * (t["a"] - t["b"]) / t["n"]],
                                    R.SEED, R.RESAMPLES)
  return len(ks), value, low, high


def picks(by, pop, keys, name, agent_type):
  """Each record's answer, keyed by (respondent, item), so two agent types line up."""
  return {(pop.agents[k[0]][0], k[1]): R.answer(by[name], k)
          for k in keys[name] if pop.kind(k[0]) == agent_type}


def changed(a, b):
  common = [k for k in a if k in b and a[k] is not None and b[k] is not None]
  return len(common), sum(a[k] != b[k] for k in common)


def main():
  by, pop, keys = load()
  print("# Side readings of the memory curve, the first %d respondents, trap records out\n" % FIRST)
  print("Raw accuracy, points, 95%% interval over %s resamples of the respondents, seed %d.\n"
        % (format(R.RESAMPLES, ","), R.SEED))
  print("| side A | agent type | minus | records | difference, points | 95% interval |")
  print("|---|---|---|---|---|---|")
  for a, b, t in [("gss22-d-jev-every", None, "survey"), ("gss22-c-jev", None, "survey"),
                  ("gss22-c-direct-gemma", None, "survey"),
                  ("gss22-c-direct-gemma", None, "demographic"),
                  ("gss22-c-cot-gemma", None, "survey"),
                  ("gss22-c-cot-gemma", "gss22-c-direct-gemma", "survey"),
                  ("gss22-c-cot-gemma", "gss22-c-direct-gemma", "demographic"),
                  ("gss22-c-direct-gemma", "gss22-c-jev", "survey")]:
    n, value, low, high = pair(by, pop, keys, a, b, t)
    print("| %s | %s | %s | %d | %+.4f | [%+.4f, %+.4f] |"
          % (a, t, b or "the floor", n, value, low, high))
  print("\n| picks that change | records both answered | changed | share |")
  print("|---|---|---|---|")
  points = [("zero", picks(by, pop, keys, "gss22-c-jev", "demographic")),
            ("10", picks(by, pop, keys, "gss22-d-jev-k10", "survey")),
            ("30", picks(by, pop, keys, "gss22-d-jev-k30", "survey")),
            ("today", picks(by, pop, keys, "gss22-c-jev", "survey")),
            ("every", picks(by, pop, keys, "gss22-d-jev-every", "survey"))]
  rows = [("Jev, %s to %s" % (na, nb), a, b) for (na, a), (nb, b) in zip(points, points[1:])]
  rows += [("Jev, zero to every", points[0][1], points[4][1]),
           ("Jev, 30 to every", points[2][1], points[4][1])]
  for t in ("demographic", "survey"):
    rows.append(("Gemma with its reasoning against without, %s" % t,
                 picks(by, pop, keys, "gss22-c-cot-gemma", t),
                 picks(by, pop, keys, "gss22-c-direct-gemma", t)))
    rows.append(("Jev against Gemma without its reasoning, %s" % t,
                 picks(by, pop, keys, "gss22-c-jev", t),
                 picks(by, pop, keys, "gss22-c-direct-gemma", t)))
  for label, a, b in rows:
    n, c = changed(a, b)
    print("| %s | %d | %d | %.1f%% |" % (label, n, c, 100 * c / n))


if __name__ == "__main__":
  main()
