"""The readout: runs scored against the truth of their built population.

Usage (from the repository root, any Python 3.9+, standard library only):

  python analysis/readout.py gss22-jev gss22-cot-4o
  python analysis/readout.py gss22-c-jev gss22-c-cot-gemma --out /tmp/readout.md

A name is read from runs/<name>/ when it is registered (the manifest, the gzipped
predictions and trace), else from runs-local/<name>/ (run.json and the plain files), and is
then marked **unregistered**: a number from it cites no run. A folder path works
too. Every run must be over the same built population, found in LAB_DATA (the environment,
else a .env here, else data/ in this repository) by the run's `data.population_path`; `items.json` and `truth.jsonl` must
still have the sha256 the run recorded, or the readout refuses.

What it prints, in Markdown, split by agent type (demographic, survey):

- **Raw accuracy with the floor row**, over each run's expected records: its
  agents' own items, the first `items.per_agent` of each. A record is right when its pick
  is the respondent's answer. The rest are wrong, and three kinds of wrong have their own
  column: `unanswered` (never asked, a failed request, or a reply with no answer for the
  item), `cut by the cap` (no answer, and the batch's reply reached the paper's token cap),
  and `off-option` (an answer that is not one of the options).
- **Accuracy by item family**, the families of the recode table, with the floor beside.
- **The crowd table**: the share of a run's answers that are the item's
  majority answer (the floor's option), beside the respondents' own share over the same
  records, and the excess in points. A typed run adds a `drawn` row: the mean probability
  it gives the majority answer. A run is read over its answered records; the rest are left
  out and counted beside. These are rates of a crowd, never accuracy.
- **The middle rate**: the share of answers that are an ordinal item's middle option
  (`MIDDLE`), with the respondents' share over the same records, and the `drawn` row.
- **For typed runs:** log-loss and Brier score of the distribution against the truth, and
  a calibration table in five bins of the pick's probability.
- **Pairs:** for every two runs, agreement and paired accuracy over their common agent x
  item records only, with the floor over the same records.
- **Intervals**, only with `--against <run>`: every other run of that run's
  group against it, over the same records. The paired raw accuracy difference, with the
  floor as a row, and the crowd's D = |excess| - |excess of the run named|, each with a 95%
  interval over 10,000 resamples of the respondents, seed 20260926. It is never printed on
  a pilot run.
- **Replays**: for every two typed runs of one condition (the same predictor
  and variant, their models shown), over their common records with the same `state_sha1`,
  the share of identical distributions, the share with the same pick, and the total
  variation distance between the two distributions, its mean and 95th percentile.
- **The partisan divide**, only with `--partisans`: whether the gap between
  Democrats and Republicans on each item's majority answer is as wide in a run as in the
  respondents, as r and k with intervals over respondents. It reads each respondent's
  party from the demographic scratch in LAB_DATA, and needs numpy.
- **Cost:** every `call` event of the trace priced at the run's own `config.prices` (list
  price), per respondent and per agent type.

  python analysis/readout.py gss22-jev gss22-cot-4o --against gss22-cot-4o
  python analysis/readout.py gss22-b-jev gss22-b-jevsteps --against gss22-b-jev --partisans

**A first N**, with `--first N`: every run is read over the records of its
population's first N respondents in draw order, and a run that does not cover all N is
refused. Under `--first`, runs over two or more built populations are read together when
their first N respondents are the same: the same ids in the same order, and for each of
them the same items, the same truth row (every field but the floor) and the same agent
files (`scratch.json` and `memory_stream/nodes.json`; the embeddings are not compared, since
the embedding route is not bitwise deterministic, and the pairs ask for the same
`state_sha1` anyway). The readout checks this from the files in LAB_DATA and refuses
otherwise, naming the first difference. Such a **head read** prints each population's floor,
labelled with its name, wherever the floor is printed. The majority answer the crowd, the
middle, the partisan tables and the crowd's D read against is the first run's population's
floor, and the report says so. Without `--first`, two populations are refused.

  python analysis/readout.py gss22-b-jev gss22-b-cot-gemma --against gss22-b-cot-gemma --first 300
  python analysis/readout.py gss22-cot-gemma gss22-cot-4o gss22-b-cot-gemma --first 100

**Whole batteries held out**. A population whose build wrote
`universe.json` has its trap records (a held-out item whose answer belongs to a
respondent the item's "Inapplicable" option describes) left out of every table, on every
run read with it, `--first` included; the accuracy table counts them in their own column.
Its `withhold.json` is the runner's; the head read accepts a population that adds
these two files to another's head, and the report names them. `--cross A:TYPE:B:TYPE`
reads the difference in raw accuracy between two sides whose states differ by design (a
survey agent against a demographic agent, or a withholding population against the plain
one), over the same respondents and items, with the bootstrap interval; `--family` reads it over
`single`, `battery` or one family only.

  python analysis/readout.py gss22-c-jev gss22-c-cot-gemma --cross gss22-c-jev:survey:gss22-c-jev:demographic
  python analysis/readout.py gss22-c-jev gss22-b-jev --first 1052 --cross gss22-c-jev:survey:gss22-b-jev:survey --family battery

**Every answer in memory**. A population built with `--every-answer`
holds every answer in its survey agents' memory, with one map per level (`withhold.json`
for every answer, `withhold-k10.json`, `withhold-k30.json`, listed under `maps` in its
`population.json`). The head read accepts it beside the plain population it was built from:
its survey agents' `nodes.json` may hold more nodes, provided the plain one's are among
them in order, and the report says so. A run may ask one agent type only (`--agents` in
the runner); under `--first` it is read over that type's agents of the head, and where the
other type's tables would be the report says the run has none. `--contrast
A:TYPE:B:TYPE:C:TYPE:D:TYPE` reads (A - B) - (C - D) in points, each pair over the same
records and both pairs over the same respondents, with the bootstrap interval.

  python analysis/readout.py gss22-c-cot-gemma gss22-c-direct-gemma --first 600 \
    --contrast gss22-c-cot-gemma:demographic:gss22-c-direct-gemma:demographic:gss22-c-cot-gemma:survey:gss22-c-direct-gemma:survey

Runs whose expected records are the same (same respondents, same items) are read together,
each group with its own floor. Every accuracy is raw accuracy: public respondents have no
retest, so the paper's own metric does not exist here, and this module never
prints its name. It never calls the bundle's scorer.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict

LAB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(LAB, "runs")
RUNS_LOCAL = os.path.join(LAB, "runs-local")
LAB_DATA_TAG = "<lab-data>/"
# The agent files a head read finds the same in every population.
AGENT_FILES = ("scratch.json", os.path.join("memory_stream", "nodes.json"))
# The files a withholding build adds to a plain one, which a head read accepts
# and names; `universe.json` also takes its trap records out of every table.
EXTRA_FILES = ("withhold.json", "universe.json")
UNIVERSE = "universe.json"
# An every-answer build's level maps beyond withhold.json, checked like it.
MAP_PREFIX, MAP_SUFFIX = "withhold-", ".json"
SINGLE, BATTERY = "single", "battery"  # the cross reading's two families of families
AGENT_TYPES = ("demographic", "survey")
RIGHT, WRONG, UNANSWERED, CUT, OFF = "right", "wrong", "unanswered", "cut", "off_option"
FLOOR = "floor"
# Jev's probabilities come on a grid of 0.01, and 0.00 is common. A 0.00 stands for
# something under 0.005, so log-loss takes a truth's probability as at least that.
EPS = 0.005
BINS = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0))
# The intervals: the respondent is the unit the draw sampled, so a resample
# draws respondents with replacement and carries each one's records whole.
SEED = 20260926
RESAMPLES = 10000
# The pilot's verdicts stand, and an interval on its runs would be a second reading
# of them, so the readout refuses to print one.
PILOT = frozenset(("gss22-jev-p", "gss22-jevsteps-p", "gss22-cot-mini-p", "gss22-cot-mini-p2",
                   "gss22-cot-4o-p"))
# The partisans: PARTYID's three Democratic codes and its three Republican
# codes, as the builder spells them in `party_identification`. A pure independent, another
# party or a blank is neither. An item is read with at least MIN_PARTY records of each.
DEMOCRATS = frozenset(("strong democrat", "not very strong democrat",
                       "independent, close to democrat"))
REPUBLICANS = frozenset(("strong republican", "not very strong republican",
                         "independent, close to republican"))
MIN_PARTY = 20

# An ordinal item's middle option: the centre of an ordered scale with an odd number
# of options, once an "Inapplicable" option (and RACWORK's "Works alone") is set aside. A
# nominal item has no middle (VOTE16, IF16WHO, JEW, REBORN, FUCITZN, PARTFULL, ...), nor
# does a scale with an even number of options, nor a count (GRANBORN: "Two" is not a
# hedge). The scratch items (INCOM16, POLVIEWS) are never held out, so they are not here.
_ABOUT_RIGHT = ["NATSPAC/Y", "NATENVIR/Y", "NATHEAL/Y", "NATCITY/Y", "NATDRUG/Y",
                "NATEDUC/Y", "NATRACE/Y", "NATARMS/Y", "NATAID/Y", "NATFARE/Y", "NATROAD",
                "NATSOC", "NATMASS", "NATPARK", "NATCHLD", "NATSCI", "NATENRGY", "COURTS",
                "TAX"]
_ONLY_SOME = ["CONARMY", "CONBUS", "CONCLERG", "CONEDUC", "CONFED", "CONFINAN", "CONJUDGE",
              "CONLABOR", "CONLEGIS", "CONMEDIC", "CONPRESS", "CONSCI", "CONTV"]
MIDDLE = dict(
    [("gss:" + i, "About right") for i in _ABOUT_RIGHT]
    + [("gss:" + i, "Only some") for i in _ONLY_SOME]
    + [("gss:" + i, "4") for i in ("WLTHWHTS", "WLTHBLKS", "WLTHHSPS")]
    + [("gss:" + i, "Depends") for i in ("AGED", "FAIR", "HELPFUL", "TRUST")]
    + [("gss:FEHIRE", "Neither agree nor disagree"),
       ("gss:MARHOMO", "Neither agree nor disagree"),
       ("gss:FINRELA", "Average"),
       ("gss:HAPPY", "Pretty happy"),
       ("gss:SATFIN", "More or less"),
       ("gss:FINALTER", "Stayed same"),
       ("gss:RACWORK", "About half and half"),
       ("gss:LETIN1A", "Remain the same as it is"),
       ("gss:GETAHEAD", "Hard work, luck equally important"),
       ("gss:PARSOL", "About the same"),
       ("gss:DIVLAW", "Stay as is"),
       ("gss:DISCAFF", "Somewhat likely"),
       ("gss:PORNLAW", "There should be laws against the distribution of pornography to "
                       "persons under 18"),
       ("gss:NEWS", "Once a week"),
       ("gss:ATTEND", "Once a month"),
       ("gss:BIBLE", "Inspired word"),
       ("gss:LIFE", "Routine")])


# --- reading ----------------------------------------------------------------

def read_json(path):
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)


def read_jsonl(path):
  opener = gzip.open if path.endswith(".gz") else open
  out = []
  with opener(path, "rt", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        out.append(json.loads(line))
      except ValueError:
        continue  # a run killed mid-write leaves one truncated line
  return out


def sha256_file(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def lab_data():
  """LAB_DATA from the environment, else from .env, else data/ in this repository."""
  if os.environ.get("LAB_DATA"):
    return os.path.expanduser(os.environ["LAB_DATA"])
  env = os.path.join(LAB, ".env")
  if os.path.isfile(env):
    with open(env) as f:
      for line in f:
        line = line.strip()
        if line.startswith("export "):
          line = line[len("export "):]
        key, _, value = line.partition("=")
        if key.strip() == "LAB_DATA":
          return os.path.expanduser(value.strip().strip('"').strip("'"))
  return os.path.join(LAB, "data")


class Run:
  """One run: its head (the manifest, or run.json when unregistered), its records by
  (agent, item), and the batches whose chat reply reached the token cap."""

  def __init__(self, name, status, head, records, events):
    self.name, self.status, self.head, self.events = name, status, head, events
    self.records = {(r.get("agent"), r.get("item")): r for r in records}
    condition = head.get("condition") or {}
    self.predictor = condition.get("predictor")
    self.variant = condition.get("variant")
    self.model = condition.get("model")
    self.typed = self.predictor == "jev"
    cap = (condition.get("request") or {}).get("max_tokens")
    self.cut = {(e.get("agent"), e.get("batch")) for e in events
                if cap and e.get("event") == "call" and e.get("kind") == "chat"
                and (e.get("tokens_out") or 0) >= cap}

  @property
  def arm(self):
    arm = "-".join(v for v in (self.predictor, self.variant) if v)
    return arm if self.typed else "%s on %s" % (arm, (self.model or "?").split("/")[-1])

  @property
  def data(self):
    return self.head.get("data") or {}


def load_run(name, runs=RUNS, local=RUNS_LOCAL):
  """A registered run by name, else an unregistered one from runs-local/, else a folder."""
  for folder, registered in ((os.path.join(runs, name), True),
                             (os.path.join(local, name), False),
                             (name, None)):
    manifest = os.path.join(folder, "manifest.json")
    run_json = os.path.join(folder, "run.json")
    if registered is not False and os.path.isfile(manifest):
      head = read_json(manifest)
      records = read_jsonl(os.path.join(folder, "predictions.jsonl.gz"))
      trace = os.path.join(folder, "trace.jsonl.gz")
      status = head.get("status")
    elif registered is not True and os.path.isfile(run_json):
      head = read_json(run_json)
      records = read_jsonl(os.path.join(folder, "predictions.jsonl"))
      trace = os.path.join(folder, "trace.jsonl")
      status = "unregistered"
    else:
      continue
    events = read_jsonl(trace) if os.path.isfile(trace) else []
    return Run(head.get("name") or os.path.basename(folder.rstrip("/")), status, head,
               records, events)
  raise ValueError("no run %r: neither runs/%s/manifest.json nor runs-local/%s/run.json"
                   % (name, name, name))


class Population:
  """A built population's items, truth and agents, read from its folder in LAB_DATA."""

  def __init__(self, folder, path=None):
    self.folder, self.path = folder, path
    self.meta = read_json(os.path.join(folder, "population.json"))
    self.name = self.meta.get("name") or os.path.basename(folder.rstrip("/"))
    self.items = read_json(os.path.join(folder, "items.json"))
    self.truth = {(str(row["respondent"]), row["item"]): row
                  for row in read_jsonl(os.path.join(folder, "truth.jsonl"))}
    self.agents = {}
    for respondent in self.meta["respondents"]:
      for kind, agent in respondent["agents"].items():
        self.agents[agent] = (str(respondent["id"]), kind)
    self.sha256 = {name: sha256_file(os.path.join(folder, name))
                   for name in ("items.json", "truth.jsonl", "population.json")}
    self.parties = None
    self.extra = [name for name in EXTRA_FILES if os.path.isfile(os.path.join(folder, name))]
    self.extra += sorted(name for name in (self.meta.get("files") or {})
                         if name.startswith(MAP_PREFIX) and name.endswith(MAP_SUFFIX))
    self.every = (self.meta.get("every") or {}).get("plain")  # the plain population an every-answer build extends
    for name in self.extra:
      recorded = (self.meta.get("files") or {}).get(name)
      if recorded != sha256_file(os.path.join(folder, name)):
        raise ValueError("%s/%s is not the file its build recorded (sha256 %s)"
                         % (self.name, name, str(recorded)[:12]))
    self.universe = set()
    if UNIVERSE in self.extra:
      self.universe = {(str(r["respondent"]), r["item"])
                       for r in read_json(os.path.join(folder, UNIVERSE))["records"]}

  def outside(self, key):
    """Whether the record is a trap record, outside its item's universe."""
    return (self.agents[key[0]][0], key[1]) in self.universe

  def party(self, respondent):
    """"D", "R" or None: the respondent's `party_identification`, read from its
    demographic agent's scratch, the one place the builder writes it."""
    if self.parties is None:
      self.parties = {}
      for r in self.meta["respondents"]:
        path = os.path.join(self.folder, "agents", r["agents"]["demographic"], "scratch.json")
        said = (read_json(path).get("party_identification") or "").strip().lower()
        self.parties[str(r["id"])] = ("D" if said in DEMOCRATS
                                      else "R" if said in REPUBLICANS else None)
    return self.parties[respondent]

  def row(self, key):
    return self.truth[(self.agents[key[0]][0], key[1])]

  def kind(self, agent):
    return self.agents[agent][1]

  @property
  def pops(self):
    return [self]

  @property
  def floors(self):
    """The floors a table prints, (label, population): one here, one per population in a
    head read."""
    return [(FLOOR, self)]

  def head(self, first):
    """The agents of the first `first` respondents, in draw order."""
    respondents = self.meta["respondents"]
    if first > len(respondents):
      raise ValueError("--first %d: %s has %d respondents" % (first, self.name, len(respondents)))
    return [agent for r in respondents[:first] for _, agent in sorted(r["agents"].items())]


class Head:
  """Built populations read together over their first `first` respondents, which
  `head_difference` found the same in each. The items, the truth and the
  agents' respondents are the first population's, which are every other's on the head;
  each population keeps its own floor, printed labelled."""

  def __init__(self, pops, first):
    self.pops, self.first, self.lead = pops, first, pops[0]
    self.name = " and ".join(pop.name for pop in pops)
    self.items, self.truth, self.sha256 = self.lead.items, self.lead.truth, self.lead.sha256
    self.agents = {}
    for pop in reversed(pops):  # every population's agents, for the cost of the whole run
      self.agents.update(pop.agents)
    # a trap record of one population is one of every other on the head
    self.universe = set().union(*(pop.universe for pop in pops))

  def outside(self, key):
    return (self.agents[key[0]][0], key[1]) in self.universe

  def row(self, key):
    return self.lead.row(key)

  def kind(self, agent):
    return self.agents[agent][1]

  def party(self, respondent):
    return self.lead.party(respondent)

  @property
  def floors(self):
    return [("%s, %s" % (FLOOR, pop.name), pop) for pop in self.pops]

  def head(self, first):
    return self.lead.head(first)


def truth_rows(pop):
  """Each respondent's truth rows by item, every field but the floor, which is the
  population's own."""
  out = defaultdict(dict)
  for (respondent, item), row in pop.truth.items():
    out[respondent][item] = {k: v for k, v in row.items() if k != FLOOR}
  return out


def head_difference(a, b, first):
  """The first way `b`'s first `first` respondents differ from `a`'s, or None: an id or its
  place, an agent, an item list, a truth row, an agent file."""
  ra, rb = a.meta["respondents"][:first], b.meta["respondents"][:first]
  for place, (x, y) in enumerate(zip(ra, rb), 1):
    if str(x["id"]) != str(y["id"]):
      return "respondent %d is %s in %s and %s in %s" % (place, x["id"], a.name, y["id"], b.name)
  rows_a, rows_b = truth_rows(a), truth_rows(b)
  for x, y in zip(ra, rb):
    rid = str(x["id"])
    if x["agents"] != y["agents"]:
      return "respondent %s's agents are %s in %s and %s in %s" % (
          rid, x["agents"], a.name, y["agents"], b.name)
    for kind, agent in sorted(x["agents"].items()):
      if a.items.get(agent) != b.items.get(agent):
        return "agent %s's items differ between %s and %s" % (agent, a.name, b.name)
      for name in AGENT_FILES:
        files = [os.path.join(pop.folder, "agents", agent, name) for pop in (a, b)]
        hashes = [sha256_file(f) if os.path.isfile(f) else None for f in files]
        if hashes[0] != hashes[1] and not more_answers(a, b, kind, name, files):
          return "agent %s's %s differs between %s and %s" % (agent, name, a.name, b.name)
    if rows_a.get(rid) != rows_b.get(rid):
      items = sorted(set(rows_a.get(rid, {})) | set(rows_b.get(rid, {})))
      item = next(i for i in items if rows_a.get(rid, {}).get(i) != rows_b.get(rid, {}).get(i))
      return "respondent %s's truth row for %s differs between %s and %s" % (
          rid, item, a.name, b.name)
  return None


def more_answers(a, b, kind, name, files):
  """Whether two survey agents' nodes differ only as an every-answer build's differ from the
  plain build's: one of the two populations is such a build and the other's
  nodes are among its own, by content and in order."""
  if kind != "survey" or name != AGENT_FILES[1] or bool(a.every) == bool(b.every):
    return False
  if not all(os.path.isfile(f) for f in files):
    return False
  every, plain = (files[0], files[1]) if a.every else (files[1], files[0])
  mine = iter(node["content"] for node in read_json(every))
  return all(node["content"] in mine for node in read_json(plain))


def population_of(runs, root, first=None):
  """The built population every run is over, its files checked against each run; with
  `first`, the populations whose first `first` respondents are the same, as a Head."""
  by_path = {}
  for run in runs:
    by_path.setdefault(run.data.get("population_path"), []).append(run)
  if len(by_path) != 1 and first is None:
    raise ValueError("the runs are over different populations: %s" % sorted(map(str, by_path)))
  pops = []
  for path, members in by_path.items():
    path = path or ""
    if not path.startswith(LAB_DATA_TAG):
      raise ValueError("%s is not over a built population (%s)"
                       % (members[0].name, members[0].data.get("population")))
    pop = Population(os.path.join(root, path[len(LAB_DATA_TAG):]), path)
    # population.json maps each agent to its respondent, so it decides whose truth a record
    # meets, as much as the truth file does
    for run in members:
      for name, key in (("items.json", "items_sha256"), ("truth.jsonl", "truth_sha256"),
                        ("population.json", "population_json_sha256")):
        if pop.sha256[name] != run.data.get(key):
          raise ValueError("%s: %s has sha256 %s, but the run recorded %s; the truth it "
                           "would be scored against is not the one it ran on"
                           % (run.name, name, pop.sha256[name][:12],
                              str(run.data.get(key))[:12]))
    if first is not None:
      pop.head(first)  # refuses a population shorter than the head
    pops.append(pop)
  if len(pops) == 1:
    return pops[0]
  for other in pops[1:]:
    difference = head_difference(pops[0], other, first)
    if difference:
      raise ValueError("%s and %s do not share their first %d respondents: %s"
                       % (pops[0].name, other.name, first, difference))
  return Head(pops, first)


def expected(run, pop, first=None):
  """The run's agent x item records: each agent's own items, the first `per_agent`; with
  `first`, only the agents of the population's first `first` respondents, all of which the
  run must cover."""
  per = (run.data.get("items") or {}).get("per_agent")
  if not isinstance(per, int):
    raise ValueError("%s: items.per_agent is %r, not one number" % (run.name, per))
  agents = run.data["agent_ids"]
  if first is not None:
    covered, agents = set(agents), [a for a in pop.head(first) if pop.kind(a) in run_types(run)]
    missing = [agent for agent in agents if agent not in covered]
    if missing:
      raise ValueError("%s does not cover the first %d respondents: %d of their %d agents "
                       "are not in it, the first %s" % (run.name, first, len(missing),
                                                        len(agents), missing[0]))
  keys = [(agent, item) for agent in agents for item in pop.items[agent][:per]]
  return [key for key in keys if not pop.outside(key)] if pop.universe else keys


def run_types(run):
  """The agent types a run asked: both unless the runner was told one."""
  said = ((run.head.get("condition") or {}).get("agent_type") or "").split("+")
  types = [t for t in said if t in AGENT_TYPES]
  return types or list(AGENT_TYPES)


def outside_keys(run, pop, first=None):
  """The trap records the run was expected to answer, which `expected` leaves out."""
  if not pop.universe:
    return []
  per = run.data["items"]["per_agent"]
  agents = ([a for a in pop.head(first) if pop.kind(a) in run_types(run)]
            if first is not None else run.data["agent_ids"])
  return [(agent, item) for agent in agents for item in pop.items[agent][:per]
          if pop.outside((agent, item))]


# --- one record -------------------------------------------------------------

def offered(record):
  """Whether the record's answer is one it could give: one of the options, or for a
  numerical item a number inside its range, `options` being [low, high]."""
  predicted, options = record.get("predicted"), record.get("options") or []
  if record.get("kind") == "numerical":
    number = isinstance(predicted, (int, float)) and not isinstance(predicted, bool)
    return number and len(options) == 2 and options[0] <= predicted <= options[1]
  return predicted in options


def outcome(run, key, truth):
  record = run.records.get(key)
  if record is None or not record.get("ok", True):
    return UNANSWERED
  predicted = record.get("predicted")
  if predicted is None:
    return CUT if (key[0], record.get("batch")) in run.cut else UNANSWERED
  if not offered(record):
    return OFF
  return RIGHT if predicted == truth else WRONG


def answer(run, key):
  """The run's pick when it is one it could give, else None."""
  record = run.records.get(key)
  if record is None or not record.get("ok", True):
    return None
  return record.get("predicted") if offered(record) else None


def distribution(run, key):
  """The typed run's `{option: probability}` for an answered item, else None."""
  record = run.records.get(key)
  if not run.typed or answer(run, key) is None or record.get("primitive") != "choice":
    return None
  return record.get("p")


def middle(key, run=None):
  item = key[1]
  value = MIDDLE.get(item)
  if value is not None and run is not None:
    record = run.records.get(key)
    if record is not None and value not in (record.get("options") or []):
      raise ValueError("MIDDLE[%s] = %r is not one of its options" % (item, value))
  return value


def ratio(num, den):
  return num / den if den else None


def mean(values):
  values = list(values)
  return sum(values) / len(values) if values else None


# --- tables -----------------------------------------------------------------
# A table is {title, columns: [(key, header, format)], rows: [dict], floor, note}. `floor`
# says where the floor sits: "row" (a row labelled floor), "column" (a `floor` value on
# every row), or None for a table that is not one of accuracy.

def table(title, columns, rows, floor=None, note=None):
  return {"title": title, "columns": columns, "rows": rows, "floor": floor, "note": note}


def accuracy_table(runs, keys, pop, outside=None):
  """`outside`, when the population has a universe: the trap records the group's keys left
  out, counted in their own column."""
  rows = []
  for run in runs:
    tally = Counter(outcome(run, key, pop.row(key)["truth"]) for key in keys)
    rows.append({"label": run.name, "arm": run.arm, "n": len(keys), "right": tally[RIGHT],
                 "accuracy": ratio(tally[RIGHT], len(keys)), "unanswered": tally[UNANSWERED],
                 "cut": tally[CUT], "off_option": tally[OFF], "outside": outside})
  for label, floor_pop in pop.floors:
    right = floor_count(keys, floor_pop)
    rows.append({"label": label, "arm": "the majority answer, out of sample", "n": len(keys),
                 "right": right, "accuracy": ratio(right, len(keys)), "unanswered": 0,
                 "cut": 0, "off_option": 0, "outside": outside})
  columns = [("label", "run", "s"), ("arm", "arm", "s"), ("n", "records", "d"),
             ("accuracy", "raw accuracy", "acc"), ("unanswered", "unanswered", "d"),
             ("cut", "cut by the cap", "d"), ("off_option", "off-option", "d")]
  note = "Every unanswered, cut or off-option record counts as wrong."
  if outside is not None:
    columns.append(("outside", "left out, outside the universe", "d"))
    note += (" A record outside its item's universe (the trap records of `universe.json`) "
             "is left out of every table and counted in the last column.")
  return table("Raw accuracy", columns, rows, floor="row", note=note)


def right_count(run, keys, pop):
  return sum(1 for key in keys if outcome(run, key, pop.row(key)["truth"]) == RIGHT)


def floor_count(keys, pop):
  return sum(1 for key in keys if pop.row(key)[FLOOR] == pop.row(key)["truth"])


def floor_cells(keys, pop):
  """Each floor's raw accuracy over `keys`, under its column's key: `floor`, or each
  population's labelled floor in a head read."""
  return {label: ratio(floor_count(keys, p), len(keys)) for label, p in pop.floors}


def floor_columns(pop, header=FLOOR):
  return [(label, header if len(pop.floors) == 1 else label, "acc") for label, _ in pop.floors]


def family_table(runs, keys, pop):
  families = defaultdict(list)
  for key in keys:
    families[pop.row(key)["family"]].append(key)
  rows = []
  for family, ks in sorted(families.items(), key=lambda kv: (-len(kv[1]), kv[0])):
    row = {"family": family, "n": len(ks), **floor_cells(ks, pop)}
    for run in runs:
      row[run.name] = ratio(right_count(run, ks, pop), len(ks))
    rows.append(row)
  columns = ([("family", "family", "s"), ("n", "records", "d")]
             + [(run.name, run.name, "acc") for run in runs] + floor_columns(pop))
  return table("Raw accuracy by item family", columns, rows, floor="column")


def crowd_table(runs, keys, pop):
  rows = []
  for run in runs:
    answered = [key for key in keys if answer(run, key) is not None]
    respondents = mean(pop.row(k)["truth"] == pop.row(k)[FLOOR] for k in answered)
    rate = mean(answer(run, k) == pop.row(k)[FLOOR] for k in answered)
    rows.append(crowd_row(run.name, run.arm, answered, keys, rate, respondents))
    if run.typed:
      drawn = mean(distribution(run, k).get(pop.row(k)[FLOOR], 0.0) for k in answered)
      rows.append(crowd_row(run.name + ", drawn", "each answer drawn from its distribution",
                            answered, keys, drawn, respondents))
  respondents = mean(pop.row(k)["truth"] == pop.row(k)[FLOOR] for k in keys)
  rows.append(crowd_row(FLOOR, "always the majority answer", keys, keys,
                        1.0 if keys else None, respondents))
  return table("The crowd: how often the majority answer is given",
               [("label", "run", "s"), ("arm", "arm", "s"), ("records", "records", "d"),
                ("left_out", "left out", "d"), ("rate", "majority rate", "pct"),
                ("respondents", "respondents' rate", "pct"), ("excess", "excess", "pts")],
               rows,
               note="Over each run's answered records; a record with no answer among the "
                    "options is left out and counted beside. The respondents' rate is "
                    "over the same records. These are rates of a crowd, not accuracy.")


def crowd_row(label, arm, answered, keys, rate, respondents):
  excess = (rate - respondents) * 100 if None not in (rate, respondents) else None
  return {"label": label, "arm": arm, "records": len(answered),
          "left_out": len(keys) - len(answered), "rate": rate, "respondents": respondents,
          "excess": excess}


def middle_table(runs, keys, pop):
  keys = [key for key in keys if middle(key) is not None]
  rows = []
  for run in runs:
    answered = [key for key in keys if answer(run, key) is not None]
    respondents = mean(pop.row(k)["truth"] == middle(k, run) for k in answered)
    rate = mean(answer(run, k) == middle(k, run) for k in answered)
    rows.append(crowd_row(run.name, run.arm, answered, keys, rate, respondents))
    if run.typed:
      drawn = mean(distribution(run, k).get(middle(k), 0.0) for k in answered)
      rows.append(crowd_row(run.name + ", drawn", "each answer drawn from its distribution",
                            answered, keys, drawn, respondents))
  respondents = mean(pop.row(k)["truth"] == middle(k) for k in keys)
  rate = mean(pop.row(k)[FLOOR] == middle(k) for k in keys)
  rows.append(crowd_row(FLOOR, "the majority answer", keys, keys, rate, respondents))
  return table("The middle: how often an ordinal item's middle option is given",
               [("label", "run", "s"), ("arm", "arm", "s"), ("records", "records", "d"),
                ("left_out", "left out", "d"), ("rate", "middle rate", "pct"),
                ("respondents", "respondents' rate", "pct"), ("excess", "excess", "pts")],
               rows,
               note="Over the items with a middle option (`MIDDLE` in analysis/readout.py) "
                    "and each run's answered records; the respondents' rate is over the "
                    "same records.")


def scores_table(runs, keys, pop):
  rows = []
  for run in runs:
    if not run.typed:
      continue
    losses, briers, clipped = [], [], 0
    for key in keys:
      p = distribution(run, key)
      if p is None:
        continue
      truth = pop.row(key)["truth"]
      q = p.get(truth) or 0.0
      clipped += q < EPS
      losses.append(-math.log(max(q, EPS)))
      options = run.records[key]["options"]
      briers.append(sum(((p.get(o) or 0.0) - (o == truth)) ** 2 for o in options))
    rows.append({"label": run.name, "arm": run.arm, "records": len(losses),
                 "log_loss": mean(losses), "brier": mean(briers), "clipped": clipped})
  if not rows:
    return None
  return table("The typed distribution against the truth",
               [("label", "run", "s"), ("arm", "arm", "s"), ("records", "records", "d"),
                ("log_loss", "log-loss", "f3"), ("brier", "Brier score", "f3"),
                ("clipped", "truth under 0.005", "d")],
               rows,
               note="Log-loss is the mean of -ln p(truth), with p taken as at least 0.005, "
                    "since Jev's probabilities come in steps of 0.01. The Brier score sums "
                    "(p - hit)^2 over the options, 0 to 2. Over the answered records.")


def calibration_table(run, keys, pop):
  bins = [[] for _ in BINS]
  for key in keys:
    p = distribution(run, key)
    if p is None:
      continue
    q = round(p.get(answer(run, key)) or 0.0, 6)  # 0.39999999999999997 is 0.40
    index = next(i for i, (low, high) in enumerate(BINS) if q < high or high == BINS[-1][1])
    bins[index].append((key, q))
  rows = []
  for (low, high), members in zip(BINS, bins):
    ks = [k for k, _ in members]
    rows.append({"bin": "%.1f-%.1f" % (low, high), "n": len(ks),
                 "mean_p": mean(q for _, q in members),
                 "hit": ratio(right_count(run, ks, pop), len(ks)), **floor_cells(ks, pop)})
  return table("Calibration of %s: the pick's probability against its hit rate" % run.name,
               [("bin", "pick's probability", "s"), ("n", "records", "d"),
                ("mean_p", "mean probability", "f3"), ("hit", "hit rate", "acc")]
               + floor_columns(pop, "floor, same records"),
               rows, floor="column")


def same_state(runs, key):
  """Whether every run that answered `key` answered it over one state (the condition for
  a pair). A run with no answered record for it is not a second state: it counts as
  unanswered, as the replay table counts it."""
  states = {run.records[key].get("state_sha1") for run in runs
            if key in run.records and run.records[key].get("ok", True)}
  return len(states) <= 1


def pairs_table(runs, pop, keys_by_run, agent_type):
  rows = []
  for a, b in itertools.combinations(runs, 2):
    if agent_type not in run_types(a) or agent_type not in run_types(b):
      continue  # one of the two asked no agent of this type
    theirs = set(keys_by_run[b.name])
    common = [k for k in keys_by_run[a.name] if k in theirs and pop.kind(k[0]) == agent_type]
    other_state = sum(1 for k in common if not same_state((a, b), k))
    common = [k for k in common if same_state((a, b), k)]
    right_a = {k for k in common if outcome(a, k, pop.row(k)["truth"]) == RIGHT}
    right_b = {k for k in common if outcome(b, k, pop.row(k)["truth"]) == RIGHT}
    agree = sum(1 for k in common if answer(a, k) is not None and answer(a, k) == answer(b, k))
    rows.append({"a": a.name, "b": b.name, "n": len(common), "agreement": ratio(agree, len(common)),
                 "acc_a": ratio(len(right_a), len(common)),
                 "acc_b": ratio(len(right_b), len(common)), **floor_cells(common, pop),
                 "only_a": len(right_a - right_b), "only_b": len(right_b - right_a),
                 "other_state": other_state})
  if not rows:
    return None
  return table("Pairs, over their common records",
               [("a", "run A", "s"), ("b", "run B", "s"), ("n", "common records", "d"),
                ("other_state", "left out, another state", "d"),
                ("agreement", "agreement", "pct"), ("acc_a", "raw accuracy A", "acc"),
                ("acc_b", "raw accuracy B", "acc")] + floor_columns(pop)
               + [("only_a", "only A right", "d"), ("only_b", "only B right", "d")],
               rows, floor="column",
               note="Agreement is the same option from both; an unanswered or off-option "
                    "record agrees with nothing. A record the two runs answered over "
                    "different states is not a pair, and is left out of both.")


def cost_table(runs, pop):
  rows = []
  for run in runs:
    prices = (run.head.get("config") or {}).get("prices")
    respondents = run.data.get("respondents")
    row = {"label": run.name, "arm": run.arm, "respondents": respondents}
    if prices:
      spent = Counter()
      for event in run.events:
        if event.get("event") != "call":
          continue
        price = prices.get(event.get("kind"))
        if price is None:
          continue
        cost = ((event.get("tokens_in") or 0) * price["in"]
                + (event.get("tokens_out") or 0) * price["out"]) / 1e6
        agent = event.get("agent")
        spent[pop.kind(agent) if agent in pop.agents else "gate"] += cost
      total = sum(spent.values())
      row.update(total=total, gate=spent["gate"],
                 per_respondent=ratio(total - spent["gate"], respondents),
                 demographic=ratio(spent["demographic"], respondents),
                 survey=ratio(spent["survey"], respondents))
    rows.append(row)
  return table("Cost at list price",
               [("label", "run", "s"), ("arm", "arm", "s"), ("respondents", "respondents", "d"),
                ("total", "whole run", "usd"), ("gate", "of which the gate", "usd"),
                ("per_respondent", "per respondent", "usd"),
                ("demographic", "demographic agent", "usd"), ("survey", "survey agent", "usd")],
               rows,
               note="Every `call` event of the trace, priced at the run's own "
                    "`config.prices`: list prices, not an invoice. Per respondent leaves "
                    "the health gate out.")


# --- intervals ------------------------------------------------

def percentile(values, q):
  """The q-th percentile of `values`, interpolated linearly between the closest ranks."""
  values = sorted(values)
  position = (len(values) - 1) * q / 100
  low = int(position)
  high = min(low + 1, len(values) - 1)
  return values[low] + (values[high] - values[low]) * (position - low)


def bootstrap(columns, statistics, seed=SEED, resamples=RESAMPLES):
  """Each statistic with its 95% interval over resamples of the respondents.

  `columns` maps a name to one sum per respondent, every column in the same respondent
  order, and a statistic is a function of the columns' totals, None where it is undefined.
  A resample draws as many respondents as there are, with replacement, each carrying its
  sums whole. The interval is the 2.5th and 97.5th percentiles of the statistic over the
  resamples. Returns [(value, low, high)], with no interval where a resample leaves the
  statistic undefined."""
  names = list(columns)
  respondents = list(zip(*(columns[name] for name in names)))

  def totals(drawn):
    return dict(zip(names, (sum(column) for column in zip(*drawn))))

  rng = random.Random(seed)
  draws = [[] for _ in statistics]
  for _ in range(resamples):
    t = totals(rng.choices(respondents, k=len(respondents)))
    for values, statistic in zip(draws, statistics):
      values.append(statistic(t))
  whole = totals(respondents)
  out = []
  for values, statistic in zip(draws, statistics):
    value = statistic(whole)
    if value is None or None in values:
      out.append((value, None, None))
    else:
      out.append((value, percentile(values, 2.5), percentile(values, 97.5)))
  return out


def interval_tables(against, runs, keys, pop, seed=SEED, resamples=RESAMPLES):
  """Each of `runs` against `against`, over the same records `keys` of one agent type:
  the paired raw accuracy difference and the crowd's D, with intervals over respondents.
  A record some run answered over another state than `against`'s is left out of every run
."""
  other_state = sum(1 for k in keys if not same_state([against] + runs, k))
  keys = [k for k in keys if same_state([against] + runs, k)]
  order = {}
  for agent, _ in keys:
    order.setdefault(pop.agents[agent][0], len(order))
  columns = {}

  def column(name, ks, value):
    """One sum of value(key) per respondent over `ks`, evaluated now."""
    sums = [0.0] * len(order)
    for key in ks:
      sums[order[pop.agents[key[0]][0]]] += value(key)
    columns[name] = sums

  def truth(key):
    return pop.row(key)["truth"]

  # Accuracy: one right count per run and the floor, over every record.
  column("n", keys, lambda k: 1)
  floors = pop.floors
  for label, floor_pop in floors:
    column(("right", label), keys, lambda k: floor_pop.row(k)[FLOOR] == truth(k))
  for run in [against] + runs:
    column(("right", run.name), keys, lambda k: outcome(run, k, truth(k)) == RIGHT)

  def difference(name):
    return lambda t: 100 * (t[("right", name)] - t[("right", against.name)]) / t["n"]

  # The crowd and the middle: each run over its own answered records, as their tables read
  # it. `target` is the majority answer, or the item's middle option.
  families = [("majority", keys, lambda k, run: pop.row(k)[FLOOR]),
              ("middle", [k for k in keys if middle(k) is not None], middle)]
  families = [f for f in families if f[1]]
  family_keys = {family: ks for family, ks, _ in families}
  for run in [against] + runs:
    for family, ks, target in families:
      answered = [k for k in ks if answer(run, k) is not None]
      column(("answered", run.name, family), answered, lambda k: 1)
      column(("respondents", run.name, family), answered,
             lambda k: truth(k) == target(k, run))
      column(("picked", run.name, family), answered,
             lambda k: answer(run, k) == target(k, run))
      if run.typed:
        column(("drawn", run.name, family), answered,
               lambda k: distribution(run, k).get(target(k, run), 0.0))

  def excess(name, family, kind):
    def statistic(t):
      answered = t[("answered", name, family)]
      if not answered:
        return None
      return 100 * (t[(kind, name, family)] - t[("respondents", name, family)]) / answered
    return statistic

  def distance(name, family, kind):
    mine = excess(name, family, kind)
    theirs = excess(against.name, family, kind if against.typed else "picked")

    def statistic(t):
      a, b = mine(t), theirs(t)
      return None if None in (a, b) else abs(a) - abs(b)
    return statistic

  crowd = [(run, family, kind) for run in runs for family, _, _ in families
           for kind in (("picked", "drawn") if run.typed else ("picked",))]
  statistics = ([difference(run.name) for run in runs]
                + [difference(label) for label, _ in floors]
                + [distance(run.name, family, kind) for run, family, kind in crowd])
  results = bootstrap(columns, statistics, seed, resamples)
  whole = {name: sum(sums) for name, sums in columns.items()}

  n, of_against = len(keys), whole[("right", against.name)] / len(keys)
  rows = []
  for (label, arm), (value, low, high) in zip(
      [(run.name, run.arm) for run in runs]
      + [(label, "the majority answer, out of sample") for label, _ in floors],
      results):
    rows.append({"label": label, "arm": arm, "n": n, "respondents": len(order),
                 "other_state": other_state,
                 "accuracy": whole[("right", label)] / n, "against": of_against,
                 "difference": value, "interval": None if low is None else (low, high)})
  accuracy = table(
      "Raw accuracy against %s, paired, with intervals over respondents" % against.name,
      [("label", "run", "s"), ("arm", "arm", "s"), ("n", "records", "d"),
       ("other_state", "left out, another state", "d"),
       ("respondents", "respondents", "d"), ("accuracy", "raw accuracy", "acc"),
       ("against", "raw accuracy of %s" % against.name, "acc"),
       ("difference", "difference, points", "pts2"), ("interval", "95% interval", "ci")],
      rows, floor="row",
      note="Each run's raw accuracy minus %s's over the same records, in points. The 95%% "
           "interval is the 2.5th and 97.5th percentiles over %s resamples of the %d "
           "respondents, each carrying all of its records, seed %d."
           % (against.name, format(resamples, ","), len(order), seed))

  rows = []
  for (run, family, kind), (value, low, high) in zip(crowd, results[len(runs) + len(floors):]):
    answered = int(whole[("answered", run.name, family)])
    kind_against = kind if against.typed else "picked"
    rows.append({"label": run.name, "row": "%s, %s" % (family, kind), "records": answered,
                 "left_out": len(family_keys[family]) - answered,
                 "excess": excess(run.name, family, kind)(whole),
                 "against": excess(against.name, family, kind_against)(whole),
                 "difference": value, "interval": None if low is None else (low, high)})
  distances = table(
      "The crowd against %s: |excess| minus |excess of %s|, with intervals over respondents"
      % (against.name, against.name),
      [("label", "run", "s"), ("row", "row", "s"), ("records", "records", "d"),
       ("left_out", "left out", "d"), ("excess", "excess", "pts"),
       ("against", "excess of %s" % against.name, "pts"),
       ("difference", "D, points", "pts2"), ("interval", "95% interval", "ci")],
      rows,
      note="D is how much farther the run's rate is from the respondents' own rate than "
           "%s's: |excess| - |excess of %s|, in points. The majority rows are the crowd "
           "table's, the middle rows the middle table's, each run read over its own "
           "answered records. A drawn row is set against %s's drawn row when it has one, "
           "else its picked row. The interval is the accuracy table's, over the same "
           "resamples. These are rates of a crowd, not accuracy."
           % (against.name, against.name, against.name))
  return [accuracy, distances]


# --- replays --------------------------------------------------

def spread(p, q, options):
  """The total variation distance between two distributions over `options`, half the sum of
  |p - q|, and whether they are identical to Jev's step of 0.01 on every option."""
  options = list(options) + sorted((set(p) | set(q)) - set(options))
  a = [p.get(o) or 0.0 for o in options]
  b = [q.get(o) or 0.0 for o in options]
  identical = all(round(100 * x) == round(100 * y) for x, y in zip(a, b))
  return sum(abs(x - y) for x, y in zip(a, b)) / 2, identical


def replay_table(runs, pop, keys_by_run, agent_type):
  """Every two typed runs of one condition, record by record, over their common records
  with the same state: does the typed backend return the same distribution twice?"""
  rows = []
  for a, b in itertools.combinations(runs, 2):
    if not (a.typed and b.typed and (a.predictor, a.variant) == (b.predictor, b.variant)):
      continue
    theirs = set(keys_by_run[b.name])
    common = [k for k in keys_by_run[a.name] if k in theirs and pop.kind(k[0]) == agent_type]
    if not common:
      continue
    compared, unanswered, other_state, distances, identical = [], 0, 0, [], 0
    for key in common:
      p, q = distribution(a, key), distribution(b, key)
      if p is None or q is None:
        unanswered += 1
        continue
      state = a.records[key].get("state_sha1")
      if state is None or state != b.records[key].get("state_sha1"):
        other_state += 1
        continue
      compared.append(key)
      distance, same = spread(p, q, a.records[key].get("options") or [])
      distances.append(distance)
      identical += same
    right_a = {k for k in compared if outcome(a, k, pop.row(k)["truth"]) == RIGHT}
    right_b = {k for k in compared if outcome(b, k, pop.row(k)["truth"]) == RIGHT}
    n = len(compared)
    rows.append({"a": a.name, "b": b.name,
                 "models": a.model if a.model == b.model else "%s / %s" % (a.model, b.model),
                 "n": n, "other_state": other_state, "unanswered": unanswered,
                 "identical": ratio(identical, n),
                 "same_pick": ratio(sum(answer(a, k) == answer(b, k) for k in compared), n),
                 "distance": mean(distances),
                 "p95": percentile(distances, 95) if distances else None,
                 "acc_a": ratio(len(right_a), n), "acc_b": ratio(len(right_b), n),
                 **floor_cells(compared, pop),
                 "only_a": len(right_a - right_b), "only_b": len(right_b - right_a)})
  if not rows:
    return None
  return table("Replays: two runs of one typed condition, record by record",
               [("a", "run A", "s"), ("b", "run B", "s"), ("models", "models", "s"),
                ("n", "records compared", "d"), ("other_state", "left out, another state", "d"),
                ("unanswered", "left out, unanswered", "d"),
                ("identical", "identical", "pct"), ("same_pick", "same pick", "pct"),
                ("distance", "mean distance", "f3"), ("p95", "95th percentile", "f3"),
                ("acc_a", "raw accuracy A", "acc"), ("acc_b", "raw accuracy B", "acc")]
               + floor_columns(pop) + [("only_a", "only A right", "d"),
                                       ("only_b", "only B right", "d")],
               rows, floor="column",
               note="Over the common records whose `state_sha1` is the same in both runs and "
                    "that both answered; the rest are counted beside. Identical is the same "
                    "probability on every option, to Jev's step of 0.01. The distance is the "
                    "total variation distance, half the sum of |p - q| over the options: 0 "
                    "is the same distribution, 1 nothing in common. A pair whose models "
                    "differ is not a replay of one model.")


# --- the partisan divide --------------------------------------

def partisan_tables(runs, keys, pop, seed=SEED, resamples=RESAMPLES, least=MIN_PARTY):
  """The partisan divide of one agent type: who the partisans are, and for
  each run whether the gap between Democrats and Republicans on each item's majority answer
  follows the respondents' own, as r and k, each with a 95% interval over resamples of the
  respondents, and the gaps item by item. Every run is read over the
  same records, the partisans' records that every run answered, so the real gaps are one
  set. Only per-item sums leave this function, never a respondent's party."""
  import numpy as np
  order = {}
  for agent, _ in keys:
    order.setdefault(pop.agents[agent][0], len(order))
  party = np.array([pop.party(rid) or "" for rid in order])
  sides = {"D": np.flatnonzero(party == "D"), "R": np.flatnonzero(party == "R")}
  counts = table("Partisans among the respondents",
                 [("label", "party", "s"), ("respondents", "respondents", "d"),
                  ("share", "share", "pct")],
                 [{"label": label, "respondents": int(n), "share": ratio(int(n), len(order))}
                  for label, n in (("Democrats", len(sides["D"])),
                                   ("Republicans", len(sides["R"])),
                                   ("neither", len(order) - len(sides["D"]) - len(sides["R"])))],
                 note="PARTYID's three Democratic codes (strong, not very strong, independent "
                      "close to the Democrats) and its three Republican codes, from each "
                      "respondent's `party_identification`. A pure independent, another "
                      "party or a blank is neither.")

  partisan = [k for k in keys if pop.party(pop.agents[k[0]][0])]
  common = [k for k in partisan if all(answer(run, k) is not None for run in runs)]
  items = sorted({item for _, item in keys})
  column = {item: j for j, item in enumerate(items)}

  def per_item(value):
    """A respondents x items matrix of value(key) summed over the common records."""
    out = np.zeros((len(order), len(items)))
    for key in common:
      out[order[pop.agents[key[0]][0]], column[key[1]]] += value(key)
    return out

  def majority(key):
    return pop.row(key)[FLOOR]

  n = per_item(lambda k: 1)
  readable = np.flatnonzero((n[sides["D"]].sum(0) >= least) & (n[sides["R"]].sum(0) >= least))
  n = n[:, readable]
  real = per_item(lambda k: pop.row(k)["truth"] == majority(k))[:, readable]
  rng = np.random.default_rng(seed)
  weights = rng.multinomial(len(order), np.full(len(order), 1 / len(order)), size=resamples)

  def gap(w, hits):
    """Each item's share of Democrats minus share of Republicans, under the respondent
    weights `w`: one weight per respondent, or one row of them per resample."""
    shares = []
    for side in ("D", "R"):
      mine = w[..., sides[side]]
      with np.errstate(divide="ignore", invalid="ignore"):
        shares.append((mine @ hits[sides[side]]) / (mine @ n[sides[side]]))
    return shares[0] - shares[1]

  def divide(w, simulated):
    """r and k from the real and simulated gaps under the weights `w`."""
    x, y = gap(w, real), gap(w, simulated)
    with np.errstate(divide="ignore", invalid="ignore"):
      k = np.abs(y).mean(-1) / np.abs(x).mean(-1)
      x = x - x.mean(-1, keepdims=True)
      y = y - y.mean(-1, keepdims=True)
      r = (x * y).sum(-1) / np.sqrt((x * x).sum(-1) * (y * y).sum(-1))
    return r, k

  def finite(value):
    return float(value) if np.isfinite(value) else None

  whole = np.ones(len(order))
  real_gap = gap(whole, real)
  by_item = [{"item": items[j], "democrats": int(n[sides["D"], i].sum()),
              "republicans": int(n[sides["R"], i].sum()), "real": 100 * float(real_gap[i])}
             for i, j in enumerate(readable)]
  rows, columns = [], []
  for run in runs:
    kinds = [("picked", lambda k: answer(run, k) == majority(k))]
    if run.typed:
      kinds.append(("drawn", lambda k: distribution(run, k).get(majority(k), 0.0)))
    for kind, value in kinds:
      label = "%s, %s" % (run.name, kind)
      columns.append((label, label, "pts"))
      row = {"label": run.name, "row": kind, "items": len(readable),
             "democrats": int(n[sides["D"]].sum()), "republicans": int(n[sides["R"]].sum()),
             "left_out": len(partisan) - len(common)}
      if len(readable):
        simulated = per_item(value)[:, readable]
        sim_gap = gap(whole, simulated)
        for i, item_row in enumerate(by_item):
          item_row[label] = 100 * float(sim_gap[i])
        r, k = divide(whole, simulated)
        row.update(r=finite(r), k=finite(k), real=100 * float(np.abs(real_gap).mean()),
                   simulated=100 * float(np.abs(sim_gap).mean()))
        for name, values in zip(("r", "k"), divide(weights, simulated)):
          if row[name] is not None and np.isfinite(values).all():
            row[name + "_interval"] = tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))
      rows.append(row)
  divide_table = table(
      "The partisan divide: the gap between Democrats and Republicans, run against respondents",
      [("label", "run", "s"), ("row", "row", "s"), ("items", "items", "d"),
       ("democrats", "Democrats' records", "d"), ("republicans", "Republicans' records", "d"),
       ("left_out", "left out", "d"), ("real", "mean absolute real gap, points", "f1"),
       ("simulated", "mean absolute simulated gap, points", "f1"),
       ("r", "r", "f3"), ("r_interval", "95% interval", "ci3"),
       ("k", "k", "f3"), ("k_interval", "95% interval", "ci3")],
      rows,
      note="Partisans only, over the records every run answered; a partisan record some run "
           "left unanswered is left out and counted beside. On each item with at least %d "
           "records from each party, the gap is the share of Democrats giving the item's "
           "majority answer minus the share of Republicans: real from the respondents' own "
           "answers, simulated from the run (picked: its pick; drawn: the mean probability "
           "it gives that answer). r is the correlation of the real and simulated gaps across "
           "items. k is the mean |simulated gap| over the mean |real gap|: 1 is a crowd as "
           "divided as the people, under 1 flattens the divide, over 1 exaggerates it. The "
           "95%% intervals are over %s resamples of the %s respondents, seed %d. "
           "Both agent types already know the person's politics: the scratch "
           "carries `party_identification` and `political_views`, and the survey agent's "
           "memory carries its other answers. These are rates of a crowd, not accuracy, and "
           "no verdict is read from them."
           % (least, format(resamples, ","), format(len(order), ","), seed))
  item_table = table(
      "The partisan divide by item: Democrats' share minus Republicans', in points",
      [("item", "item", "s"), ("democrats", "Democrats' records", "d"),
       ("republicans", "Republicans' records", "d"), ("real", "real gap", "pts")] + columns,
      by_item,
      note="The gaps r and k are read from, over the same records: the share giving the "
           "item's majority answer among Democrats minus among Republicans, in points. "
           "Aggregates only.")
  return [counts, divide_table, item_table]


# --- the readout ------------------------------------------------------------

def readout(runs, root=None, against=None, seed=SEED, resamples=RESAMPLES, partisans=False,
            first=None, cross=(), family=None, contrast=()):
  """The report: the runs, their groups by expected records, the pairs, the replays, the
  cost, with `against` (a run's name) the intervals of its group's other runs against it,
  with `partisans` each group's partisan divide, and with `first` all of it over the first
  `first` respondents, across populations that share them. `cross` is a list
  of ((run, agent type), (run, agent type)) sides read across states that differ by design,
  over `family` (`single`, `battery` or one family) when given. `contrast` is
  a list of four sides, (A, B, C, D), each read as (A - B) - (C - D)."""
  if first is not None and first < 1:
    raise ValueError("--first %d: a head has at least one respondent" % first)
  pop = population_of(runs, root or lab_data(), first)
  keys_by_run = {run.name: expected(run, pop, first) for run in runs}
  groups = []
  for run in runs:
    keys = keys_by_run[run.name]
    for group in groups:
      if group["keys"] == keys:
        group["runs"].append(run)
        break
    else:
      groups.append({"keys": keys, "runs": [run]})
  for group in groups:
    lead = group["runs"][0]
    populations = ", ".join(dict.fromkeys(r.data.get("population") for r in group["runs"]))
    group["label"] = "%s%s, %s items per agent" % (
        "the first %d respondents of " % first if first is not None else "", populations,
        lead.data["items"]["per_agent"])
    group["by_type"] = {}
    for agent_type in AGENT_TYPES:
      keys = [k for k in group["keys"] if pop.kind(k[0]) == agent_type]
      if not keys:
        continue
      outside = (len([k for k in outside_keys(lead, pop, first)
                      if pop.kind(k[0]) == agent_type]) if pop.universe else None)
      tables = [accuracy_table(group["runs"], keys, pop, outside),
                family_table(group["runs"], keys, pop),
                crowd_table(group["runs"], keys, pop),
                middle_table(group["runs"], keys, pop),
                scores_table(group["runs"], keys, pop)]
      tables += [calibration_table(run, keys, pop) for run in group["runs"] if run.typed]
      if partisans:
        tables += partisan_tables(group["runs"], keys, pop, seed, resamples)
      group["by_type"][agent_type] = [t for t in tables if t]
  pairs = {t: pairs_table(runs, pop, keys_by_run, t) for t in AGENT_TYPES}
  replays = {t: replay_table(runs, pop, keys_by_run, t) for t in AGENT_TYPES}
  by_name = {run.name: run for run in runs}
  crossed = []
  for a, b in cross:
    for name, _ in (a, b):
      if name not in by_name:
        raise ValueError("--cross: %s is not one of the runs read" % name)
    crossed.append(cross_table((by_name[a[0]], a[1]), (by_name[b[0]], b[1]), keys_by_run, pop,
                               family, seed, resamples))
  contrasted = []
  for four in contrast:
    for name, _ in four:
      if name not in by_name:
        raise ValueError("--contrast: %s is not one of the runs read" % name)
    contrasted.append(contrast_table([(by_name[n], t) for n, t in four], keys_by_run, pop,
                                     family, seed, resamples))
  return {"runs": runs, "population": pop, "groups": groups, "first": first,
          "pairs": {t: p for t, p in pairs.items() if p},
          "replays": {t: r for t, r in replays.items() if r},
          "intervals": intervals(against, groups, pop, seed, resamples) if against else None,
          "cross": crossed, "contrast": contrasted,
          "cost": cost_table(runs, pop)}


# --- the cross reading ----------------------------------------

def in_family(family, name):
  """Whether a record of family `name` is read under the filter `family`."""
  if family is None:
    return True
  if family == SINGLE:
    return name == SINGLE
  if family == BATTERY:
    return name != SINGLE
  return name == family


def side_records(run, agent_type, keys_by_run, pop, family=None, flag="--cross"):
  """{(respondent, item): key} of one side, a run's agents of one type."""
  if agent_type not in AGENT_TYPES:
    raise ValueError("%s: %r is not an agent type (%s)" % (flag, agent_type, ", ".join(AGENT_TYPES)))
  keys = {}
  for key in keys_by_run[run.name]:
    if pop.kind(key[0]) == agent_type and in_family(family, pop.row(key)["family"]):
      keys[(pop.agents[key[0]][0], key[1])] = key
  return keys


def same_records(a, b, sides, flag="--cross"):
  """Refuse two sides that do not hold the same records, naming the first difference."""
  for (run, agent_type), mine, theirs in ((a, sides[0], sides[1]), (b, sides[1], sides[0])):
    extra = [record for record in mine if record not in theirs]  # in draw and bank order
    if extra:
      raise ValueError("%s: %s's %s agents hold respondent %s's %s, which the other side "
                       "does not, so the two sides are not over the same records"
                       % (flag, run.name, agent_type, extra[0][0], extra[0][1]))


def contrast_table(four, keys_by_run, pop, family=None, seed=SEED, resamples=RESAMPLES):
  """(A - B) - (C - D) in points of raw accuracy: each pair over the same
  records, the two pairs over the same respondents, each respondent's records carried whole
  through the resamples. Each pair's difference is printed beside the contrast."""
  flag = "--contrast"
  sides = [side_records(run, t, keys_by_run, pop, family, flag) for run, t in four]
  same_records(four[0], four[1], sides[:2], flag)
  same_records(four[2], four[3], sides[2:], flag)
  first_pair = list(dict.fromkeys(r for r, _ in sides[0]))
  second_pair = list(dict.fromkeys(r for r, _ in sides[2]))
  if set(first_pair) != set(second_pair):
    gone = [r for r in first_pair if r not in set(second_pair)] or [
        r for r in second_pair if r not in set(first_pair)]
    raise ValueError("--contrast: respondent %s is in one pair and not the other, so the two "
                     "pairs are not over the same respondents" % gone[0])
  if not first_pair:
    raise ValueError("--contrast: no record of %s's %s agents%s" % (
        four[0][0].name, four[0][1], "" if family is None else " in family %s" % family))
  order = {r: i for i, r in enumerate(first_pair)}
  columns = {}
  for name, side, run in (("a", 0, four[0][0]), ("b", 1, four[1][0]),
                          ("c", 2, four[2][0]), ("d", 3, four[3][0])):
    sums, count = [0.0] * len(order), [0.0] * len(order)
    for record, key in sides[side].items():
      count[order[record[0]]] += 1
      sums[order[record[0]]] += outcome(run, key, pop.row(key)["truth"]) == RIGHT
    columns[name] = sums
    columns["n" + name] = count
  first = lambda t: 100 * (t["a"] / t["na"] - t["b"] / t["nb"])
  second = lambda t: 100 * (t["c"] / t["nc"] - t["d"] / t["nd"])
  (d1, lo1, hi1), (d2, lo2, hi2), (c, lo, hi) = bootstrap(
      columns, [first, second, lambda t: first(t) - second(t)], seed, resamples)
  label = lambda side: "%s, %s" % (side[0].name, side[1])
  row = {"first": "%s minus %s" % (label(four[0]), label(four[1])),
         "second": "%s minus %s" % (label(four[2]), label(four[3])),
         "family": family or "all", "n_first": len(sides[0]), "n_second": len(sides[2]),
         "respondents": len(order), "d1": d1, "ci1": None if lo1 is None else (lo1, hi1),
         "d2": d2, "ci2": None if lo2 is None else (lo2, hi2), "contrast": c,
         "interval": None if lo is None else (lo, hi)}
  return table(
      "Contrast: (%s) minus (%s)" % (row["first"], row["second"]),
      [("first", "first pair", "s"), ("second", "second pair", "s"), ("family", "family", "s"),
       ("n_first", "records, first pair", "d"), ("n_second", "records, second pair", "d"),
       ("respondents", "respondents", "d"),
       ("d1", "first pair, points", "pts2"), ("ci1", "95% interval", "ci"),
       ("d2", "second pair, points", "pts2"), ("ci2", "95% interval", "ci"),
       ("contrast", "contrast, points", "pts2"), ("interval", "95% interval", "ci")],
      [row],
      note="Each pair is one side's raw accuracy minus the other's over the same respondents "
           "and items; the contrast is the first pair's difference minus the second's, in "
           "points. A record is counted whether or not its two sides met one state. The 95%% "
           "intervals are the 2.5th and 97.5th percentiles over %s resamples of the %s "
           "respondents, each carrying all of its records in both pairs, seed %d." % (format(resamples, ","), format(len(order), ","), seed))


def cross_table(a, b, keys_by_run, pop, family=None, seed=SEED, resamples=RESAMPLES):
  """Side A's raw accuracy minus side B's, each a (run, agent type), over the same
  respondents and items, where the two sides' states differ by design: a survey agent
  against a demographic agent, or a withholding population against the plain one. The
  pairs never meet such records, so this reading is its own, and says so.
  Refused when the two sides do not hold the same respondents and items."""
  sides = [side_records(run, agent_type, keys_by_run, pop, family) for run, agent_type in (a, b)]
  same_records(a, b, sides)
  if not sides[0]:
    raise ValueError("--cross: no record of %s's %s agents%s" % (
        a[0].name, a[1], "" if family is None else " in family %s" % family))
  shared = list(sides[0])
  order = {}
  for respondent, _ in shared:
    order.setdefault(respondent, len(order))
  columns = {}

  def column(name, value):
    sums = [0.0] * len(order)
    for record in shared:
      sums[order[record[0]]] += value(record)
    columns[name] = sums

  def right(side, run):
    return lambda r: outcome(run, sides[side][r], pop.row(sides[side][r])["truth"]) == RIGHT

  column("n", lambda r: 1)
  column("a", right(0, a[0]))
  column("b", right(1, b[0]))
  for label, floor_pop in pop.floors:
    column(("floor", label), lambda r, p=floor_pop: p.row(sides[0][r])[FLOOR]
           == p.row(sides[0][r])["truth"])
  (value, low, high), = bootstrap(
      columns, [lambda t: 100 * (t["a"] - t["b"]) / t["n"]], seed, resamples)
  differ = 0
  for r in shared:
    states = [run.records.get(side[r], {}).get("state_sha1")
              for (run, _), side in ((a, sides[0]), (b, sides[1]))]
    differ += states[0] is None or states[0] != states[1]
  n = len(shared)
  row = {"a": "%s, %s" % (a[0].name, a[1]), "b": "%s, %s" % (b[0].name, b[1]),
         "family": family or "all", "n_a": n, "n_b": n, "respondents": len(order),
         "differ": ratio(differ, n), "acc_a": sum(columns["a"]) / n,
         "acc_b": sum(columns["b"]) / n, "difference": value,
         "interval": None if low is None else (low, high)}
  for label, _ in pop.floors:
    row[label] = sum(columns[("floor", label)]) / n
  return table(
      "Cross reading, the two sides' states differ by design: %s against %s"
      % (row["a"], row["b"]),
      [("a", "side A", "s"), ("b", "side B", "s"), ("family", "family", "s"),
       ("n_a", "records A", "d"), ("n_b", "records B", "d"),
       ("respondents", "respondents", "d"), ("differ", "states differ", "pct"),
       ("acc_a", "raw accuracy A", "acc"), ("acc_b", "raw accuracy B", "acc")]
      + floor_columns(pop) + [("difference", "difference, points", "pts2"),
                              ("interval", "95% interval", "ci")],
      [row], floor="column",
      note="A's raw accuracy minus B's over the same respondents and items, in points: what "
           "the difference between the two sides' states is worth, never a pair over one "
           "state. States differ is the share of records whose `state_sha1` is not "
           "the same on both sides. The 95%% interval is the 2.5th and 97.5th percentiles over "
           "%s resamples of the %s respondents, each carrying all of its records, seed %d "
           "."
           % (format(resamples, ","), format(len(order), ","), seed))


def intervals(against, groups, pop, seed, resamples):
  """The intervals tables of the group holding the run named `against`, by agent type."""
  group = next((g for g in groups if against in [run.name for run in g["runs"]]), None)
  if group is None:
    raise ValueError("--against %s is not one of the runs read" % against)
  pilot = sorted(PILOT & {run.name for run in group["runs"]})
  if pilot:
    raise ValueError("no intervals on the pilot's runs (%s): its verdicts stand, and an "
                     "interval would read them a second time" % ", ".join(pilot))
  base = next(run for run in group["runs"] if run.name == against)
  others = [run for run in group["runs"] if run is not base]
  if not others:
    raise ValueError("--against %s: no other run is over the same records" % against)
  by_type = {}
  for agent_type in AGENT_TYPES:
    keys = [k for k in group["keys"] if pop.kind(k[0]) == agent_type]
    if keys:
      by_type[agent_type] = interval_tables(base, others, keys, pop, seed, resamples)
  return {"against": against, "by_type": by_type}


def tables(report):
  """Every table of the report, in the order it prints."""
  out = []
  for group in report["groups"]:
    for agent_type in AGENT_TYPES:
      out += group["by_type"].get(agent_type, [])
  out += list(report["pairs"].values())
  out += list(report["replays"].values())
  out += report.get("cross") or []
  out += report.get("contrast") or []
  if report.get("intervals"):
    for agent_type in AGENT_TYPES:
      out += report["intervals"]["by_type"].get(agent_type, [])
  out.append(report["cost"])
  return out


# --- Markdown ---------------------------------------------------------------

def cell(value, fmt):
  if value is None:
    return "-"
  if fmt == "s":
    return str(value)
  if fmt == "d":
    return "%d" % value
  if fmt == "acc":
    return "%.3f" % value
  if fmt == "f3":
    return "%.3f" % value
  if fmt == "f1":
    return "%.1f" % value
  if fmt == "pct":
    return "%.1f%%" % (100 * value)
  if fmt == "pts":
    return "%+.1f" % value
  if fmt == "pts2":
    return "%+.2f" % value
  if fmt == "ci":
    return "[%+.2f, %+.2f]" % value
  if fmt == "ci3":
    return "[%.3f, %.3f]" % value
  if fmt == "usd":
    return "$%.4f" % value
  raise ValueError(fmt)


def markdown_table(t, level="####"):
  lines = ["%s %s" % (level, t["title"]), ""]
  if t.get("note"):
    lines += [t["note"], ""]
  lines.append("| " + " | ".join(h for _, h, _ in t["columns"]) + " |")
  lines.append("|" + "|".join("---" for _ in t["columns"]) + "|")
  for row in t["rows"]:
    lines.append("| " + " | ".join(cell(row.get(k), f) for k, _, f in t["columns"]) + " |")
  return lines + [""]


def render(report):
  runs = report["runs"]
  lines = ["# Readout: %s" % ", ".join(run.name for run in runs), "",
           "Every accuracy here is **raw accuracy**: the respondents are public and have no "
           "retest. The floor is the 2022 weighted majority answer of each item, "
           "over the respondents not drawn.", ""]
  unregistered = [run.name for run in runs if run.status == "unregistered"]
  if unregistered:
    lines += ["**Unregistered:** %s. A number from an unregistered run cites no run and is "
              "not citable." % ", ".join(unregistered), ""]
  # A registered run that is not citable is evidence of a failure, never an accuracy
  invalidated = [run.name for run in runs if run.status not in ("citable", "unregistered")]
  if invalidated:
    lines += ["**Not citable:** %s, registered as `invalidated`. Its numbers are printed as "
              "evidence of what failed; none of them is an accuracy."
              % ", ".join(invalidated), ""]
  lines += ["## The runs", "",
            "| run | status | arm | population | code | stopped |", "|---|---|---|---|---|---|"]
  for run in runs:
    outcome_ = run.head.get("outcome") or {}
    lines.append("| %s | %s | %s | %s | %s | %s |" % (
        run.name, run.status, run.arm, run.data.get("population"),
        ((run.head.get("code") or {}).get("sha") or "?")[:8],
        outcome_.get("stopped") or ("-" if outcome_ else "no outcome")))
  lines.append("")
  for pop in report["population"].pops:
    lines.append("Population: `%s`, its truth file sha256 `%s`." % (
        pop.path or runs[0].data.get("population_path"), pop.sha256["truth.jsonl"][:12]))
  lines.append("")
  for pop in report["population"].pops:
    if pop.every:
      lines += ["%s holds every answer each respondent gave in its survey agents' memory, "
                "built from %s, with one map per level (%s)%s." % (
          pop.name, pop.every, ", ".join("`%s` for %s" % (f, level) for level, f in
                                         sorted((pop.meta.get("maps") or {}).items())),
          ": its %d records outside an item's universe are left out of every "
          "table, on every run read with it" % len(pop.universe)
          if pop.universe else ""), ""]
    elif pop.extra:
      lines += ["%s adds %s to what a plain build writes%s." % (
          pop.name, " and ".join("`%s`" % name for name in pop.extra),
          ": its %d records outside an item's universe are left out of every "
          "table, on every run read with it" % len(pop.universe)
          if pop.universe else ""), ""]
  first = report.get("first")
  if first is not None and len(report["population"].pops) > 1:
    pop = report["population"]
    lines += ["Read over **the first %d respondents**, which %s share: the same ids in the "
              "same order, items, truth rows and agent files. Each floor is its "
              "own population's, labelled. The majority answer of the crowd, the middle, the "
              "partisan tables and the crowd's D is %s's floor." % (
                  first, pop.name, pop.lead.name), ""]
  elif first is not None:
    lines += ["Read over **the first %d respondents** of %s."
              % (first, report["population"].name), ""]
  for group in report["groups"]:
    lines += ["## %s: %s" % (group["label"], ", ".join(r.name for r in group["runs"])), ""]
    for agent_type in AGENT_TYPES:
      if agent_type not in group["by_type"]:
        if not any(agent_type in run_types(run) for run in group["runs"]):
          lines += ["### %s agents" % agent_type.capitalize(), "",
                    "No run of this group asked %s agents (`--agents`)."
                    % agent_type, ""]
        continue
      lines += ["### %s agents" % agent_type.capitalize(), ""]
      for t in group["by_type"][agent_type]:
        lines += markdown_table(t)
  if report["pairs"]:
    lines += ["## Pairs", ""]
    for agent_type, t in report["pairs"].items():
      lines += markdown_table(dict(t, title="%s, %s agents" % (t["title"], agent_type)), "###")
  if report["replays"]:
    lines += ["## Replays", ""]
    for agent_type, t in report["replays"].items():
      lines += markdown_table(dict(t, title="%s, %s agents" % (t["title"], agent_type)), "###")
  if report.get("cross"):
    lines += ["## Cross readings", ""]
    for t in report["cross"]:
      lines += markdown_table(t, "###")
  if report.get("contrast"):
    lines += ["## Contrasts", ""]
    for t in report["contrast"]:
      lines += markdown_table(t, "###")
  if report.get("intervals"):
    lines += ["## Intervals against %s" % report["intervals"]["against"], ""]
    for agent_type, ts in report["intervals"]["by_type"].items():
      lines += ["### %s agents" % agent_type.capitalize(), ""]
      for t in ts:
        lines += markdown_table(t)
  lines += ["## Cost", ""] + markdown_table(report["cost"], "###")
  return "\n".join(lines).rstrip() + "\n"


def parse_cross(text):
  """((run, agent type), (run, agent type)) of `A:TYPE:B:TYPE`."""
  parts = text.split(":")
  if len(parts) != 4:
    raise ValueError("--cross %s: expected A:TYPE:B:TYPE" % text)
  return (parts[0], parts[1]), (parts[2], parts[3])


def parse_contrast(text):
  """Four (run, agent type) sides of `A:TYPE:B:TYPE:C:TYPE:D:TYPE`."""
  parts = text.split(":")
  if len(parts) != 8:
    raise ValueError("--contrast %s: expected A:TYPE:B:TYPE:C:TYPE:D:TYPE" % text)
  return [(parts[i], parts[i + 1]) for i in range(0, 8, 2)]


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("runs", nargs="+", help="run names (registered, or in runs-local/) or folders")
  ap.add_argument("--out", default=None, help="write the Markdown here instead of stdout")
  ap.add_argument("--against", default=None, metavar="RUN",
                  help="add the intervals of RUN's group's other runs against RUN")
  ap.add_argument("--partisans", action="store_true",
                  help="add each group's partisan divide, with intervals")
  ap.add_argument("--first", type=int, default=None, metavar="N",
                  help="read every run over its population's first N respondents, and "
                       "populations that share them together")
  ap.add_argument("--cross", action="append", default=[], metavar="A:TYPE:B:TYPE",
                  help="A's agents of TYPE against B's agents of TYPE, whose states differ by "
                       "design, over the same respondents and items")
  ap.add_argument("--family", default=None,
                  help="the cross readings and contrasts over single, battery or one family only")
  ap.add_argument("--contrast", action="append", default=[], metavar="A:TYPE:B:TYPE:C:TYPE:D:TYPE",
                  help="(A - B) - (C - D), each pair over the same records, with intervals "
                       "")
  args = ap.parse_args(argv)
  try:
    text = render(readout([load_run(name) for name in args.runs], against=args.against,
                          partisans=args.partisans, first=args.first,
                          cross=[parse_cross(c) for c in args.cross], family=args.family,
                          contrast=[parse_contrast(c) for c in args.contrast]))
  except ValueError as exc:
    sys.exit("no readout: %s" % exc)
  if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
      f.write(text)
  else:
    sys.stdout.write(text)


if __name__ == "__main__":
  main()
