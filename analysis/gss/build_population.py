"""The builder: GSS 2022 respondents become a built population under LAB_DATA.

Usage (from the repository root):

  python analysis/gss/build_population.py --dry   draw, slice and floor; print, write nothing
  python analysis/gss/build_population.py         ... and write gss22-a, embedding its nodes
  python analysis/gss/build_population.py --name gss22-b --n 1052   a longer draw
  python analysis/gss/build_population.py --name gss22-c --n 1052 --withhold-families gss22-b
                                                     gss22-b plus what is withheld
  python analysis/gss/build_population.py --name gss22-d --n 1052 --every-answer gss22-c
                                                     every answer in memory, a map per level

A build, in order:

1. The 2022 file is read, refusing a `.dta` whose sha256 is not the recorded one, and every
   respondent walks the recode table (`recode.py`, S2.1, S2.2).
2. **The draw.** A respondent is eligible with at least 60 scorable items.
   `--n` of them (100 by default) are drawn without replacement, with probability
   proportional to `wtssnrps_as` and the seed 20260926. The draw order is the order of the
   keys `log(u) / w` (Efraimidis and Spirakis), which is sequential weighted sampling: the
   pilot is the first 20 and the `gpt-4o` pilot the first 10, and a longer draw with the
   same seed keeps the shorter one as its head.
3. **The slice.** 30 of the respondent's scorable items, drawn uniformly with a seed
   of its own (`<seed>:<id>`) and kept in bank order. The seven scratch items are never
   scorable, since the table does not score them.
4. **The scratch.** The shipped keys in the shipped order, less the five a public
   respondent cannot fill (`SCRATCH`). A key the respondent has no answer for is left out.
5. **The nodes.** One per answered item outside the slice, in bank order:
   `Q: <question as asked>\\nA: <answer>`, importance 50, `created` from 1. The embeddings
   come through the fork's own door (`get_text_embedding`, D2 and D3: OpenRouter), keyed
   by the node's content, as the paper's `_add_node` keys them.
6. **The floor.** Each scored item's weighted mode over the respondents not drawn,
   scorable answers only; a tie goes to the option the bank lists first.
7. **The files**, under `LAB_DATA/populations/<name>/`:
   - `agents/<id>-demographic/` and `agents/<id>-survey/`, in the bank's layout
     (`scratch.json`, `meta.json`, `memory_stream/nodes.json` and `embeddings.json`);
   - `items.json`: each agent's held-out ids, the same list for both of a respondent's agents;
   - `truth.jsonl`: per respondent and held-out item, the answer and the floor's option;
   - `population.json`: the seed, the sha256 of the source, the table, `items.json` and the
     truth file, the draw, and what was dropped and why;
   - `build.json`: when, and what the embeddings cost. Two builds of the same seed differ
     here and in the survey agents' `embeddings.json`, whose vectors the service does not
     return identically; everything else is the same.
8. **With `--withhold-families <plain>`**, two more files, named by sha256 in
   `population.json`, and nothing else changed:
   - `withhold.json`: for each survey agent and each held-out item, the `node_id`s withheld
     when that item is asked: every node of the item's family, none for a `single` item
;
   - `universe.json`: the held-out records whose respondent sits outside the item's
     universe, the trap records.
   The embeddings are the plain population's, copied rather than asked again, and the
   build is written beside its folder first: unless items, truth and every agent file are
   byte-identical to `<plain>`'s, it is deleted and the build refused.
9. **With `--every-answer <plain>`**, the survey agent's memory holds every
   answer the respondent gave, the held-out ones included, one node each in bank order, and
   three maps say what each held-out item may see:
   - `withhold.json` (level `every`): the item's own node and every node of its family;
   - `withhold-k10.json`, `withhold-k30.json`: all but the first 10 or 30 nodes, in one
     shuffle per respondent (seed `<seed>:<id>:context`), outside the item and its family;
   - `universe.json`, as above.
   `population.json` lists the maps by level under `maps`. The plain population's vectors
   are copied, and only a text it lacks is embedded. The build is refused, and leaves
   nothing, unless items, truth, universe, every scratch and meta, every demographic agent
   and every survey agent's plain nodes (by content, in order, with their vectors) are the
   plain population's, and no map lets an item see its own answer or a family node.

The population is written under LAB_DATA.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import threading
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))


def _recode():
  spec = importlib.util.spec_from_file_location("recode", os.path.join(HERE, "recode.py"))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


recode = _recode()

NAME = "gss22-a"
SEED = 20260926
DRAW = 100
SLICE = 30
MIN_SCORABLE = 60
WEIGHT = "wtssnrps_as"
IMPORTANCE = 50
EMBEDDING_MODEL = "text-embedding-3-small"  # the paper's default in get_text_embedding
EMBEDDING_USD_PER_MTOK = 0.02
SHIPPED = os.path.join(recode.FORK, "agent_bank", "populations", "gss_agents")

# The shipped scratch's 30 keys in their order, each with the 2022 variable that fills it.
# None marks a key left out: a synthetic name or place would be evidence the
# respondent never gave.
SCRATCH = [
  ("first_name", None), ("last_name", None),
  ("age", "age"), ("sex", "sex"), ("ethnicity", "ethnic"), ("race", "race"),
  ("detailed_race", "racecen1"), ("hispanic_origin", "hispanic"),
  ("street_address", None), ("city", None), ("state", None),
  ("political_views", "polviews"), ("party_identification", "partyid"),
  ("residence_at_16", "reg16"), ("same_residence_since_16", "mobile16"),
  ("family_structure_at_16", "famdif16"), ("family_income_at_16", "incom16"),
  ("fathers_highest_degree", "padeg"), ("mothers_highest_degree", "madeg"),
  ("mothers_work_history", "mawrkgrw"), ("marital_status", "marital"),
  ("work_status", "wrkstat"), ("military_service_duration", "vetyears"),
  ("religion", "relig"), ("religion_at_16", "relig16"), ("born_in_us", "born"),
  ("us_citizenship_status", "uscitzn"), ("highest_degree_received", "degree"),
  ("speak_other_language", "othlang"), ("total_wealth", "wealth"),
]
# Other variables a scratch value reads.
SCRATCH_ALSO = ("family16",)

# The trap records: five items the 2022 file answers for respondents
# whom the bank's "Inapplicable" option describes. The definitions follow an earlier
# review's `repro_notes.py`, code for code: RELIG 3, 4, 6, 7, 8, 9 and 12 are not Christian, SPREL 4
# is a spouse with no religion, WRKSTAT 4 to 8 is not working.
NON_CHRISTIAN = frozenset((3, 4, 6, 7, 8, 9, 12))
NO_RELIGION = 4
NOT_WORKING = frozenset((4, 5, 6, 7, 8))
UNIVERSE = {
  "REBORN": ("relig", NON_CHRISTIAN, "the respondent's religion is not Christian (RELIG)"),
  "SPFUND": ("sprel", {NO_RELIGION}, "the respondent's spouse has no religion (SPREL)"),
  "WKSUB1": ("wrkstat", NOT_WORKING, "the respondent does not work (WRKSTAT 4 to 8)"),
  "WKSUP1": ("wrkstat", NOT_WORKING, "the respondent does not work (WRKSTAT 4 to 8)"),
  "SATJOB": ("wrkstat", NOT_WORKING, "the respondent does not work (WRKSTAT 4 to 8)"),
}
UNIVERSE_ALSO = ("relig", "sprel", "wrkstat")
SINGLE = "single"  # the family of an item that belongs to none
# The files a withholding build adds, and the ones it must leave byte-identical.
EXTRA = ("withhold.json", "universe.json")
SAME = ("items.json", "truth.jsonl")
# The levels of an every-answer build, each a map; `every` is withhold.json.
LEVELS = (10, 30)
EVERY = "every"
CONTEXT = "%s:%s:context"  # the seed of each respondent's one shuffle of its answers
AGENT_FILES = ("scratch.json", "meta.json", os.path.join("memory_stream", "nodes.json"),
               os.path.join("memory_stream", "embeddings.json"))

# Where the shipped vocabulary names a 2022 category differently. Every other value is the
# file's label, spelled as the shipped bank spells it when it has the same label.
RELABEL = {
  # The shipped family_structure_at_16 is "Lived with parents" or FAMDIF16's reason; with
  # no reason given, FAMILY16's own label.
  "famdif16": {1: "Parent died", 2: "Divorce, separated", 3: "Armed forces",
               4: "Institution", 5: "Other (left home before 16, e.g., for school)"},
  "mobile16": {1: "Same city"},
  # The shipped detailed_race has six groups; RACECEN1 names 13.
  "racecen1": {3: "Native American or Alaska Native",
               **{c: "Asian" for c in range(4, 11)},
               **{c: "Native Hawaiian or Other Pacific Islander" for c in range(11, 15)},
               16: "Hispanic or Latino"},
}
LIVED_WITH_PARENTS = "Lived with parents"  # FAMILY16 1, "both own parents"
US_CITIZEN = "A U.S. citizen"  # USCITZN is asked only of the foreign-born (BORN 2)


# --- the scratch ----------------------------------------------------------------

def vocabulary(root=SHIPPED):
  """{key: {lowercased value: the shipped spelling}} over every shipped scratch."""
  vocab = defaultdict(dict)
  if not os.path.isdir(root):
    return vocab
  for folder in sorted(os.listdir(root)):
    path = os.path.join(root, folder, "scratch.json")
    if not os.path.isfile(path):
      continue
    with open(path) as f:
      for key, value in json.load(f).items():
        if isinstance(value, str):
          vocab[key].setdefault(value.lower(), value)
  return vocab


def _spell(key, label, vocab):
  return vocab.get(key, {}).get(label.lower()) or label[:1].upper() + label[1:]


def scratch(record, labels, vocab):
  """The respondent's scratch: the shipped keys this public file fills, in shipped order."""
  out = {}
  for key, variable in SCRATCH:
    if variable is None:
      continue
    if key == "family_structure_at_16" and record.get("family16") == 1:
      out[key] = LIVED_WITH_PARENTS
      continue
    if key == "family_structure_at_16" and recode._missing(record.get("famdif16")):
      variable = "family16"  # no reason given: the structure itself
    if key == "us_citizenship_status" and record.get("born") == 1:
      out[key] = US_CITIZEN
      continue
    x = record.get(variable)
    if recode._missing(x):
      continue
    code = int(x)
    if key == "age":
      out[key] = code
    elif code in RELABEL.get(variable, {}):
      out[key] = RELABEL[variable][code]
    elif code in labels.get(variable, {}):
      out[key] = _spell(key, labels[variable][code], vocab)
    else:
      raise KeyError("%s: code %s of %s has no label" % (key, code, variable))
  return out


# --- the draw, the slice, the nodes, the floor -----------------------------------

def bank_order(bank):
  return {q: i for i, q in enumerate(bank)}


def scorable(got):
  return [q for q, a in got.items() if a.scorable]


def draw(weights, n, seed):
  """n ids, drawn without replacement with probability proportional to their weight, in order."""
  rng = random.Random(seed)
  keys = [(math.log(rng.random()) / weights[rid], rid) for rid in sorted(weights)]
  return [rid for _, rid in sorted(keys, reverse=True)[:n]]


def held_out(got, rid, order, seed, k):
  """k scorable items of one respondent, drawn with its own seed, in bank order."""
  pool = sorted(scorable(got), key=order.get)
  return sorted(random.Random("%s:%s" % (seed, rid)).sample(pool, k), key=order.get)


def survey_nodes(got, held, bank, order):
  """[(item_id, node)]: one observation per answered item outside the slice, in bank order."""
  kept = sorted((q for q in got if q not in held), key=order.get)
  out = []
  for i, q in enumerate(kept):
    a = got[q]
    content = "Q: %s\nA: %s" % (a.text or bank[q].text, a.answer)
    out.append((q, {"node_id": i, "node_type": "observation", "content": content,
                    "importance": IMPORTANCE, "created": i + 1, "last_retrieved": i + 1,
                    "pointer_id": None}))
  return out


def outside(record, held):
  """The held-out items whose universe the respondent sits outside."""
  out = []
  for q in held:
    rule = UNIVERSE.get(q)
    if rule and not recode._missing(record.get(rule[0])) and int(record[rule[0]]) in rule[1]:
      out.append(q)
  return out


def families_of(rows):
  """{item_id: family} over every row that is a truth or a memory."""
  out = {}
  for r in rows:
    if r.kind == recode.NONE:
      continue
    if out.setdefault(r.item_id, r.family) != r.family:
      raise ValueError("%s is in two families, %s and %s" % (r.item_id, out[r.item_id], r.family))
  return out


def withheld(p, families):
  """{"gss:<held-out id>": [node_id]}: for each held-out item of one survey agent, every node
  of its family, memory-only rows included, and nothing for a `single` item."""
  out = {}
  for q in p["held"]:
    family = families[q]
    out["gss:" + q] = ([] if family == SINGLE else
                       [node["node_id"] for item, node in p["nodes"] if families[item] == family])
  return out


def every_maps(p, families, seed=SEED, levels=LEVELS):
  """{level: {"gss:<held-out id>": [node_id]}} for one survey agent holding every answer
. `every` withholds the item's own node and every node of its family,
  and level k all but the first k nodes of the respondent's one shuffle that are neither.
  So k10 keeps a subset of what k30 keeps, and k30 of what every keeps."""
  nodes = p["every_nodes"]
  ids = [node["node_id"] for _, node in nodes]
  shuffle = random.Random(CONTEXT % (seed, p["id"])).sample(ids, len(ids))
  maps = {EVERY: {}, **{"k%d" % k: {} for k in levels}}
  for q in p["held"]:
    family = families[q]
    gone = sorted(node["node_id"] for item, node in nodes
                  if item == q or (family != SINGLE and families[item] == family))
    maps[EVERY]["gss:" + q] = gone
    out = set(gone)
    pool = [i for i in shuffle if i not in out]
    for k in levels:
      kept = set(pool[:k])
      maps["k%d" % k]["gss:" + q] = [i for i in ids if i not in kept]
  return maps


def map_leak(p, maps, families):
  """The first record whose state a map leaves its own answer or a family node in, or None:
  the check an every-answer build is refused on."""
  item_of = {node["node_id"]: item for item, node in p["every_nodes"]}
  for level, by_item in maps.items():
    for key, gone in by_item.items():
      q, kept = key[len("gss:"):], set(item_of) - set(gone)
      for node_id in kept:
        item = item_of[node_id]
        if item == q or (families[q] != SINGLE and families[item] == families[q]):
          return "%s: %s's %s keeps %s, of its own family" % (level, p["id"], key, item)
  return None


def floor(walked, weights, drawn, bank):
  """{item_id: option}: the weighted mode of each scored item over the respondents not drawn."""
  drawn = set(drawn)
  mass = defaultdict(Counter)
  for rid, got in walked.items():
    if rid in drawn:
      continue
    for q, a in got.items():
      if a.scorable:
        mass[q][a.answer] += weights[rid]
  return {q: max(bank[q].options, key=lambda o: (c[o], -bank[q].options.index(o)))
          for q, c in sorted(mass.items())}


# --- the build ------------------------------------------------------------------

def plan(records, rows, bank, labels, vocab, n=DRAW, k=SLICE, least=MIN_SCORABLE, seed=SEED,
         every=False):
  """Everything but the embeddings and the files: a dict the writer and the tests read. With
  `every`, each person also carries `every_nodes`, one per answered item, the held-out ones
  included."""
  order = bank_order(bank)
  walked, weights, info = {}, {}, {}
  for record in records:
    rid = str(int(record["id"]))
    walked[rid] = recode.answers(record, rows)
    weights[rid] = float(record[WEIGHT])
    info[rid] = record
  eligible = {rid: w for rid, w in weights.items() if len(scorable(walked[rid])) >= least}
  if len(eligible) < n:
    raise ValueError("%d eligible respondents, fewer than the %d to draw" % (len(eligible), n))
  drawn = draw(eligible, n, seed)
  base = floor(walked, weights, drawn, bank)
  people = []
  for position, rid in enumerate(drawn, 1):
    got = walked[rid]
    held = held_out(got, rid, order, seed, k)
    missing = [q for q in held if q not in base]
    if missing:
      raise ValueError("no floor for %s: every answer to them was drawn" % missing)
    people.append({
      "id": rid, "position": position, "weight": weights[rid],
      "form": int(info[rid]["form"]), "mode": None if recode._missing(info[rid].get("mode"))
      else int(info[rid]["mode"]),
      "scratch": scratch(info[rid], labels, vocab), "held": held,
      "nodes": survey_nodes(got, set(held), bank, order),
      "truth": {q: got[q].answer for q in held}, "floor": {q: base[q] for q in held},
      "outside": outside(info[rid], held),
      "scorable": len(scorable(got)), "answered": len(got)})
    if every:
      people[-1]["every_nodes"] = survey_nodes(got, set(), bank, order)
  return {"people": people, "floor": base, "respondents": len(walked), "eligible": len(eligible),
          "n": n, "k": k, "least": least, "seed": seed}


def agent_ids(rid):
  return {"demographic": "%s-demographic" % rid, "survey": "%s-survey" % rid}


def _dump(path, data, indent=2):
  text = json.dumps(data, indent=indent, ensure_ascii=False)
  with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.write(text + "\n")


def _write_agent(folder, agent_id, scratch_, nodes, embeddings):
  os.makedirs(os.path.join(folder, "memory_stream"))
  _dump(os.path.join(folder, "scratch.json"), scratch_)
  _dump(os.path.join(folder, "meta.json"), {"id": agent_id})
  _dump(os.path.join(folder, "memory_stream", "nodes.json"), nodes)
  with open(os.path.join(folder, "memory_stream", "embeddings.json"), "w") as f:
    json.dump(embeddings, f)


def embed_all(texts, embed, threads=8):
  """{text: vector} for every distinct text, `threads` requests at a time."""
  texts = sorted(set(texts))
  with concurrent.futures.ThreadPoolExecutor(threads) as pool:
    return dict(zip(texts, pool.map(embed, texts)))


def write(built, rows, bank, out, embed, source_sha256, table_sha256, name=NAME, plain=None,
          every=None):
  """Write the population under `out`, which must not exist; return population.json's dict.

  With `plain`, the folder of a population built plain from the same seed and size, the
  build withholds: it adds `withhold.json` and `universe.json`, takes its
  embeddings from `plain` rather than `embed`, and is refused unless everything else it
  shares with `plain` is byte-identical. It is written beside `out` and moved there only
  once that check passed, so a refused build leaves nothing.

  With `every`, the folder of a population built from the same seed and size, the survey
  agents hold every answer and the build writes one map per level. Its
  vectors are `every`'s where it has them and `embed`'s for a text it lacks; it is staged
  and checked the same way, by `every_difference`."""
  if os.path.exists(out):
    raise FileExistsError("%s exists: a built population is never overwritten" % out)
  if plain is None and every is None:
    return _write(built, rows, bank, out, embed, source_sha256, table_sha256, name)
  if plain is not None and every is not None:
    raise ValueError("a build withholds families or holds every answer, not both")
  staging = out + ".partial"
  if os.path.exists(staging):
    raise FileExistsError("%s exists: a refused build left it; look, then remove it" % staging)
  try:
    if every is not None:
      base = os.path.basename(every.rstrip(os.sep))
      written = _write(built, rows, bank, staging, plain_embedder(every, embed), source_sha256,
                       table_sha256, name, every=base)
      difference = every_difference(staging, every)
      if difference:
        raise ValueError("refused: %s, so it is not %s with every answer in memory"
                         % (difference, base))
    else:
      written = _write(built, rows, bank, staging, plain_embedder(plain), source_sha256,
                       table_sha256, name, withhold=os.path.basename(plain.rstrip(os.sep)))
      difference = build_difference(staging, plain)
      if difference:
        raise ValueError("refused: %s, so it is not %s plus what is withheld"
                         % (difference, os.path.basename(plain.rstrip(os.sep))))
  except BaseException:
    shutil.rmtree(staging, ignore_errors=True)
    raise
  os.rename(staging, out)
  return written


def plain_embedder(plain, fresh=None):
  """The plain population's stored vectors, as an embedder that asks nothing: a fresh
  vector would be a difference nobody chose. With `fresh`, a text the plain
  population lacks is embedded by it; without, it is refused."""
  vectors = {}
  agents = os.path.join(plain, "agents")
  for agent in sorted(os.listdir(agents)):
    path = os.path.join(agents, agent, "memory_stream", "embeddings.json")
    with open(path) as f:
      vectors.update(json.load(f))

  def embed(text):
    if text not in vectors:
      if fresh is not None:
        return fresh(text)
      raise ValueError("%r has no vector in %s" % (text[:60], plain))
    return vectors[text]
  return embed


def build_difference(new, plain):
  """The first file `new` holds that is not byte-identical to `plain`'s, or None: items,
  truth, the agents and every agent file."""
  for name in SAME:
    if recode.sha256_of(os.path.join(new, name)) != recode.sha256_of(os.path.join(plain, name)):
      return "%s differs from the plain build's" % name
  agents = sorted(os.listdir(os.path.join(new, "agents")))
  if agents != sorted(os.listdir(os.path.join(plain, "agents"))):
    return "the agents differ from the plain build's"
  for agent in agents:
    for name in AGENT_FILES:
      paths = [os.path.join(root, "agents", agent, name) for root in (new, plain)]
      if not os.path.isfile(paths[1]) or (recode.sha256_of(paths[0])
                                          != recode.sha256_of(paths[1])):
        return "agent %s's %s differs from the plain build's" % (agent, name)
  return None


def every_difference(new, plain):
  """The first way `new` is not `plain` with every answer in its survey agents' memory, or
  None: items, truth and universe; the agents; every scratch and meta; every
  demographic agent's files; and each survey agent's plain nodes, by content and in order,
  each with the plain vector."""
  for name in SAME + ("universe.json",):
    if recode.sha256_of(os.path.join(new, name)) != recode.sha256_of(os.path.join(plain, name)):
      return "%s differs from the plain build's" % name
  agents = sorted(os.listdir(os.path.join(new, "agents")))
  if agents != sorted(os.listdir(os.path.join(plain, "agents"))):
    return "the agents differ from the plain build's"
  for agent in agents:
    names = AGENT_FILES if agent.endswith("-demographic") else AGENT_FILES[:2]
    for name in names:
      paths = [os.path.join(root, "agents", agent, name) for root in (new, plain)]
      if not os.path.isfile(paths[1]) or (recode.sha256_of(paths[0])
                                          != recode.sha256_of(paths[1])):
        return "agent %s's %s differs from the plain build's" % (agent, name)
    if agent.endswith("-demographic"):
      continue
    def read(root, f):
      with open(os.path.join(root, "agents", agent, "memory_stream", f), encoding="utf-8") as fh:
        return json.load(fh)
    mine = [n["content"] for n in read(new, "nodes.json")]
    theirs = [n["content"] for n in read(plain, "nodes.json")]
    it = iter(mine)
    if not all(content in it for content in theirs):
      return "agent %s's plain nodes are not in its memory, in order" % agent
    vectors, plain_vectors = read(new, "embeddings.json"), read(plain, "embeddings.json")
    if any(vectors.get(c) != plain_vectors[c] for c in theirs):
      return "agent %s's vector for a plain node is not the plain build's" % agent
  return None


def _write(built, rows, bank, out, embed, source_sha256, table_sha256, name, withhold=None,
           every=None):
  order = bank_order(bank)
  families = {r.item_id: r.family for r in rows if r.kind == recode.SCORED}
  people = built["people"]
  memory = "every_nodes" if every else "nodes"
  if every and any("every_nodes" not in p for p in people):
    raise ValueError("an every-answer build needs a plan made with every=True")
  vectors = embed_all([node["content"] for p in people for _, node in p[memory]], embed)
  os.makedirs(os.path.join(out, "agents"))
  items, truth = {}, []
  for p in people:
    ids = agent_ids(p["id"])
    nodes = [node for _, node in p[memory]]
    _write_agent(os.path.join(out, "agents", ids["demographic"]), ids["demographic"],
                 p["scratch"], [], {})
    _write_agent(os.path.join(out, "agents", ids["survey"]), ids["survey"], p["scratch"],
                 nodes, {node["content"]: vectors[node["content"]] for node in nodes})
    held = ["gss:" + q for q in p["held"]]
    items[ids["demographic"]] = held
    items[ids["survey"]] = held
    truth += [{"respondent": p["id"], "item": "gss:" + q, "family": families[q],
               "truth": p["truth"][q], "floor": p["floor"][q]} for q in p["held"]]
  _dump(os.path.join(out, "items.json"), items)
  with open(os.path.join(out, "truth.jsonl"), "w", encoding="utf-8", newline="\n") as f:
    for record in truth:
      f.write(json.dumps(record, ensure_ascii=False) + "\n")
  maps = {}
  if every:
    all_families = families_of(rows)
    by_level = {}
    for p in people:
      mine = every_maps(p, all_families, built["seed"])
      leak = map_leak(p, mine, all_families)
      if leak:
        raise ValueError("refused: %s" % leak)
      for level, by_item in mine.items():
        by_level.setdefault(level, {})[agent_ids(p["id"])["survey"]] = by_item
    for level, by_agent in by_level.items():
      maps[level] = "withhold.json" if level == EVERY else "withhold-%s.json" % level
      _dump(os.path.join(out, maps[level]), by_agent)
  if withhold:
    all_families = families_of(rows)
    _dump(os.path.join(out, "withhold.json"),
          {agent_ids(p["id"])["survey"]: withheld(p, all_families) for p in people})
  if withhold or every:
    _dump(os.path.join(out, "universe.json"), {
      "rule": "trap records: a held-out item whose 2022 answer belongs to a "
              "respondent the bank's 'Inapplicable' option describes",
      "items": {"gss:" + q: rule[2] for q, rule in sorted(UNIVERSE.items())},
      "records": [{"respondent": p["id"], "item": "gss:" + q}
                  for p in people for q in p["outside"]]})
  nodes_per = [len(p[memory]) for p in people]
  keys = [k for k, v in SCRATCH if v is not None]
  population = {
    "name": name,
    "source": {"path": recode.SOURCE, "sha256": source_sha256},
    "recode": {"path": os.path.relpath(recode.TABLE, recode.LAB), "sha256": table_sha256},
    "seed": built["seed"], "weight": WEIGHT,
    "draw": {"n": built["n"], "respondents": built["respondents"], "eligible": built["eligible"],
             "rule": "at least %d scorable items" % built["least"],
             "pilot": 20, "gpt-4o pilot": 10},
    "slice": {"k": built["k"], "rule": "uniform over scorable items, seed '<seed>:<id>'"},
    "nodes": {"importance": IMPORTANCE, "min": min(nodes_per), "max": max(nodes_per),
              "embedding_model": EMBEDDING_MODEL},
    "floor": {"items": len(built["floor"]),
              "rule": "weighted mode over the %d respondents not drawn"
                      % (built["respondents"] - built["n"])},
    "dropped": {
      "respondents": {"fewer than %d scorable items" % built["least"]:
                      built["respondents"] - built["eligible"]},
      "scratch keys": {k: "a synthetic name or place is evidence the respondent never gave "
                          "" for k, v in SCRATCH if v is None},
      "scratch values": {k: sum(k not in p["scratch"] for p in people) for k in keys
                         if any(k not in p["scratch"] for p in people)},
      "items": {r.item_id: r.status.split(": ", 1)[1] for r in rows if r.kind == recode.NONE},
    },
    "files": {name_: recode.sha256_of(os.path.join(out, name_))
              for name_ in SAME + (EXTRA if withhold or every else ())
              + tuple(sorted(f for f in maps.values() if f not in EXTRA))},
    "respondents": [
      {"id": p["id"], "position": p["position"], "weight": p["weight"], "form": p["form"],
       "mode": p["mode"], "scorable": p["scorable"], "answered": p["answered"],
       "nodes": len(p[memory]), "agents": agent_ids(p["id"])} for p in people],
  }
  if withhold:
    population["withhold"] = {
      "plain": withhold,
      "rule": "when a survey agent is asked an item of a family other than 'single', every "
              "node of that family is withheld for that one item",
      "records outside the universe": sum(len(p["outside"]) for p in people)}
  if every:
    population["maps"] = {level: maps[level] for level in
                          [EVERY] + ["k%d" % k for k in LEVELS]}
    population["every"] = {
      "plain": every,
      "rule": "the survey agent holds every answer, the held-out ones included; level every "
              "withholds the asked item's own node and every node of its family, and level k "
              "all but the first k nodes of the respondent's one shuffle that are neither "
              "",
      "levels": [EVERY] + ["k%d" % k for k in LEVELS],
      "shuffle_seed": CONTEXT % (built["seed"], "<id>"),
      "records outside the universe": sum(len(p["outside"]) for p in people)}
  _dump(os.path.join(out, "population.json"), population)
  return population, vectors


# --- the 2022 file --------------------------------------------------------------

def value_labels(path):
  """{variable: {code: label}} for the scratch's variables, from the .dta's label sets."""
  import pandas as pd
  wanted = {v for _, v in SCRATCH if v} | set(SCRATCH_ALSO)
  with pd.io.stata.StataReader(path) as reader:
    reader.read(convert_categoricals=False)
    sets = reader.value_labels()
    names = dict(zip(reader._varlist, reader._lbllist))
  return {v: {int(c): s for c, s in sets.get(names.get(v), {}).items() if 0 <= c < 1000}
          for v in wanted}


def columns(rows):
  wanted = (["id", "form", "mode", WEIGHT] + [v for r in rows for v in r.variables]
            + [v for _, v in SCRATCH if v] + list(SCRATCH_ALSO) + list(UNIVERSE_ALSO))
  return list(dict.fromkeys(wanted))


def fork_embedder():
  """The fork's `get_text_embedding`, with the keys of .env, and a counter of its calls."""
  if recode.FORK not in sys.path:
    sys.path.insert(0, recode.FORK)
  from lab import check_ready, trace
  check_ready.load_env()
  check_ready.require("cot")  # OPENROUTER_API_KEY, the key the embeddings need
  from simulation_engine.gpt_structure import get_text_embedding
  spent = Counter()
  lock = threading.Lock()

  def count(record):
    if record["event"] == "call":
      with lock:
        spent["calls"] += 1
        spent["tokens"] += record.get("tokens_in") or 0
    elif record["event"] in ("rate_limit", "error"):
      with lock:
        spent[record["event"]] += 1

  trace.SINKS.append(count)
  return get_text_embedding, spent


def main(argv):
  parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
  parser.add_argument("--dry", action="store_true", help="plan and print; write nothing")
  parser.add_argument("--name", default=NAME)
  parser.add_argument("--n", type=int, default=DRAW,
                      help="respondents to draw (default %(default)s, as gss22-a was built)")
  parser.add_argument("--withhold-families", default=None, metavar="PLAIN",
                      help="add withhold.json and universe.json to a copy of the population "
                           "PLAIN, built plain from the same seed and size")
  parser.add_argument("--every-answer", default=None, metavar="PLAIN",
                      help="every answer in the survey agents' memory and a map per level, "
                           "against PLAIN, built from the same seed and size")
  args = parser.parse_args(argv)
  rows = recode.validate(recode.load(), recode.bank_items())
  bank = recode.bank_items()
  path = os.path.join(recode.lab_data(), recode.SOURCE)
  frame = recode.read_source(path)
  records = frame[columns(rows)].to_dict("records")
  built = plan(records, rows, bank, value_labels(path), vocabulary(), n=args.n,
               every=bool(args.every_answer))
  people = built["people"]
  memory = "every_nodes" if args.every_answer else "nodes"
  nodes = [len(p[memory]) for p in people]
  texts = {node["content"] for p in people for _, node in p[memory]}
  print("%d respondents, %d eligible (at least %d scorable items); drew %d with seed %d"
        % (built["respondents"], built["eligible"], built["least"], built["n"], built["seed"]))
  print("survey nodes per agent: min %d, median %d, max %d; %d distinct node texts to embed"
        % (min(nodes), sorted(nodes)[len(nodes) // 2], max(nodes), len(texts)))
  print("floor: %d items, over the %d respondents not drawn"
        % (len(built["floor"]), built["respondents"] - built["n"]))
  print("records outside an item's universe: %d per agent type"
        % sum(len(p["outside"]) for p in people))
  if args.dry:
    return
  out = os.path.join(recode.lab_data(), "populations", args.name)
  started = datetime.datetime.now().astimezone()
  if args.withhold_families:
    plain = os.path.join(recode.lab_data(), "populations", args.withhold_families)
    population, _ = write(built, rows, bank, out, None, recode.SOURCE_SHA256,
                          recode.sha256_of(recode.TABLE), args.name, plain=plain)
    spent, route = Counter(), "copied from %s" % args.withhold_families
  elif args.every_answer:
    plain = os.path.join(recode.lab_data(), "populations", args.every_answer)
    embed, spent = fork_embedder()
    population, _ = write(built, rows, bank, out, embed, recode.SOURCE_SHA256,
                          recode.sha256_of(recode.TABLE), args.name, every=plain)
    route = "copied from %s, and openrouter for a text it lacks" % args.every_answer
  else:
    embed, spent = fork_embedder()
    population, _ = write(built, rows, bank, out, embed, recode.SOURCE_SHA256,
                          recode.sha256_of(recode.TABLE), args.name)
    route = "openrouter"
  usd = spent["tokens"] * EMBEDDING_USD_PER_MTOK / 1e6
  _dump(os.path.join(out, "build.json"), {
    "built_at": started.isoformat(timespec="seconds"),
    "embeddings": {"route": route, "model": EMBEDDING_MODEL, "texts": len(texts),
                   "calls": spent["calls"], "tokens": spent["tokens"],
                   "rate_limits": spent["rate_limit"], "errors": spent["error"],
                   "usd_at_list_price": round(usd, 6)}})
  print("wrote %s: %d agents; embeddings %d calls, %d tokens, US$%.4f at list price"
        % (out, 2 * len(people), spent["calls"], spent["tokens"], usd))
  print("items.json %s  truth.jsonl %s" % (population["files"]["items.json"][:12],
                                           population["files"]["truth.jsonl"][:12]))


if __name__ == "__main__":
  main(sys.argv[1:])
