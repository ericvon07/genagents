"""Read where a judge wins and loses by kind of item: options, consensus, partisanship.

Usage (from the repository root, with numpy installed):

  python analysis/categories.py                  # the gss22-c pair, as the snapshot read it
  python analysis/categories.py --boot 0         # the same without the intervals
  python analysis/categories.py --jev gss22-c-jev --gemma gss22-c-cot-gemma \
      --siblings gss22-b-jev

An exploratory reading, not a rule: none of its cuts was fixed before the runs were read. It reads the survey agents of two registered
runs over one built population (Jev and a chain of thought, the same records, the same
state), and a third run of Jev with the siblings in memory for the pick mixes only. It
prints Markdown, aggregates only: no respondent id, no row.

What it counts, and how:

- **Records.** Survey agents only. The trap records of the population's `universe.json` are
  left out. A record is right when `predicted` equals the respondent's truth
  string; the floor is right when the floor's option does. Raw accuracy only.
- **An item's features**, all over the survey records of that item in this population: the
  number of options; the **concentration**, the share of respondents giving the item's most
  common answer; the **polarisation**, Cramér's V between the respondent's party (Democrat,
  Republican or neither, read from the demographic scratch: any label naming the party,
  leaners included) and the answer. Items with fewer than `--least` survey records (50) are
  left out of the cuts by concentration and polarisation, whose tertiles are over items.
- **The intervals** are 95%, over `--boot` resamples (10,000) of the respondents, each
  carrying all of its survey records, numpy seed 20260926: the readout's bootstrap, this script's own
  draws. The groups (tertiles, the worst items) are fixed from the full sample before the
  resampling and are not redrawn, so an interval does not carry the choice of the group.
- **The judge's own majority guess** is, per item, the option with the highest mean of
  Jev's distributions over that item's survey agents (for the chain of thought, its most
  frequent pick). It uses no truth, but it is in-sample: every respondent's distribution
  enters the guess that is then scored on that respondent.
- **Interior options** are every option but the first and the last, on items of three to
  five options with no "Other" or "Inapplicable" option.
"""

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAB = Path(__file__).resolve().parent.parent
SEED = 20260926
EXAMPLES = ("gss:NATHEAL/Y:Too much", "gss:NATEDUC/Y:Too much")


def lab_data():
  value = os.environ.get("LAB_DATA")
  if not value and (LAB / ".env").exists():
    for line in (LAB / ".env").read_text().splitlines():
      if line.startswith("LAB_DATA="):
        value = line.split("=", 1)[1].strip()
  if not value:
    return LAB / "data"
  return Path(value)


def sha256_of(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def population_of(manifest):
  data = manifest["data"]
  folder = Path(data["population_path"].replace("<lab-data>", str(lab_data())))
  for name, key in (("population.json", "population_json_sha256"), ("truth.jsonl", "truth_sha256")):
    if sha256_of(folder / name) != data[key]:
      sys.exit("%s/%s is not the file %s recorded" % (folder, name, manifest["name"]))
  pop = json.loads((folder / "population.json").read_text())
  trap = set()
  if (folder / "universe.json").exists():
    if sha256_of(folder / "universe.json") != pop["files"]["universe.json"]:
      sys.exit("%s/universe.json is not the file population.json recorded" % folder)
    universe = json.loads((folder / "universe.json").read_text())
    trap = {(r["respondent"], r["item"]) for r in universe["records"]}
  truth = {}
  for line in (folder / "truth.jsonl").open():
    r = json.loads(line)
    truth[(r["respondent"], r["item"])] = r
  party = {}
  for r in pop["respondents"]:
    label = json.loads((folder / "agents" / r["agents"]["demographic"] / "scratch.json")
                       .read_text()).get("party_identification") or ""
    label = label.lower()
    party[r["id"]] = "D" if "democrat" in label else "R" if "republican" in label else "I"
  return folder, truth, trap, party


def load(name, truth, trap, party):
  folder = LAB / "runs" / name
  if not (folder / "manifest.json").exists():
    sys.exit("%s is not a registered run" % name)
  rows = []
  with gzip.open(folder / "predictions.jsonl.gz", "rt") as f:
    for line in f:
      r = json.loads(line)
      if r["agent_type"] != "survey":
        continue
      rid = r["agent"].rsplit("-", 1)[0]
      if (rid, r["item"]) in trap:
        continue
      t = truth[(rid, r["item"])]
      rows.append(dict(rid=rid, item=r["item"], family=t["family"], options=tuple(r["options"]),
                       pred=r["predicted"], truth=t["truth"], floor=t["floor"], p=r.get("p"),
                       state=r["state_sha1"], party=party[rid]))
  d = pd.DataFrame(rows).set_index(["rid", "item"]).sort_index()
  d["right"] = (d.pred == d.truth).astype(float)
  d["floor_right"] = (d.floor == d.truth).astype(float)
  return d


def cramers_v(a, b):
  O = pd.crosstab(a, b).values.astype(float)
  if min(O.shape) < 2:
    return 0.0
  n = O.sum()
  E = O.sum(1, keepdims=True) * O.sum(0, keepdims=True) / n
  return math.sqrt(((O - E) ** 2 / E).sum() / (n * (min(O.shape) - 1)))


class Boot:
  """Per-respondent column sums, resampled once and shared by every statistic."""

  def __init__(self, rids, resamples, seed=SEED):
    self.rids = list(rids)
    self.at = {r: i for i, r in enumerate(self.rids)}
    self.cols = {}
    self.resamples = resamples
    self.seed = seed
    self._totals = None

  def column(self, name, rid, value):
    v = np.zeros(len(self.rids))
    np.add.at(v, [self.at[r] for r in rid], np.asarray(value, float))
    self.cols[name] = v
    self._totals = None
    return name

  def counts(self):
    """How many times each respondent is drawn in each resample, drawn once."""
    if not hasattr(self, "_counts"):
      rng = np.random.default_rng(self.seed)
      R = len(self.rids)
      self._counts = np.empty((self.resamples, R))
      for b in range(self.resamples):
        self._counts[b] = np.bincount(rng.integers(0, R, R), minlength=R)
    return self._counts

  def totals(self):
    if self._totals is None:
      names = list(self.cols)
      T = self.counts() @ np.column_stack([self.cols[n] for n in names])
      self._totals = {n: T[:, i] for i, n in enumerate(names)}
    return self._totals

  def interval(self, statistic):
    if not self.resamples:
      return None
    s = statistic(self.totals())
    return np.percentile(s, 2.5), np.percentile(s, 97.5)


def pts(x):
  return "%+.2f" % (100 * x)


def with_interval(point, iv):
  return pts(point) if iv is None else "%s [%s, %s]" % (pts(point), pts(iv[0]), pts(iv[1]))


def lift_cells(boot, tag, mask, judges):
  """The floor, each judge's raw accuracy and judge minus floor with its interval."""
  ref = next(iter(judges.values()))
  sub = ref[mask]
  n = boot.column(tag + ":n", sub.index.get_level_values(0), np.ones(len(sub)))
  fl = boot.column(tag + ":floor", sub.index.get_level_values(0), sub.floor_right)
  cells = [str(len(sub)), "%.3f" % sub.floor_right.mean()]
  lifts = []
  for name, d in judges.items():
    s = d[mask]
    c = boot.column(tag + ":" + name, s.index.get_level_values(0), s.right)
    lifts.append((c, s.right.mean() - sub.floor_right.mean()))
    cells.append("%.3f" % s.right.mean())
  return cells, [(c, point, n, fl) for c, point in lifts]


def finish(boot, pending):
  return [with_interval(point, boot.interval(lambda t, c=c, n=n, fl=fl: (t[c] - t[fl]) / t[n]))
          for c, point, n, fl in pending]


def table(header, rows):
  out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
  out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
  return "\n".join(out)


def mix(series):
  return series.value_counts(normalize=True)


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--jev", default="gss22-c-jev")
  ap.add_argument("--gemma", default="gss22-c-cot-gemma")
  ap.add_argument("--siblings", default="gss22-b-jev",
                  help="Jev with the siblings in memory, for the pick mixes only")
  ap.add_argument("--boot", type=int, default=10000)
  ap.add_argument("--least", type=int, default=50)
  ap.add_argument("--worst", type=int, default=10)
  ap.add_argument("--example", action="append", help="ITEM:OPTION, the option's share by party")
  args = ap.parse_args()

  manifests = {n: json.loads((LAB / "runs" / n / "manifest.json").read_text())
               for n in (args.jev, args.gemma)}
  if len({m["data"]["truth_sha256"] for m in manifests.values()}) != 1:
    sys.exit("the two judges were not run over one population")
  folder, truth, trap, party = population_of(manifests[args.jev])
  jev = load(args.jev, truth, trap, party)
  gem = load(args.gemma, truth, trap, party).reindex(jev.index)
  if gem.pred.isna().any() or (gem.state != jev.state).any():
    sys.exit("the judges do not hold the same records over the same state")
  sib_manifest = json.loads((LAB / "runs" / args.siblings / "manifest.json").read_text())
  _, sib_truth, _, _ = population_of(sib_manifest)
  sib = load(args.siblings, sib_truth, trap, party).reindex(jev.index)
  judges = {"Jev": jev, "Gemma": gem}
  rid = jev.index.get_level_values(0)
  item = jev.index.get_level_values(1)
  boot = Boot(sorted(set(rid)), args.boot)

  # Item features, over the survey records of each item.
  f = []
  for it, g in jev.groupby(level=1):
    counts = g.truth.value_counts()
    jp = pd.DataFrame([p for p in g.p if p]).mean()
    f.append(dict(item=it, family=g.family.iloc[0], n=len(g), options=len(g.options.iloc[0]),
                  top=counts.iloc[0] / counts.sum(), V=cramers_v(g.party, g.truth),
                  floor=g.floor.iloc[0], jev_guess=jp.idxmax(),
                  gem_guess=gem.loc[g.index].pred.value_counts().idxmax()))
  f = pd.DataFrame(f).set_index("item")
  for name, d in judges.items():
    f[name] = d.groupby(level=1).right.mean()
  f["floor_acc"] = jev.groupby(level=1).floor_right.mean()
  f["gap"] = f.Jev - f.Gemma
  kept = f[f.n >= args.least].copy()
  kept["conc"] = pd.qcut(kept.top, 3, labels=["spread out", "middle", "consensus"])
  kept["pol"] = pd.qcut(kept.V, 3, labels=["least partisan", "middle", "most partisan"])
  in_kept = item.isin(kept.index)

  out = ["# Where the judges win and lose, by kind of item",
         "",
         "Survey agents of `%s` and `%s` over `%s`, %d records, %d respondents, %d items; "
         "trap records out. Raw accuracy. Exploratory: no cut was fixed before the runs. "
         "Intervals: 95%%, %d resamples of respondents, seed %d."
         % (args.jev, args.gemma, folder.name, len(jev), len(boot.rids), len(f), args.boot, SEED),
         ""]

  # 1. By number of options.
  rows, pend = [], []
  bands = [("2", 2, 2), ("3", 3, 3), ("4", 4, 4), ("5", 5, 5), ("6 or more", 6, 99)]
  nopt = jev.options.map(len).values
  for label, lo, hi in bands:
    mask = (nopt >= lo) & (nopt <= hi)
    cells, p = lift_cells(boot, "opt" + label, mask, judges)
    rows.append([label, (f.options.between(lo, hi)).sum()] + cells)
    pend.append(p)
  rows = [r + finish(boot, p) for r, p in zip(rows, pend)]
  out += ["## By number of options", "",
          table(["options", "items", "records", "floor", "Jev", "Gemma", "Jev − floor",
                 "Gemma − floor"], rows), ""]

  # 2 and 3. By concentration and by polarisation, items with enough records.
  for key, title, feature in (("conc", "By how much people agree (share of the most common answer)", "top"),
                              ("pol", "By how partisan the item is (Cramér's V, party by answer)", "V")):
    rows, pend = [], []
    for level in kept[key].cat.categories:
      its = kept.index[kept[key] == level]
      mask = item.isin(its)
      cells, p = lift_cells(boot, key + level, mask, judges)
      rows.append([level, len(its), "%.2f" % kept.loc[its, feature].mean()] + cells)
      pend.append(p)
    rows = [r + finish(boot, p) for r, p in zip(rows, pend)]
    out += ["## " + title, "",
            "Items with at least %d survey records (%d of %d); tertiles over items."
            % (args.least, len(kept), len(f)), "",
            table(["items", "count", "mean " + feature, "records", "floor", "Jev", "Gemma",
                   "Jev − floor", "Gemma − floor"], rows), ""]

  # How often each judge picks the majority answer, and what leaving it buys.
  rows = []
  for level in kept.conc.cat.categories:
    mask = item.isin(kept.index[kept.conc == level])
    sub = jev[mask]
    row = [level, "%.3f" % sub.floor_right.mean()]
    for d in judges.values():
      row.append("%.3f" % (d[mask].pred == d[mask].floor).mean())
    for d in judges.values():
      s = d[mask]
      left = s[s.pred != s.floor]
      row.append("%.3f / %.3f" % (left.right.mean(), left.floor_right.mean()))
    rows.append(row)
  out += ["## The majority answer: how often given, how often picked, what leaving it buys", "",
          "*left: right / floor right* is, over the records where the judge's pick is not the "
          "floor's option, how often the pick is right and how often the floor would have been.",
          "",
          table(["items", "people give it", "Jev picks it", "Gemma picks it",
                 "Jev left: right / floor right", "Gemma left: right / floor right"], rows), ""]

  # 4. By party, on the most partisan tertile against the rest.
  rows, pend = [], []
  most = item.isin(kept.index[kept.pol == "most partisan"])
  for label, zone in (("most partisan items", most), ("the other items", in_kept & ~most)):
    for p, pname in (("D", "Democrats"), ("I", "neither"), ("R", "Republicans")):
      mask = zone & (jev.party.values == p)
      cells, pe = lift_cells(boot, "party" + label + p, mask, judges)
      rows.append([label, pname] + cells)
      pend.append(pe)
  rows = [r + finish(boot, p) for r, p in zip(rows, pend)]
  out += ["## By the respondent's party", "",
          table(["items", "party", "records", "floor", "Jev", "Gemma", "Jev − floor",
                 "Gemma − floor"], rows), ""]
  rows = []
  for spec in args.example or EXAMPLES:
    it, option = spec.rsplit(":", 1)
    g = jev.xs(it, level=1, drop_level=False)
    for p, pname in (("D", "Democrats"), ("I", "neither"), ("R", "Republicans")):
      s = g[g.party == p]
      rows.append([it.replace("gss:", ""), option, pname, len(s), "%.2f" % (s.truth == option).mean(),
                   "%.2f" % (s.pred == option).mean(),
                   "%.2f" % (gem.loc[s.index].pred == option).mean()])
  out += ["**One option's share by party** (survey agents, `%s`):" % folder.name, "",
          table(["item", "option", "party", "records", "people", "Jev", "Gemma"], rows), ""]

  # 5. Jev against Gemma: where the gap lives.
  contrib = (kept.n * kept.gap).sort_values() / len(jev)
  worst = contrib.index[: args.worst]
  in_worst = item.isin(worst)
  n_all = boot.column("gap:n", rid, np.ones(len(jev)))
  j_all = boot.column("gap:jev", rid, jev.right)
  g_all = boot.column("gap:gem", rid, gem.right)
  n_w = boot.column("gap:nw", rid, in_worst)
  j_w = boot.column("gap:jw", rid, jev.right * in_worst)
  g_w = boot.column("gap:gw", rid, gem.right * in_worst)
  overall = jev.right.mean() - gem.right.mean()
  carried = (jev.right[in_worst].sum() - gem.right[in_worst].sum()) / len(jev)
  rest = jev.right[~in_worst].mean() - gem.right[~in_worst].mean()
  iv_all = boot.interval(lambda t: (t[j_all] - t[g_all]) / t[n_all])
  iv_rest = boot.interval(lambda t: (t[j_all] - t[j_w] - t[g_all] + t[g_w]) / (t[n_all] - t[n_w]))
  out += ["## Jev against Gemma: where the gap lives", "",
          "- Jev − Gemma, every record: %s" % with_interval(overall, iv_all),
          "- carried by the %d items that cost Jev most (chosen on these data, so no interval): "
          "%s points of it" % (args.worst, pts(carried)),
          "- Jev − Gemma on the other %d items: %s" % (len(f) - args.worst, with_interval(rest, iv_rest)),
          "- items (of %d with at least %d records) where Jev is ahead: %d; level: %d"
          % (len(kept), args.least, (kept.gap > 0).sum(), (kept.gap == 0).sum()), ""]
  rows = []
  for it in worst:
    r = kept.loc[it]
    g = jev.xs(it, level=1, drop_level=False)
    people, mj, ms, mg = mix(g.truth), mix(g.pred), mix(sib.loc[g.index].pred), mix(gem.loc[g.index].pred)
    top = people.index[0]
    jtop = mj.index[0]
    rows.append([it.replace("gss:", ""), r.family, r.n, r.options, "%.3f" % r.floor_acc,
                 "%.3f" % r.Jev, "%.3f" % r.Gemma,
                 "%s: %.2f / %.2f / %.2f / %.2f" % (top, people[top], mj.get(top, 0), ms.get(top, 0), mg.get(top, 0)),
                 "%s: %.2f / %.2f / %.2f / %.2f" % (jtop, people.get(jtop, 0), mj[jtop], ms.get(jtop, 0), mg.get(jtop, 0))])
  out += ["**The items that cost Jev most**, with two options' shares as *people / Jev / Jev "
          "with siblings (`%s`) / Gemma*: the people's most common answer, and Jev's." % args.siblings, "",
          table(["item", "family", "records", "options", "floor", "Jev", "Gemma",
                 "people's top answer", "Jev's top pick"], rows), ""]

  def tv(d):
    return pd.Series({it: 0.5 * mix(g.truth).subtract(mix(g.pred), fill_value=0).abs().sum()
                      for it, g in d.groupby(level=1)})
  tvj, tvg = tv(jev).loc[kept.index], tv(gem).loc[kept.index]
  inner_rows = []
  ordered = [it for it in f.index
             if 3 <= f.loc[it, "options"] <= 5
             and not any(("Other" in o) or ("Inapplicable" in o) for o in jev.xs(it, level=1).options.iloc[0])]
  om = item.isin(ordered)
  interior = np.array([p in o[1:-1] for p, o in zip(jev.truth, jev.options)])
  inner_rows.append(["people", "%.3f" % interior[om].mean()])
  for name, d in judges.items():
    inner_rows.append([name, "%.3f" % np.array([p in o[1:-1] for p, o in zip(d.pred, d.options)])[om].mean()])
  out += ["**How far each judge's mix of picks is from the people's**, per item (total "
          "variation, 0 is the same mix), weighted by records: Jev %.3f, Gemma %.3f; its "
          "correlation over items with Jev − Gemma: %.2f."
          % (np.average(tvj, weights=kept.n), np.average(tvg, weights=kept.n),
             np.corrcoef(kept.gap, tvj - tvg)[0, 1]), "",
          "**Interior options** (%d ordered items of three to five options, %d records): the "
          "share of answers that are neither the first nor the last option." % (len(ordered), om.sum()), "",
          table(["who", "interior share"], inner_rows), ""]

  # 6. Does a judge know the majority answer without being told?
  rows = []
  for level in list(kept.conc.cat.categories) + ["all"]:
    its = kept.index if level == "all" else kept.index[kept.conc == level]
    rows.append([level, len(its), "%.2f" % (kept.loc[its, "jev_guess"] == kept.loc[its, "floor"]).mean(),
                 "%.2f" % (kept.loc[its, "gem_guess"] == kept.loc[its, "floor"]).mean()])
  guess_right = (jev.truth.values == f.loc[item, "jev_guess"].values).astype(float)
  m = in_kept
  n_k = boot.column("g:n", rid[m], np.ones(m.sum()))
  gr = boot.column("g:guess", rid[m], guess_right[m])
  jr = boot.column("g:jev", rid[m], jev.right[m])
  fr = boot.column("g:floor", rid[m], jev.floor_right[m])
  out += ["## Does a judge know the majority answer without being told?", "",
          "Neither judge is ever given the floor: it is the 2022 weighted majority over the "
          "respondents not drawn. The share of items where the judge's own majority guess "
          "is the floor's option:", "",
          table(["items", "count", "Jev (mean distribution)", "Gemma (most frequent pick)"], rows), "",
          "**Giving everyone Jev's own majority guess** (in-sample, no truth), over the %d "
          "records of these items: %.3f; Jev as it ran %.3f; the floor %.3f."
          % (m.sum(), guess_right[m].mean(), jev.right[m].mean(), jev.floor_right[m].mean()),
          "- Jev as it ran − its own guess: %s"
          % with_interval(jev.right[m].mean() - guess_right[m].mean(),
                          boot.interval(lambda t: (t[jr] - t[gr]) / t[n_k])),
          "- the floor − Jev's own guess: %s"
          % with_interval(jev.floor_right[m].mean() - guess_right[m].mean(),
                          boot.interval(lambda t: (t[fr] - t[gr]) / t[n_k])), ""]

  # 7. Jev's calibration on these records.
  conf = np.array([max(p.values()) if p else np.nan for p in jev.p])
  bins = pd.cut(conf, [0, .4, .5, .6, .7, .8, .9, 1.0001])
  cal = pd.DataFrame(dict(conf=conf, right=jev.right.values, bin=bins)).groupby("bin", observed=True)
  out += ["## Jev's confidence against its accuracy", "",
          table(["Jev's probability on its pick", "records", "mean probability", "right"],
                [[str(b), len(g), "%.3f" % g.conf.mean(), "%.3f" % g.right.mean()] for b, g in cal]), ""]

  # 8. Item-level correlations, weighted by records.
  def wcorr(x, y, w):
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    return np.average((x - mx) * (y - my), weights=w) / math.sqrt(
      np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))
  rows = []
  for yname, y in (("Jev", kept.Jev), ("Gemma", kept.Gemma), ("Jev − floor", kept.Jev - kept.floor_acc),
                   ("Gemma − floor", kept.Gemma - kept.floor_acc), ("Jev − Gemma", kept.gap)):
    rows.append([yname] + ["%.2f" % wcorr(kept[x].astype(float), y, kept.n) for x in ("options", "top", "V")])
  out += ["## Over items, weighted by records: correlations", "",
          table(["accuracy of", "options", "concentration", "polarisation"], rows), ""]

  print("\n".join(out))


if __name__ == "__main__":
  main()
