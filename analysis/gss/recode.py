"""The recode table: how a 2022 GSS answer becomes truth, memory, or nothing.

Usage (from the repository root):

  python analysis/gss/recode.py            validate the table against the bank
  python analysis/gss/recode.py --check    ... and against the 2022 file in LAB_DATA

`analysis/gss/recode_2022.csv` is the reviewed state; this module only reads it. One row per
bank item and wording, with these columns:

- `item_id` -- the bank's id (`<osf>/figure2/data/question_master/gss/main.csv`).
- `family` -- the battery the item belongs to, for the readout by family, or
  `single`.
- `status` -- one of three:
  - `scored`: the bank's wording, and codes that map one-to-one onto the bank's options.
    The answer is truth when the item is held out and a memory node otherwise.
  - `memory only: <why>`: another wording or presentation of the same item. The answer is a
    memory node, never truth.
  - `not scored: <why>`: neither truth nor memory.
- `variables` -- the 2022 variables that carry this row, space-separated. A respondent
  answers at most one variable of an item, across all its rows.
- `form` -- `X`, `Y` or `both`: the ballot form of the respondents these variables reach.
- `codes` -- `code=answer` pairs joined by ` | `. On a `scored` row the answer is one of the
  bank's options, verbatim; on a `memory only` row it is the answer's label. `-` marks a
  code that is an answer outside the bank's options, or a non-answer the file codes as a
  value (refused, don't know): neither truth nor memory. A code the table does not list is
  an error, never a silent drop.
- `text` -- the question as asked, for a `memory only` row whose wording differs from the
  bank's; empty means the bank's text.
- `note` -- the judgement, with its source.

A missing code in the file (don't know, inapplicable, skipped, no answer: the Stata missing
values, read as NaN) is neither truth nor memory.
"""
import csv
import hashlib
import math
import os
import sys
from collections import Counter, namedtuple

LAB = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLE = os.path.join(LAB, "analysis", "gss", "recode_2022.csv")
FORK = LAB
SOURCE = os.path.join("source", "2022", "GSS2022.dta")
SOURCE_SHA256 = "334ac5a7d7623e779800dedf5b1b39e3b6883c5d2bac6a26302b3b3f26773148"

SCORED, MEMORY, NONE = "scored", "memory only", "not scored"
FORMS = {"X": 1, "Y": 2, "both": None}  # the file's `form`: 1 standard <x>, 2 alternate <y>
DROP = "-"

Row = namedtuple("Row", "item_id family kind status variables form codes text note")
Answer = namedtuple("Answer", "item_id answer scorable text variable")


def _kind(status):
  for kind in (SCORED, MEMORY, NONE):
    if status == kind or status.startswith(kind + ": "):
      return kind
  raise ValueError("status %r: expected scored, 'memory only: <why>' or 'not scored: <why>'"
                   % status)


def _codes(cell):
  out = {}
  for pair in filter(None, (p.strip() for p in cell.split(" | "))):
    code, answer = pair.split("=", 1)
    if int(code) in out:
      raise ValueError("code %s appears twice in %r" % (code, cell))
    out[int(code)] = answer
  return out


def load(path=TABLE):
  """The table's rows, in file order."""
  with open(path, encoding="utf-8", newline="") as f:
    rows = []
    for r in csv.DictReader(f):
      kind = _kind(r["status"])
      if kind != NONE and r["form"] not in FORMS:
        raise ValueError("%s: form %r" % (r["item_id"], r["form"]))
      rows.append(Row(r["item_id"], r["family"], kind, r["status"], tuple(r["variables"].split()),
                      r["form"], _codes(r["codes"]), r["text"], r["note"]))
  return rows


def bank_items():
  """The 150 GSS items the paper evaluates, from the fork's own bank reader."""
  if FORK not in sys.path:
    sys.path.insert(0, FORK)
  from lab import items
  return {i.id.split(":", 1)[1]: i for i in items.load(["gss"])}


def validate(rows, bank):
  """Raise on a table that disagrees with the bank or with itself."""
  heads = Counter(r.item_id for r in rows if r.kind != MEMORY)
  wrong = sorted(q for q in set(bank) | set(heads) if heads.get(q) != 1 or q not in bank)
  if wrong:
    raise ValueError("every bank item needs exactly one scored or not-scored row: %s" % wrong)
  scored = {r.item_id for r in rows if r.kind == SCORED}
  seen = {}
  for r in rows:
    if r.kind == NONE:
      if r.variables or r.codes:
        raise ValueError("%s: a not-scored row carries no variables or codes" % r.item_id)
      continue
    if not r.variables or not r.codes:
      raise ValueError("%s: row %r has no variables or no codes" % (r.item_id, r.status))
    for v in r.variables:
      if v in seen:
        raise ValueError("variable %s is in %s and %s" % (v, seen[v], r.item_id))
      seen[v] = r.item_id
    if r.kind == MEMORY:
      if r.item_id not in scored:
        raise ValueError("%s: a memory-only row for an item that is not scored" % r.item_id)
      continue
    options = bank[r.item_id].options
    kept = [a for a in r.codes.values() if a != DROP]
    stray = [a for a in kept if a not in options]
    if stray:
      raise ValueError("%s: %r are not among the bank's options %r" % (r.item_id, stray, options))
    if len(set(kept)) != len(kept):
      raise ValueError("%s: two codes map to one option (not one-to-one)" % r.item_id)
  return rows


def _missing(x):
  return x is None or (isinstance(x, float) and math.isnan(x))


def answers(respondent, rows):
  """{item_id: Answer} for every item this respondent answered, in table order.

  `respondent` maps the file's variable names (and `form`) to codes, NaN or None for
  missing. `scorable` is true only on a `scored` row; `text` is None where the question is
  the bank's. Raises on a code the table does not list, on two variables answered for one
  item, and on a respondent outside the form a row names.
  """
  out = {}
  form = respondent.get("form")
  for r in rows:
    if r.kind == NONE:
      continue
    for v in r.variables:
      x = respondent.get(v)
      if _missing(x):
        continue
      code = int(x)
      if code not in r.codes:
        raise KeyError("%s: code %s of %s is not in the table" % (r.item_id, code, v))
      if FORMS[r.form] is not None and not _missing(form) and int(form) != FORMS[r.form]:
        raise ValueError("%s: %s is form %s, the respondent is form %s"
                         % (r.item_id, v, r.form, int(form)))
      if r.item_id in out or (r.item_id, DROP) in out:
        raise ValueError("%s: answered twice (%s)" % (r.item_id, v))
      if r.codes[code] == DROP:
        out[(r.item_id, DROP)] = None
        continue
      out[r.item_id] = Answer(r.item_id, r.codes[code], r.kind == SCORED, r.text or None, v)
  return {k: a for k, a in out.items() if a is not None}


def sha256_of(path):
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
        key, _, value = line.strip().removeprefix("export ").partition("=")
        if key.strip() == "LAB_DATA":
          return os.path.expanduser(value.strip().strip('"').strip("'"))
  return os.path.join(LAB, "data")


def read_source(path=None):
  """The 2022 file as codes, refusing a file whose sha256 is not the one recorded."""
  import pandas as pd
  path = path or os.path.join(lab_data(), SOURCE)
  got = sha256_of(path)
  if got != SOURCE_SHA256:
    raise SystemExit("%s has sha256 %s, not the recorded %s" % (path, got, SOURCE_SHA256))
  return pd.read_stata(path, convert_categoricals=False)


def check(frame, rows):
  """Walk every respondent through the table; return per-item counts, aggregates only.

  Raises on a variable the file lacks, a code the table does not list, an item answered
  twice, or a form that disagrees with its row -- all through `answers`.
  """
  absent = sorted(v for r in rows for v in r.variables if v not in frame.columns)
  if absent:
    raise ValueError("variables not in the file: %s" % absent)
  cols = ["form"] + [v for r in rows for v in r.variables]
  carriers = {}
  for r in rows:
    carriers.setdefault(r.item_id, []).extend(r.variables)
  counts = {q: Counter() for q in carriers}
  for record in frame[cols].to_dict("records"):
    got = answers(record, rows)
    for q, variables in carriers.items():
      if q in got:
        counts[q]["truth" if got[q].scorable else "memory only"] += 1
      elif any(not _missing(record[v]) for v in variables):
        counts[q]["dropped"] += 1
  return counts


def main(argv):
  rows = validate(load(), bank_items())
  kinds = Counter(r.kind for r in rows)
  print("%s: %d rows, %d items scored, %d not scored, %d memory-only rows"
        % (os.path.relpath(TABLE, LAB), len(rows), kinds[SCORED], kinds[NONE], kinds[MEMORY]))
  if "--check" not in argv:
    return
  counts = check(read_source(), rows)
  print("every respondent of the 2022 file walked the table: no unknown code, no item answered"
        " twice, no form out of place")
  print("%-11s %-18s %6s %12s %8s" % ("item", "family", "truth", "memory only", "dropped"))
  for r in rows:
    if r.kind == SCORED:
      c = counts[r.item_id]
      print("%-11s %-18s %6d %12d %8d" % (r.item_id, r.family, c["truth"], c["memory only"],
                                           c["dropped"]))


if __name__ == "__main__":
  main(sys.argv[1:])
