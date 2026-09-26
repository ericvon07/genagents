"""The item bank: the paper's three instruments, as the question dicts the seam takes (003-IP step 5).

The bank is the authors' own, `<osf>/figure2/data/question_master/`: `gss/main.csv` (177
items), `big_five/main.csv` (the BFI-44) and `econ_games/main.csv` (five games), each row a
question number, an id, the question's text and its options as JSON. Nothing is edited on
the way in: the text is the prompt's `Q:` line and the options its `Option:` or `Range:`
line, verbatim.

- GSS: the 150 items the paper evaluates, the 177 less `EXCLUDED_QS` below, copied from
  `<osf>/figure2/code/source/new_analysis/analyze_gss_filtered.py:29` (a `*` there is part
  of the item's id, not a wildcard). All categorical.
- BFI-44: 44 categorical items on the same five-point agreement scale.
- Economic games: games 2, 3 and 5 are categorical, games 1 and 4 numerical with decimals
  (`float_resp`). All five are `flagged` (JE-K: the typed model's weak ground, asked last).
  Game 4's range is `[0, 40]` in the bank while its text asks for 0.00 to 4.00 and the
  authors' analysis reads it on (0, 4); the bank is kept as written (GT-3).

`batches` cuts the items into the requests the runner sends: in bank order, never across an
instrument or a change of kind, at most `size` items each (DS-H). Each item's `id` is
`<instrument>:<the bank's id>`, which is what a prediction record carries.
"""
import csv
import hashlib
import json
import os

FORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANK = os.path.join(os.path.dirname(FORK), "osf-replication", "figure2", "data", "question_master")

INSTRUMENTS = ("gss", "bfi44", "econ")
FILES = {"gss": "gss/main.csv", "bfi44": "big_five/main.csv", "econ": "econ_games/main.csv"}

# <osf>/figure2/code/source/new_analysis/analyze_gss_filtered.py:29, EXCLUDED_QS, verbatim.
EXCLUDED_QS = [
  "BORN", "DEGREE*", "DWELOWN", "EDUC*", "FAMDIF16", "HISPANIC", "MADEG*", "MAEDUC*",
  "MARITAL", "MARTYPE*", "MAWRKGRW", "PADEG*", "PAEDUC*", "PARTYID", "RACE*", "REG16",
  "RELIG*", "RELPERSN", "RVISITOR", "SEX*", "SPEDUC*", "SPRTPRSN", "SPWRKSTA", "VETYEARS",
  "VISITORS", "WIDOWED", "ZODIAC",
]


class Item(dict):
  """id, instrument, text, options, kind, float_resp, flagged."""

  def __getattr__(self, name):
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)


def _rows(bank, instrument):
  with open(os.path.join(bank, FILES[instrument]), encoding="utf-8-sig", newline="") as f:
    return list(csv.DictReader(f))


def load(instruments=INSTRUMENTS, bank=BANK):
  """The items of `instruments`, in the order given and each in bank order."""
  items = []
  for instrument in instruments:
    if instrument not in FILES:
      raise ValueError("instrument %r; known: %s" % (instrument, ", ".join(INSTRUMENTS)))
    for row in _rows(bank, instrument):
      qid = row["Question ID"]
      if instrument == "gss" and qid in EXCLUDED_QS:
        continue
      kind = (row.get("Type") or "categorical").strip()
      items.append(Item(id="%s:%s" % (instrument, qid), instrument=instrument,
                        text=row["Question"], options=json.loads(row["Options"]), kind=kind,
                        float_resp=kind == "numerical", flagged=instrument == "econ"))
  return items


def bank_sha1(instruments=INSTRUMENTS, bank=BANK):
  """A fingerprint of the files read and the exclusion list applied."""
  h = hashlib.sha1()
  for instrument in instruments:
    with open(os.path.join(bank, FILES[instrument]), "rb") as f:
      h.update(FILES[instrument].encode() + b"\0" + f.read())
  h.update(json.dumps(EXCLUDED_QS).encode())
  return h.hexdigest()


def batches(items, size):
  """Bank order, cut at every change of instrument or kind, then into chunks of `size`."""
  out, run = [], []
  for item in items:
    if run and (item.instrument, item.kind) != (run[-1].instrument, run[-1].kind):
      out += [run[i:i + size] for i in range(0, len(run), size)]
      run = []
    run.append(item)
  out += [run[i:i + size] for i in range(0, len(run), size)]
  return out


def questions(batch):
  """The dict the seam takes, {text: options}; two items with the same text cannot share one."""
  texts = [item.text for item in batch]
  if len(set(texts)) != len(texts):
    raise ValueError("a batch repeats a question's text: %r" % [i.id for i in batch])
  return {item.text: item.options for item in batch}
