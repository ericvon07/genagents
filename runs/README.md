# Runs

The runs behind the survey results in version 20 of the System One essay (October 2026), with
the people they predicted and the code that scores them, so that every number can be
recomputed.

## What is here

- `runs/<name>/`: one prediction run each. `manifest.json` records the condition, the code
  (a commit of this repository), the population, the seed, the prices and the sha1 of the two
  data files: `predictions.jsonl.gz`, one line per person and held-out question with the
  prediction (and, for Jev, the probabilities), and `trace.jsonl.gz`, every request and
  decision.
- `data/gss22-*.tar.gz`: the four populations the runs predicted, built from respondents of
  the 2022 General Social Survey (NORC at the University of Chicago, gss.norc.org). Each holds
  `truth.jsonl`, the respondents' own answers to their 30 held-out questions with the majority
  answer beside each; `items.json`, the questions each agent was asked; the agents' profiles
  (`scratch.json`) and memories (`memory_stream/nodes.json`, the person's other answers); and
  the build's records (`population.json`, `build.json`, and, where a population hides
  questions, `withhold*.json` and `universe.json`). The embeddings of the memories are left
  out: the readers do not use them, and re-running a prediction means rebuilding the
  population with `analysis/gss/build_population.py`, which asks for them again.
- `analysis/`: `readout.py` (accuracy, the majority guess, agreement between runs, the
  intervals, cost), `categories.py` (the topic readings) and `memory_curve_extras.py` (the
  memory curve and the direct answers), and `analysis/gss/`, the recode table and the builder
  that turns the 2022 survey file into a population.

`items.json`, `truth.jsonl`, the agents' files and every file under `runs/` are byte for byte
what the runs used, and the readers check their hashes. `population.json` and
`universe.json` had internal reference codes removed from their descriptive text, and the
hashes that point to them were updated to match.

## The populations

| Population | People | What their memory holds | Runs |
|---|---|---|---|
| `gss22-a` | the first 100 drawn | their other answers, related questions included | `gss22-jev`, `gss22-jev-r`, `gss22-cot-gemma`, `gss22-cot-4o`, `gss22-cot-4o-2` |
| `gss22-b` | 1,052, whose first 100 are `gss22-a`'s | the same | `gss22-b-jev`, `gss22-b-cot-gemma` |
| `gss22-c` | the same 1,052 | each question's related questions hidden while it is asked | `gss22-c-jev`, `gss22-c-cot-gemma`, `gss22-c-direct-gemma` |
| `gss22-d` | the same 1,052 | related questions hidden, and 10, 30 or every other answer kept | `gss22-d-jev-k10`, `gss22-d-jev-k30`, `gss22-d-jev-every` |

Every person has two agents: `demographic`, the profile alone, and `survey`, the profile plus
other answers. `-cot-` runs are a language model writing its reasoning before answering, as
the paper's code does; `gss22-c-direct-gemma` is Gemma answering directly, at temperature 0;
`-jev` runs are Jev's probabilities, taking the most likely answer. `gss22-jev-r` and
`gss22-cot-4o-2` repeat `gss22-jev` and `gss22-cot-4o` to see how often each gives the same
answer twice.

## Recomputing the essay's numbers

From the repository root, with Python 3.9+ (numpy for `--partisans` and `categories.py`):

```sh
data/unpack.sh
# The first 100 people: Jev, Gemma and GPT-4o
python analysis/readout.py gss22-cot-gemma gss22-cot-4o gss22-b-cot-gemma --first 100 --against gss22-cot-4o
# The same answer twice
python analysis/readout.py gss22-jev gss22-jev-r
python analysis/readout.py gss22-cot-4o gss22-cot-4o-2
# All 1,052 people, and the split by party
python analysis/readout.py gss22-b-cot-gemma gss22-b-jev --against gss22-b-cot-gemma --partisans
# Related questions hidden, and what the other answers add over the profile
python analysis/readout.py gss22-c-jev gss22-c-cot-gemma --against gss22-c-cot-gemma
python analysis/readout.py gss22-c-jev gss22-c-cot-gemma --cross gss22-c-jev:survey:gss22-c-jev:demographic
python analysis/readout.py gss22-b-jev gss22-b-cot-gemma --cross gss22-b-jev:survey:gss22-b-jev:demographic
# Topic by topic, and health spending by party
python analysis/categories.py
# More answers in memory, and Gemma without its written reasoning
python analysis/memory_curve_extras.py
python analysis/readout.py gss22-c-cot-gemma gss22-c-direct-gemma --first 600 \
    --contrast gss22-c-cot-gemma:demographic:gss22-c-direct-gemma:demographic:gss22-c-cot-gemma:survey:gss22-c-direct-gemma:survey
```

`python analysis/readout.py --help` lists every reading. Accuracy here is the plain share of
answers right: the respondents are public and were never asked twice, so the paper's
adjustment for each person's own consistency cannot be applied.
