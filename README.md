# hontology

Define an ontology. Detect it in a live news feed. Find out honestly how well
that worked.

`hontology` is an end-to-end system for **ontology-driven event detection**: you
describe the things you care about — in your own words, through a UI — and the
pipeline finds evidence of them in the [GDELT](https://www.gdeltproject.org/)
global news feed, then scores itself against ground truth you accumulate as you
go.

> **Status: the pipeline is complete.** Define an ontology, follow the live
> GDELT feeds, filter and fetch article text, retrieve, judge, build ground
> truth, and score the result. Verified end to end against the live feed with
> a local model served by llama.cpp. The hosted provider (OpenRouter) is tested
> against a faked API only: it has not served a run yet.
>
> **In progress: does an ontology's structure help a model detect events?** A
> pre-registered comparison of judging a 44-concept supply-chain ontology flat,
> top-down through its class hierarchy, and event by event, on the same
> articles, model and leaf wording. All four arms have been judged on the first
> 320 articles of a frozen labelled sample. The reference labels are blind
> annotations by a Claude model, audited against 22 human-labelled articles,
> and the sample's stopping rule is not met yet. Only the baseline has run on
> the pilot event calendar, and none of its detections has been checked by a
> person yet. No results are published.

---

## The experiment

The question the current work answers: does a proper ontology, with event classes
in a hierarchy, detect events in the news better than a flat list of the same
concepts? It is answered with a comparison planned before any arm was compared
with the baseline; deviations from that plan are recorded as they are made:

| Arm | Judging |
|---|---|
| **Baseline** | Flat: one call per article over the leaf concepts retrieval selected, at most 500 pairs per calendar window |
| **Hierarchical** | Top-down: parent classes asked as routing questions that lean to yes, then the children of every class answered yes; leaves asked in the baseline's wording, one call per sibling set |
| **Extract** | Event by event: the article's distinct events listed once, then each event routed down the hierarchy and given at most one leaf |
| **Extract + embeddings** | As Extract, but below the top level each event is offered the leaves closest to it in embedding space |

Every arm sees the same articles, the same model and the same leaf wording. The
baseline judges the pairs its embedding cutoff selected; the top-down arms use
the cutoff only to decide which articles to read, then route to classes their
own way. In every arm a pair counts as retrieved when its class was shown to a
deciding judge call, so retrieval and the judge are scored on one definition.

They are scored two ways:

- on an event calendar of supply-chain disruptions, their precursors and quiet
  control days, where a detection counts as verified only once a person has
  confirmed it is the calendar's event, with the raw rate shown beside it;
- on a labelled sample of articles drawn in a frozen order, where an arm counts
  as better only when the paired interval on its F1 difference lies above zero.

Tokens and model time are reported beside the scores. The baseline also differs
from the other arms in its 500-pair cap, so a gain against it is not due to
structure alone.

---

## Why this exists

Most "LLM classifies documents" projects stop at a working pipeline and a
plausible-looking accuracy number. The hard part is not the classification — it
is knowing whether the number means anything. This project is built around four
claims that are easy to state and annoying to actually implement:

**1. The ontology belongs to the user, not the code.** Concepts, their
definitions, their inclusion and exclusion criteria, and how they relate — as a
class hierarchy, with typed relations between classes — are data, editable in
the UI or in an OWL tool such as Protégé. Changing what you're looking for should
not require changing the pipeline.

**2. Labels rot, and the system should notice.** A ground-truth label is an
answer to a question — "does this article evidence *this* concept, as currently
worded?" Edit the wording and the old answer may no longer apply. Every label is
stamped with the ontology version it was made against, and metrics drop stale
labels from the denominator rather than silently scoring against them. Labels
from a machine annotator are kept apart from a person's, and you choose which to
score against.

**3. Retrieval failure and judgment failure need separate numbers.** A pipeline
that misses an event because the retriever never surfaced the concept needs a
completely different fix from one whose retriever ranked it first and whose judge
then said no. Reporting a single F1 conflates them. Retrieval is scored on what each
deciding call was actually shown, recorded per article, and the judge on
precision and recall over those same pairs. Candidates are also stored as the
full pre-cutoff pool, so the embedding ranking is scored with recall@k and MRR,
and a different cutoff can be tried after the fact.

**4. A result must say exactly what produced it.** Every stage is versioned by
what decides its output: the retrieval version by the embedding settings, the
classes' wording and a fingerprint of the ranking code; the judge version by its
settings and a fingerprint of the prompt as rendered. One gap remains: a
top-down arm's routing wording is still versioned with its judge, not its
retrieval, although its run keys already treat it as retrieval. Each prompt id
is pinned to its wording, so a prompt edited in place is refused rather than
quietly changing what earlier verdicts mean, and a run is never continued under
different wording.

---

## How it works

```
GDELT v2 feeds ──▶ ingest ──▶ filter ──▶ fetch ──▶ retrieval ──▶ judge ──▶ verdicts
 events + GKG,    watermark,  feed codes  article   embeddings    LLM          │
 15-min slices    backfill,   linked to   text,     in pgvector   (local or    │
                  idempotent  the classes near-dups               hosted)      │
                                                                               ▼
        ground truth: labelled sample, label bank,      ───────────▶     evaluation
        machine annotation sets, event calendars                   per stage and per arm,
                                                                    with intervals and cost
```

**Feed codes decide what is fetched; meaning decides what is judged.** Before
anything is downloaded, the feed tags each article with event codes (CAMEO) and
themes (GKG). Linking your classes to those codes gives a cheap filter that keeps
the scrape budget for articles your ontology could plausibly match.
Near-duplicate articles are grouped so each story is judged once. Retrieval then
embeds class definitions and article text and ranks classes by similarity in
pgvector; for a top-down prompt it goes on to route the article, or its events,
down the hierarchy. Each article's retrieval is stored, with the classes each
deciding call was shown.

**Flat or hierarchical, one pipeline.** Classes can be arranged with
`subclass_of`, several parents allowed. Retrieval ranks leaf classes only, and
every class of a flat ontology is a leaf, so adding structure leaves a flat run
exactly as it was. The judge has four modes: one call per pair, one per article,
top-down through the hierarchy, or event by event.

**Both models are pluggable.** The embedding model and the decoder are picked
apart, each from Ollama or llama.cpp locally or OpenRouter hosted. A hosted model
is pinned to one host, so two calls in a run are never served by different
setups, and a hosted embedding model's vectors are cached under its host. Prompt
wording lives in a data file (`pipeline/judge/prompts.toml`), exactly as the
model sees it.

**Runs are content-addressed by stage.** A run config is sectioned
(`common` / `candidates` / `judge`) and each stage's artifact key chains to its
upstream:

```
key(candidates) = hash(candidates + relevant common + ontology_version)
key(retrieval)  = key(candidates), for a flat prompt
                = hash(key(candidates) + routing wording + routing decoder), top-down
key(judge)      = hash(judge + relevant common + ontology_version) + "_" + key(retrieval)
```

Change only the prompt and `key(candidates)` is unchanged, so the embedding
ranking is reused instead of recomputed. Change only what a top-down prompt's
deciding calls say, and `key(retrieval)` is unchanged too: the second run adopts
the first's stored retrievals and is shown exactly the same leaves.

Behavior is hashed; infrastructure is not. The same run against a local server
at a different address is *the same run*, and treating that as a new experiment
would make cross-machine comparison impossible. A hosted provider's pinned host
is the exception: hosts of one model run it at different precisions, so the
host is hashed.

---

## Quick start

Requirements: Docker, Python 3.12+, [uv](https://docs.astral.sh/uv/), and two
models, an embedding model and a decoder for the judge: [Ollama](https://ollama.com/)
or a [llama.cpp](https://github.com/ggml-org/llama.cpp) server locally, or an
[OpenRouter](https://openrouter.ai/) key (`HONTOLOGY_OPENROUTER_API_KEY`) for hosted
ones.

```bash
cp .env.example .env
make install      # create the venv, install dependencies
make up           # start Postgres with pgvector
make migrate      # apply the schema
make doctor       # check Postgres and the LLM provider are reachable
make api          # http://127.0.0.1:8100/docs
make ui           # http://localhost:8501   (in a second terminal)
```

Runs, ingest, labelling and evaluation can also be scripted from the command
line: `hontology --help`. The [usage guide](docs/usage.md#setting-up) covers the UI, serving
models with llama.cpp, and importing a first ontology: the app starts empty, by
design.

For development, `make test` runs the suite (it needs the database from
`make up`), and `make lint` runs the same lint, format and type checks as CI.

---

## Layout

```
src/hontology/
  config.py, db/, ontology/   settings, the schema, the ontology and its versions
  pipeline/
    ingest/                   feed/ (GDELT slices), codes/ (CAMEO, themes, places),
                              articles/ (filter, fetch, extract, de-duplicate)
    retrieve/                 embeddings, candidate selection, cutoff tuning,
                              strategies/ (routing, extraction, the stored record)
    judge/                    judging modes, providers, prompts
    runs/                     run configs, orchestration, sweeps, stage versions
  evaluation/                 metrics; labels/, calendar/, and scoring by pairs,
                              arms, stages and run outputs
  apps/                       api/ (FastAPI), cli/ (Typer), ui/ (Streamlit)
tests/                        the same layout
```

---

## Documentation

- **[Usage guide](docs/usage.md)**: setting up, defining and structuring an
  ontology, following the feed, running detection, calendars, labelling and
  evaluation, testing.
- **[Design notes](docs/design-notes.md)**: decisions that are load-bearing and
  non-obvious, and what each one prevents.
- **[Roadmap and future work](docs/roadmap.md)**: what is built, what is in
  progress, and what was deferred and why.
