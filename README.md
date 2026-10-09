# hontology

Define an ontology. Detect it in a live news feed. Find out honestly how well
that worked.

`hontology` is an end-to-end system for **ontology-driven event detection**: you
describe the things you care about — in your own words, through a UI — and the
pipeline finds evidence of them in the [GDELT](https://www.gdeltproject.org/)
global news feed, then scores itself against ground truth you accumulate as you
go.

> **Status: the pipeline is complete.** Define an ontology, follow the live
> GDELT feeds, filter and fetch article text, retrieve candidates, judge them,
> build ground truth, and score the result. Verified end to end against the
> live feed, with local and hosted models.
>
> **In progress: does an ontology's structure help a model detect events?** A
> pre-registered comparison of judging a 44-concept supply-chain ontology flat,
> top-down through its class hierarchy, and event by event, on the same
> articles, model and leaf wording. The baseline has run on a pilot event
> calendar; every arm is being judged and scored on a 320-article labelled
> sample. No results are published yet.

---

## The experiment

The question the current work answers: does a proper ontology, with event classes
in a hierarchy, detect events in the news better than a flat list of the same
concepts? It is answered with a comparison stated before any result was seen:

| Arm | Judging |
|---|---|
| **Baseline** | Flat: one call per article over the leaf concepts retrieval selected, at most 500 pairs per calendar window |
| **Hierarchical** | Top-down: parent classes asked as routing questions that lean to yes, then the children of every class answered yes; leaves asked exactly as in the baseline |
| **Extract** | Event by event: the article's distinct events listed once, then each event routed down the hierarchy and given at most one leaf |
| **Extract + embeddings** | As Extract, but below the top level each event is offered the leaves closest to it in embedding space |

Every arm sees the same articles, the same model and the same leaf wording. The
baseline judges exactly the pairs retrieval selected; the top-down arms use
retrieval only to decide which articles to read, then reach classes their own
way. They are scored two ways: on an event calendar of supply-chain
disruptions, their precursors and quiet control days, with every detection
verified by a person; and on a labelled sample of articles drawn in a frozen
order, where an arm counts as better only when the paired interval on its F1
difference lies above zero. Cost — calls, tokens and time — is reported beside
every number.

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
then said no. Reporting a single F1 conflates them. Candidates are stored as the
full pre-cutoff pool, so retrieval is scored with recall@k and MRR, and a
different cutoff can be tried after the fact, while the judge is scored on
precision and recall over what actually reached it.

**4. A result must say exactly what produced it.** Every stage is versioned by
what decides its output: the retrieval version by the embedding settings, the
classes' wording and a fingerprint of the ranking code; the judge version by its
settings and a fingerprint of the prompt as rendered. Each prompt id is pinned
to its wording, so a prompt edited in place is refused rather than quietly
changing what earlier verdicts mean, and a run is never continued under
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
pgvector.

**Flat or hierarchical, one pipeline.** Classes can be arranged with
`subclass_of`, several parents allowed. Retrieval ranks leaf classes only, and
every class of a flat ontology is a leaf, so adding structure leaves a flat run
exactly as it was. The judge has four modes: one call per pair, one per article,
top-down through the hierarchy, or event by event.

**The judge is pluggable.** Ollama and llama.cpp serve local models; OpenRouter
serves hosted ones, pinned to one host so two calls in a run are never served by
different setups. Prompt wording lives in a data file
(`pipeline/judge/prompts.toml`), exactly as the model sees it.

**Runs are content-addressed by stage.** A run config is sectioned
(`common` / `candidates` / `judge`) and each stage's artifact key chains to its
upstream:

```
key(candidates) = hash(candidates + relevant common + ontology_version)
key(judge)      = hash(judge + relevant common + ontology_version) + "_" + key(candidates)
```

Change only the prompt and `key(candidates)` is unchanged, so retrieval is reused
instead of recomputed. Behavior is hashed; infrastructure is not — the same run
against a model served from a different host is *the same run*, and treating a
host change as a new experiment would make cross-machine comparison impossible.

---

## Quick start

Requirements: Docker, Python 3.12+, [uv](https://docs.astral.sh/uv/), and a
model for the judge: [Ollama](https://ollama.com/) or a
[llama.cpp](https://github.com/ggml-org/llama.cpp) server locally, or an
[OpenRouter](https://openrouter.ai/) key (`HONTOLOGY_OPENROUTER_API_KEY`) for a
hosted model.

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
    retrieve/                 embeddings, candidate selection, cutoff tuning
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
