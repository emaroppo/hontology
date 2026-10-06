# hontology

Define an ontology. Detect it in a live news feed. Find out honestly how well
that worked.

`hontology` is an end-to-end system for **ontology-driven event detection**: you
describe the things you care about — in your own words, through a UI — and the
pipeline finds evidence of them in the [GDELT](https://www.gdeltproject.org/)
global news feed, then scores itself against ground truth you accumulate as you
go.

> **Status: all seven stages implemented.** Define an ontology, pull the live
> GDELT feed, scrape article text, retrieve candidates, judge them, build
> ground truth, and score the result. Verified end to end against the live
> feed with a local model.
>
> **In progress: does a class hierarchy beat a flat list?** A pre-registered
> comparison of a flat 44-concept supply-chain ontology against the same
> concepts arranged as a class hierarchy and judged top-down, on the same
> articles, retrieval, model and leaf wording. The flat baseline is running on
> a pilot event calendar; no results yet.

---

## The experiment

The question the current work answers: does a proper ontology, with event classes
in a hierarchy judged top-down, detect events in the news better than a flat list
of the same concepts? It is answered with a comparison stated before any result
was seen:

| Arm | Judging |
|---|---|
| **Baseline** | Flat: one batched call per article over the leaf concepts retrieval selected, at most 500 pairs per calendar window |
| **Hierarchical** | Top-down: top-level classes first, then the children of every class answered yes; uncapped, every call's cost recorded |

Both arms see the same articles, the same retrieval, the same model and the same
leaf wording; only the judging differs. They are scored two ways: on an event
calendar of supply-chain disruptions, their precursors and quiet control days,
with every detection verified by a person; and on a labelled sample of articles,
where an arm counts as better only when the paired interval on its F1 difference
lies above zero. Cost is reported beside every number.

---

## Why this exists

Most "LLM classifies documents" projects stop at a working pipeline and a
plausible-looking accuracy number. The hard part is not the classification — it
is knowing whether the number means anything. This project is built around three
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
labels from the denominator rather than silently scoring against them. Without
this, a normal afternoon of editing definitions quietly corrupts every number
computed afterwards, and nothing tells you.

**3. Retrieval failure and judgment failure need separate numbers.** A pipeline
that misses an event because the retriever never surfaced the concept needs a
completely different fix from one whose retriever ranked it first and whose judge
then said no. Reporting a single F1 conflates them. Candidates are stored as the
full pre-cutoff pool, so retrieval is scored with recall@k and MRR while the
judge is scored on precision and recall over what actually reached it.

---

## How it works

```
GDELT v2 feed  ──▶  ingest  ──▶  documents  ──▶  retrieval  ──▶  judge  ──▶  verdicts
 (15-min slices)   watermark,     scrape cache   code | semantic    LLM        │
                   backfill,      (disk-backed)  (both emit the     (pluggable)│
                   idempotent                     same contract)               │
                                                                               ▼
                            ground-truth bank  ─────────────────────▶     evaluation
                            observations + pair labels                per-stage metrics,
                                                                    A/B with significance
```

**Two retrieval strategies, one contract.** The `code` source joins the feed's
own CAMEO event codes to your concepts through a curated mapping. The `semantic`
source embeds concept definitions and article bodies and retrieves by cosine
similarity in pgvector. Both emit identical `(document, concept, score, rank)`
rows, so downstream stages cannot tell them apart and you can A/B the strategies
directly.

**Flat or hierarchical, one pipeline.** Classes can be arranged with
`subclass_of`, several parents allowed. Retrieval ranks leaf classes only, and
every class of a flat ontology is a leaf, so adding structure leaves a flat run
exactly as it was. A hierarchical judge then descends the tree: it asks about the
top-level classes in one call, and only below a yes asks about the children.

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

Requirements: Docker, Python 3.12+, [uv](https://docs.astral.sh/uv/), and
either [Ollama](https://ollama.com/) or a
[llama.cpp](https://github.com/ggml-org/llama.cpp) server if you want to run the
judge locally.

```bash
cp .env.example .env
make install      # create the venv, install dependencies
make up           # start Postgres with pgvector
make migrate      # apply the schema
make doctor       # check Postgres and the LLM provider are reachable
make api          # http://127.0.0.1:8100/docs
make ui           # http://localhost:8501   (in a second terminal)
```

The [usage guide](docs/usage.md#setting-up) covers the UI, serving models with
llama.cpp, and importing a first ontology: the app starts empty, by design.

---

## Documentation

- **[Usage guide](docs/usage.md)**: setting up, defining and structuring an
  ontology, following the feed, running detection, calendars, labelling and
  evaluation, testing.
- **[Design notes](docs/design-notes.md)**: decisions that are load-bearing and
  non-obvious, and what each one prevents.
- **[Roadmap and future work](docs/roadmap.md)**: what is built, what is in
  progress, and what was deferred and why.
