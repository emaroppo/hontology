# hontology

Define an ontology. Detect it in a live news feed. Find out honestly how well
that worked.

`hontology` is an end-to-end system for **ontology-driven event detection**: you
describe the things you care about — in your own words, through a UI — and the
pipeline finds evidence of them in the [GDELT](https://www.gdeltproject.org/)
global news feed, then scores itself against ground truth you accumulate as you
go.

> **Status: in development.** You can define, version, import and export an
> ontology today. Ingest, detection and evaluation are next — see
> [Roadmap](#roadmap).

---

## Why this exists

Most "LLM classifies documents" projects stop at a working pipeline and a
plausible-looking accuracy number. The hard part is not the classification — it
is knowing whether the number means anything. This project is built around three
claims that are easy to state and annoying to actually implement:

**1. The ontology belongs to the user, not the code.** Concepts, their
definitions, their inclusion and exclusion criteria, and how they group together
are data, editable in the UI. Changing what you're looking for should not require
changing the pipeline.

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
[Ollama](https://ollama.com/) if you want to run the judge locally.

```bash
cp .env.example .env
make install      # create the venv, install dependencies
make up           # start Postgres with pgvector
make migrate      # apply the schema
make api          # http://127.0.0.1:8100/docs
make ui           # http://localhost:8501   (in a second terminal)
```

**The app starts empty, by design** — there is no bundled ontology, because the
whole premise is that the ontology is yours. Create one in the UI, or import a
JSON file:

```bash
curl -X POST localhost:8100/ontologies/import -H 'Content-Type: application/json' -d '{
  "slug": "supply-chain",
  "name": "Supply chain disruption",
  "concepts": [
    {
      "name": "Port closure",
      "definition": "A commercial seaport suspends or halts vessel operations.",
      "inclusion_criteria": "Closure has already taken effect; partial or full.",
      "exclusion_criteria": "Threatened, planned or announced-for-the-future closures; routine maintenance."
    }
  ]
}'
```

Inclusion and exclusion criteria are not documentation — they go into the judge
prompt verbatim. Writing a sharp exclusion is usually the highest-leverage edit
available for precision.

---

## Design notes

A few decisions that are load-bearing and non-obvious:

- **Pair labels have no foreign key to observations.** An earlier version made a
  label a child of an asserted event, which meant the most valuable label in the
  system — *the pipeline proposed this and it was wrong* — was unrepresentable,
  because a rejected match has no event to hang from. Precision could not be
  measured at all. The two tiers are independent for that reason.
- **Machine-proposed labels do not count as ground truth until a human confirms
  them.** Scoring an LLM against labels another LLM produced unsupervised
  measures inter-model agreement, not correctness. Confirming a proposal promotes
  it to `adjudicated`, which does count; the promotion step is the whole point.
- **Embeddings are content-addressed** by a hash of the exact text embedded, not
  by which fields were used. Keying on the field composition alone leaves a stale
  vector in place when a definition is edited, and similarity silently keeps
  scoring the old wording.
- **Concept weight is excluded from the snapshot hash.** It never reaches the
  embedding or the prompt, so changing it must not invalidate labels.
- **Auto-proposed code links carry the run that proposed them; hand-made links
  are NULL.** Recomputation rebuilds only the former, so manual curation survives.
- **Scrape failures are cached as rows.** A dead or paywalled URL with no row
  gets re-fetched on every run forever; the row makes a permanent failure cost
  one request, with an explicit opt-in to retry.
- **ALL-CAPS text is lowercased before embedding.** This one was found by
  debugging, not design. CAMEO's root labels are all upper case, and feeding
  seven of them to `nomic-embed-text` returns only **three** distinct vectors —
  several bit-for-bit identical, mean pairwise cosine 0.97. Every concept then
  matched the same handful of codes with entirely plausible-looking scores.
  Lowercasing the same labels gives seven distinct vectors at mean cosine 0.65,
  and "Riot" starts matching `ASSAULT` / `PROTEST` / `FIGHT` instead of
  `MAKE PUBLIC STATEMENT`. The failure mode is bad ranking with healthy numbers,
  which is why it is now a regression test.

## Roadmap

- [x] Schema: ontology, taxonomy, vectors, corpus, runs, labels, snapshots
- [x] Pluggable LLM provider layer (Ollama; Anthropic behind the same protocol)
- [x] Ontology service, import/export, snapshots, API and editor UI
- [x] CAMEO ingest, embeddings, similarity review
- [ ] GDELT ingest with watermarking, backfill and a polite bounded scraper
- [ ] Run configuration, both retrieval sources, judge loop with resume
- [ ] Ground-truth bank, labeling queue, staleness handling
- [ ] Metrics, A/B comparison, consistency checks, regression gate, leaderboard
