# hontology

Define an ontology. Detect it in a live news feed. Find out honestly how well
that worked.

`hontology` is an end-to-end system for **ontology-driven event detection**: you
describe the things you care about — in your own words, through a UI — and the
pipeline finds evidence of them in the [GDELT](https://www.gdeltproject.org/)
global news feed, then scores itself against ground truth you accumulate as you
go.

> **Status: in development.** The pipeline runs end to end today: define an
> ontology, pull the live GDELT feed, scrape article text, retrieve candidates
> and judge them. The evaluation layer is next — see [Roadmap](#roadmap).

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

## Keeping up with the feed

GDELT publishes an export slice every 15 minutes, stamped on the quarter hour in
UTC. Ingest is one-shot by default:

```bash
make ingest          # catch up to the newest published slice, then exit
make ingest-status   # watermark, lag, slice outcomes, document counts
hontology ingest backfill 20260801000000 20260801234500
```

**Continuous ingest is opt-in.** Nothing starts it for you — the API never
launches a scheduler, and a plain `docker compose up` brings up only the
database. A clone of this repo does not begin crawling on its own.

```bash
make watch           # follow the feed in the foreground, Ctrl-C to stop
make watch-docker    # or as a container: docker compose --profile watch up -d
```

What the watcher does that a `while true; sleep 900` loop does not:

- **Polls in phase with the feed.** A fixed interval drifts out of alignment with
  a quarter-hour publish schedule and ends up asking at the least useful moment.
  Each wake-up targets the next boundary plus a grace period, because publication
  lags the stamp — asking for the current wall-clock quarter hour reliably 404s.
- **Catches up after downtime.** The watermark records the last completed slice.
  On restart the ingester walks the gap in order, bounded per pass so a long
  outage becomes several ordered passes rather than one unbounded crawl.
- **Advances the watermark contiguously.** It moves only to the immediate
  successor of the current mark, so a slice that failed transport is retried
  rather than stepped over by a later success. A slice the feed never published
  is terminal after a few hours — otherwise one permanent gap stalls ingest
  forever.
- **Refuses to double-ingest.** A Postgres advisory lock means a second watcher,
  or a manual catch-up racing the scheduled one, is a no-op rather than a
  duplicate download.
- **Records every attempt.** Per-slice status (`ok` / `empty` / `missing` /
  `failed` / `pending`) with row counts, so a gap in the corpus is *visible*
  rather than indistinguishable from a quiet news period.
- **Shuts down gracefully.** SIGTERM finishes and commits the slice in flight
  instead of losing it, and the sleep is interruptible so stopping is immediate.

### Fetching article text

The feed gives URLs, not article bodies. Fetching them is a separate, bounded step:

```bash
hontology ingest scrape --limit 200
```

Extraction runs a **fallback chain behind a quality gate** — trafilatura first,
then readability, with a third-party reader service available but off by default
because it sends target URLs to an external host.

The gate is the part that matters. A failed fetch is easy to spot; the expensive
failure is a *successful* fetch of something that is not an article — a 404 page
served with a 200, a paywall stub, a cookie wall, a navigation dump. That text
reads fine, embeds fine, and quietly degrades everything downstream. So junk is
treated as **failure**: it does not end the chain, it falls through to the next
extractor, which is what makes having a chain worth anything.

Fetching is polite by construction: robots.txt honored and cached per host
(a 5xx on robots means *disallow*, per RFC 9309, rather than assuming permission),
requests to one host serialized with a minimum gap while concurrency happens
across hosts, retries only on transient statuses, and a hard per-run budget.

`lag_slices` is the number to watch. It is `null` before the first ingest rather
than `0`, because a fresh install that has never run is idle, not current.

---

## Running detection

A run is one JSON file. Check what it will cost before paying for it:

```bash
hontology run keys examples/baseline.json --ontology-version v1
#   candidates  5b2874d141ad
#   judge       219a3416600d_5b2874d141ad

hontology run keys examples/lenient-prompt.json --ontology-version v1
#   candidates  5b2874d141ad      <- identical: retrieval will be reused
#   judge       5dc88ccdca0d_5b2874d141ad
```

Those two configs differ only in `judge.prompt_id`, so they share a candidates
key and the second run **copies** the first's retrieval instead of re-embedding
the corpus. Prompt iteration is the loop you run most, and this is what makes it
cost only the judging.

```bash
hontology run start <ontology-id> examples/baseline.json --documents 100
hontology run resume <run-id>      # continues, skipping pairs already judged
hontology run list
```

**Resume is per pair.** Each verdict commits as it completes, so a run killed two
thirds through restarts from where it stopped. With a local model that is
wall-clock time; with a hosted one it is money.

**Failures are recorded, not raised.** An unparseable response becomes a verdict
row carrying the error, and the loop continues. A separate liveness check asserts
every pair produced a usable verdict — worth its own gate because a
malformed-output bug does not move precision or recall, it silently removes pairs
from the denominator.

**Repeated sampling beats self-reported confidence.** Set `judge.samples` above 1
and the majority vote's fraction replaces the model's own number. A model asked
five times and answering yes three times is uncertain in a way its stated 0.95
does not capture.

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
- [x] GDELT ingest: watermarking, catch-up, backfill, optional continuous watcher
- [x] Article scraping: extractor chain, quality gate, robots and rate limiting
- [x] Run configuration, both retrieval sources, judge loop with resume
- [ ] Ground-truth bank, labeling queue, staleness handling
- [ ] Metrics, A/B comparison, consistency checks, regression gate, leaderboard
