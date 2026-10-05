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

The UI is six pages, each a pure HTTP client of the API:

| Page | What it is for |
|---|---|
| **Ontology** | Author concepts, import/export, resolve versions, health checks |
| **Code Links** | Curate concept↔CAMEO associations from similarity proposals |
| **Labelling** | The queue, adjudication of machine proposals, bank import/export |
| **Evaluation** | Funnel, detections, per-stage metrics, breakdowns, errors, leaderboard |
| **Runs** | Start and watch runs; preview stage keys before paying for them |

`make doctor` is the first thing to run if something is not working: it reports
the database and the model provider separately, and degrades gracefully when
only one is up — the ontology layer works fine with no model at all.

**Serving models with llama.cpp.** Set `"provider": "llamacpp"` in a run's
`judge` or `candidates` section and point `HONTOLOGY_LLAMACPP_HOST` at a
`llama-server`; it is spoken to over its OpenAI-compatible API. One server holds
one model, so embeddings usually need a second one started with `--embeddings`,
at `HONTOLOGY_LLAMACPP_EMBED_HOST`. The model name in a config may be the server's
id, an alias, or the bare file name without `.gguf` — the path is where the model
lives, not what it is, so it stays out of the run's identity. Three things a plain
OpenAI client would get wrong are refused rather than tolerated: a config naming
a model the server is not holding (the server would answer with its own model
anyway), thinking left to the chat template's default (reasoning models think
unless told not to, which would make `think: false` untrue), and a per-slot
context smaller than `context_window`, which llama.cpp fixes at startup rather
than per request.

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

### Filtering before you fetch

The feed tags every event with a CAMEO code *before* anything is downloaded. That
is the only signal available pre-scrape — semantic similarity needs the article
body, so it cannot tell you whether the body was worth fetching. With ~260 new
URLs every fifteen minutes (~25k/day), spending the scrape budget on documents an
ontology could plausibly match is worth more than any downstream tuning.

```bash
hontology ingest filter-preview <ontology-id>   # what it would keep
hontology ingest scrape --ontology-id <id>      # apply it
```

**Opt-in, and off by default.** It only means anything for an ontology whose
concepts are mapped onto the code system; for one that isn't, applying it would
silently fetch nothing. When no document matches, the scraper says so rather than
returning a quiet zero.

**Everything is still ingested** — only fetching is gated. Feed rows are cheap and
the corpus is shared between ontologies, so discarding a document because *this*
ontology can't use it would corrupt the corpus for the next one.

The tier fallback walks event → base → root and stops at the **first tier that
carries a code**, matched or not. Falling through after a specific code failed
would let an unmatched `1451` ("riot") be retried as `14` ("PROTEST") and match
everything protest-shaped — far more than the event supports.

### The knowledge graph: articles with no political event

The event export only lists articles from which GDELT could code a political
event. A factory fire, a ransomware attack or a drug shortage has no state actor,
so its articles never appear there. Measured on one 15-minute slice, the export
listed 225 article URLs and GDELT's Global Knowledge Graph (GKG) listed 1,337:
**five in six articles GDELT reads are invisible to the event export.**

The GKG is ingested as a second feed beside the export, with its own watermark,
and tags each article with themes (`CYBER_ATTACK`, `SHORTAGE`, `WB_167_PORTS`…)
and every country it mentions. Themes are a second code system beside CAMEO, so
the same curated links drive the filter for both feeds, and a document passes
if either matches it.

```bash
hontology ingest themes                                # load the theme vocabulary
hontology ontology links-import <id> links.csv         # concept,system,code rows
```

A GKG slice is about seventy times the size of an export slice, so a historical
backfill can keep only articles mentioning given countries. A slice records the
scope it was kept for, and is reprocessed rather than skipped when a later request
asks for a place outside it: "ok for Hong Kong" is not "ok for the Netherlands".

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

Or from the API, where a run is an async job — `POST /runs` returns an id
immediately and the work continues in the background:

```bash
curl -X POST localhost:8100/runs -H 'Content-Type: application/json' \
  -d '{"ontology_id": 1, "document_limit": 50, "config": {}}'
curl localhost:8100/runs/1     # status, stage, progress_done/progress_total
```

The **Runs** page in the UI wraps both, and previews the stage keys as you edit
the config so you can see whether a change will reuse retrieval before starting.

**Resume is per pair.** Each verdict commits as it completes, so a run killed two
thirds through restarts from where it stopped. With a local model that is
wall-clock time; with a hosted one it is money.

**Failures are recorded, not raised.** An unparseable response becomes a verdict
row carrying the error, and the loop continues. A separate liveness check asserts
every pair produced a usable verdict — worth its own gate because a
malformed-output bug does not move precision or recall, it silently removes pairs
from the denominator.

**Repeated sampling beats self-reported confidence.** Set `judge.samples` above 1
and the vote's fraction replaces the model's own number. A model asked five times
and answering yes three times is uncertain in a way its stated 0.95 does not
capture. `judge.aggregation` chooses how those votes resolve, as an explicit
precision/recall trade:

| `aggregation` | Matches when | Use when |
|---|---|---|
| `majority` | the plurality says so | balanced; the default |
| `unanimous` | *every* sample says so | a false positive is expensive |
| `any` | *any* sample says so | catching what the model only sometimes notices |

**One call per document, or one per pair.** `judge.prompt_id` carries the
construction mode. `strict_v1` asks about one concept at a time; `strict_batch_v1`
judges every candidate concept for an article in a single call, sending the body
once instead of N times. Same guidance, so the two are directly comparable — they
share a candidates key and fork only the judge key, because batching changes how
the model is asked, not what is retrieved.

Batching trades isolation for cost: a malformed response costs every pair for
that document rather than one, so failures are recorded against each of them and
the omission count is reported. A model that quietly drops concepts from its
array is visible rather than silently shrinking the denominator. Sampling does
not apply to batched runs — repeating the call re-rolls every verdict together,
so the votes are not independent.

**Document embeddings are cached.** Bodies are embedded once and stored
content-addressed, so a sweep over selection parameters — which change nothing
about the text — costs no embedding calls at all. `--refresh-embeddings` forces a
clean recompute for when the cache itself is the suspect; it is a runtime flag
rather than a config field, because recomputing an identical vector gives an
identical result and must not fork the artifact tree.

---

## Evaluating

```bash
hontology eval run 1                    # per-stage metrics for a run
hontology eval run 1 --record           # ...and add it to the leaderboard
hontology eval compare 1 2              # paired comparison
hontology eval consistency 1 --against 2
hontology eval baseline 1 --out baseline.json
hontology eval gate 1                   # liveness, then metric floors
hontology eval leaderboard

hontology eval breakdown 1 --dimension concept   # sliced metrics
hontology eval errors 1                          # every mistake, with reasoning
hontology eval sweep 1 examples/sweep.json       # plan; --execute to run
hontology eval filter-report 1                   # per-code cost and benefit
hontology ontology lint 1                        # ontology health

hontology eval funnel 1                          # where the volume went
hontology eval detections 1 --out found.csv      # what the pipeline found
hontology eval detections 1 --events             # one row per (concept, place, date)
```

**Getting the findings out.** Everything else exports the machinery's inputs or
its scores; `eval detections` exports its *output*. Two shapes: one row per
matched `(document, concept)` pair with the evidence quote — the audit trail —
or aggregated to one row per `(concept, locus, date)`, since one riot reported by
four outlets is one fact. Every row carries a **verification** status, because a
detection is a model's claim rather than a fact, and exporting confirmed and
unreviewed ones indistinguishably would launder model output into apparent
ground truth.

**Event calendars give a first number with no labels.** A calendar is a reviewed
CSV of events known from outside the pipeline to have happened (a port strike in
the Netherlands on 2025-10-08), their precursors (the strike notice the day
before), and quiet control days. Each entry is a country, or several written
`HUN|SVK` when the place is genuinely ambiguous, plus a window of days around its
date.

```bash
hontology ingest calendar calendar.csv                 # backfill both feeds for its windows
hontology ingest calendar-preview calendar.csv --ontology-id <id>
hontology ingest scrape --calendar calendar.csv --ontology-id <id>
hontology run start <id> run.json --calendar calendar.csv
hontology eval calendar <run-id> calendar.csv
```

Scoring reports event recall over the positives, precursor recall, a false-alarm
rate over the controls, and lead time where a precursor was matched before its
disruption's day. Every entry is scored through the stages its documents pass,
`in_feed → passed_filter → fetched → retrieved → matched`, so a miss says where it
happened: an event whose articles never reached the feed needs a different fix
from one the judge rejected.

**Funnel before metrics.** `eval funnel` shows attrition stage by stage and needs
no ground truth at all, which makes it the first thing to read when a run
produces less than expected — precision cannot tell a precise pipeline from a
broken scraper. Each step reports its share of the *previous* step, because that
is what localises a problem, and carries a note saying what a healthy drop looks
like there. Most attrition is the pipeline working.

**Slice before you conclude.** A pooled F1 cannot distinguish "uniformly
mediocre" from "excellent on nine concepts and hopeless on the tenth", and those
need different work. `eval breakdown` slices by concept, category or locus with a
Wilson interval per row.

**Read the errors, not just the count.** `eval errors` joins every
misclassification to the model's own evidence quote and reasoning trace — both
recorded on every verdict — so failure patterns can be read in bulk.

**Lint the ontology before labelling against it.** `ontology lint` catches
strength drift (a concept named "*Successful* negotiation" whose definition only
requires "a concrete commitment" will match a mere promise, and the judge is
*right* to — the bug is in the ontology) and near-duplicate concepts that
retrieval cannot separate.

**Sweeps are resumable and share retrieval.** A 3-prompt × 2-cutoff sweep is six
runs but only **two** distinct retrieval keys, so embedding happens twice, not
six times; cells already run are skipped entirely.

The bank itself travels as CSV:

```bash
hontology labels export <ontology-id> --out bank.csv
hontology labels import <ontology-id> bank.csv        # skips existing by default
hontology labels export <ontology-id> --observations
```

Rows are keyed by **document URL and concept name**, never by database ids, so a
bank exported here imports cleanly elsewhere. Three rules keep an import from
damaging one: existing labels are not overwritten unless you ask, unknown
documents become stubs the scraper fills in later, and the `ontology_version`
stamp is preserved rather than re-stamped — re-stamping would claim a human read
today's wording when they did not, which quietly defeats staleness detection.

Three things the evaluation layer refuses to do:

**Report a rate without its denominator.** Every metric is printed with the
label count it was computed over, and precision and recall carry Wilson
intervals while F1 carries a bootstrap one. A ten-point F1 gap on thirty pairs
is noise, and an interval makes that visible instead of arguable.

**Compare unpaired.** Two runs judged the same pairs, so only the pairs they
answered *differently* say anything about which is better. McNemar's test over
those discordant pairs, and the verdict says plainly when a difference is inside
the noise.

**Pass a gate that checked nothing.** The regression gate runs liveness first —
did every pair produce a usable verdict — because a malformed-output bug does
not move precision or recall, it silently shrinks the denominator. Then metric
floors. And if no trusted label covers the run, the gate reports
**inconclusive** and exits non-zero rather than green, because a gate that
passes on an empty bank is worse than no gate.

Retrieval and judgment are scored separately, always. A concept the retriever
never surfaced needs a different fix from one it ranked first and the judge then
rejected, and a single F1 hides which you have.

---

## Testing

```bash
make test        # everything
make test-fast   # only what needs no services
```

A missing service skips the tests that need it rather than failing them: without
Postgres the database tests are skipped with a note to run `make up`, and without
a reachable model provider the one live-model test is skipped. Set
`HONTOLOGY_TEST_REQUIRE_SERVICES=1` in CI so an unreachable service fails the
run instead of quietly shrinking it.

Tests **never touch the development database.** They run against a separate
`<database>_test`, created and migrated on first use, and a guard refuses to
start if the target database name does not end in `_test`. Each test then runs
inside a transaction that is rolled back afterwards — bound with
`join_transaction_mode="create_savepoint"`, so the `commit()` calls inside
application code (the judge loop commits per pair on purpose) become savepoint
releases rather than real writes.

The combination means a test can be arbitrarily destructive and still leave
nothing behind: after the full suite the test database contains zero ontologies,
concepts or labels.

This exists because it was needed. Two tests once contained an unqualified
`DELETE FROM pair_labels`, which emptied the table for the whole database rather
than the fixture and destroyed a session's worth of labelling. Six more asserted
against a bare `select(PairLabel)`, reading whichever row came back first rather
than their own. Both classes of bug are now structurally impossible rather than
individually fixed.

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
- **A model's self-reported confidence is not trusted until it earns it.** The
  queue ranks partly by model uncertainty, so a run that reports 0.5 on
  everything would look maximally uncertain on every pair and flood it. Each
  run's confidence distribution is therefore checked before being used as
  signal. On this project's own local 9B judge it fails: one run emitted three
  distinct values (0.90/0.95/1.00, σ=0.033), another emitted **0.85 for every
  single verdict** (σ=0.000) — and mean confidence when it said *matched* (0.900)
  was lower than when it said *not matched* (0.950). That number carries no
  information about correctness, which is exactly why repeated sampling's vote
  fraction replaces it and why the queue ignores it.
- **ALL-CAPS text is lowercased before embedding.** This one was found by
  debugging, not design. CAMEO's root labels are all upper case, and feeding
  seven of them to `nomic-embed-text` returns only **three** distinct vectors —
  several bit-for-bit identical, mean pairwise cosine 0.97. Every concept then
  matched the same handful of codes with entirely plausible-looking scores.
  Lowercasing the same labels gives seven distinct vectors at mean cosine 0.65,
  and "Riot" starts matching `ASSAULT` / `PROTEST` / `FIGHT` instead of
  `MAKE PUBLIC STATEMENT`. The failure mode is bad ranking with healthy numbers,
  which is why it is now a regression test.

## Future work

Deferred deliberately, with the reason.

- **Rank by information gain, not just relevance.** The ingest filter answers
  "could this article match?" but not "is it worth reading?". The intended
  ranking is `Σ |weight| × (1 − activation)` over an article's matched concepts,
  where activation is how much that concept is *already known* to be firing in
  that place — so a first signal scores high and more of something established
  scores low. Needs meaningful per-concept payoff weights and a per-locus
  activation state, neither of which is defined yet. **Sequenced after the
  ontology is refined.**
- **Locus-scoped tracking, a per-place UI, and a one-command bootstrap.** All
  three assume a country-scoped domain, which cuts against a domain-agnostic
  engine. **Pinned until the domain direction is settled.**
- **Concept groups have no API or UI.** The service layer supports forking a
  group and retuning its edge weights, but it is reachable only through import.
- **No machine pre-labelling pass**, though the schema and the adjudication flow
  were built for one.
- **Descriptive per-stage drill-down.** The funnel and detection export both work
  with no labels; score distributions and per-stage inspection do not exist yet.
- **Batched judging is unmeasured.** It has run against the local 9B model only
  as a smoke test, which confirmed the response parses into one verdict per
  concept. Its accuracy relative to per-pair judging is untested, and all three
  aggregation rules are covered only by tests with a mocked provider.

## Roadmap

- [x] Schema: ontology, taxonomy, vectors, corpus, runs, labels, snapshots
- [x] Pluggable LLM provider layer (Ollama, llama.cpp; Anthropic behind the same protocol)
- [x] Ontology service, import/export, snapshots, API and editor UI
- [x] CAMEO ingest, embeddings, similarity review
- [x] GDELT ingest: watermarking, catch-up, backfill, optional continuous watcher
- [x] Article scraping: extractor chain, quality gate, robots and rate limiting
- [x] Run configuration, both retrieval sources, judge loop with resume
- [x] Ground-truth bank, labelling queue, staleness handling
- [x] Metrics, A/B comparison, consistency checks, regression gate, leaderboard
