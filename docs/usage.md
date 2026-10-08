# Usage guide

How to run [hontology](../README.md), stage by stage. Why it is built this way
is in the [design notes](design-notes.md); what comes next is in the
[roadmap](roadmap.md).

- [Setting up](#setting-up)
- [Structuring the ontology](#structuring-the-ontology)
- [Keeping up with the feed](#keeping-up-with-the-feed)
- [Running detection](#running-detection)
- [Evaluating](#evaluating)
- [Testing](#testing)

---

## Setting up

Install and start the services as in the [quick start](../README.md#quick-start), then:

The UI is seven pages, each a pure HTTP client of the API, listed in the sidebar.
The settings several pages share, the ontology, the run, the truth scores are
computed against and the labelled sample, are in a panel that opens from the top
right corner of every page; its button shows the current choice, and a choice
made on one page holds on the others. Everything else a page needs is in its
body.

| Page | What it is for |
|---|---|
| **Home** | Every judged run scored end to end on the labelled sample (human labels or a machine annotation set), as the arms report scores it. Tabs: **Leaderboard** (stage versions, coverage of the sample, F1 with intervals, cost; calendar recall on request), **Comparison** (arms against a baseline, paired, the `eval arms` tables), **Single run** (end to end, a couple of numbers per stage, disagreements, funnel, detections) |
| **Ontology** | Author a flat event set in place; for a hierarchy, view the tree and edit leaf wording (structure comes from Protégé via OWL import); import/export, version history, health checks |
| **Filtering** | Link classes to feed codes (CAMEO, GKG themes), ticking similarity candidates; preview what the pre-download filter would keep. **Evaluation**: a version of the links (a snapshot, or the live ones) scored link by link against the labels (TP, FP, FN, TN, unique TP), the calendar, and corpus cost (articles admitted, and admitted by no other link) |
| **Retrieval** | Laid out as Judgement is. **Try a cutoff**: the ranking for your own pasted text (embedded and dropped, never stored), a corpus article, or the articles that rank one class highest, as the run chosen in Settings ranks them, under a cutoff in a collapsible section that starts from the run's own. **Evaluation**: a retrieval version (ranking settings, leaf wording, ranking code) under any cutoff, set by hand or loaded from a run: recall computed live on every labelled article, at each depth and by class, and cost, the pairs sent to the judge, on the articles of the run whose ranking it is |
| **Judgement** | **Evaluation**: a judge version across the runs that used it, on the pairs it was responsible for or only on those retrieval selected, with calibration. Pick a class and a few articles and see what the model says, asked exactly as a chosen run asks; edit the prompt's system text and the class wording and see each answer beside the original, the label and the run's recorded verdict. Nothing is stored; wording that works can be saved to the class. Per-pair prompts only. A trial can ask about pasted text instead of corpus articles, the same text Retrieval's Your text tab ranked; it is sent to the run's judge and never stored |
| **Runs** | Put a run together in a form or as JSON, switching freely between the two (the same config either way), starting from the defaults or any run's config; preview its stage keys before paying for it; start and watch runs |
| **Labelling** | Two tabs: **Sample**, blind whole-document labelling of a frozen sample, in order; **Queue**, single pairs ranked by what a label would teach, adjudication of machine proposals, bank import/export |

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

## Structuring the ontology

A flat list of concepts becomes a class hierarchy by adding relations, written
by class name in the export format (version 2):

```json
"relations": [
  ["Import ban or quota", "subclass_of", "Trade restriction"],
  ["Import ban or quota", "subclass_of", "Sanctions"],
  ["Logistics strike notice", "precursor_of", "Port strike"]
]
```

`subclass_of` makes the ontology a directed acyclic graph: cycles are refused,
and a relation set is replaced whole, so a bad file leaves the old one in place.
A class may have several parents, read as **either**, not both: an import ban can
be adopted as trade policy or as a sanction, without every ban being a sanction.
The top level is every class without a parent; there is no artificial root.
`precursor_of` is stored for later use and drives no logic yet.

Structure is part of an ontology's version only once there is some, so a flat
ontology keeps its version hash. Leaf wording is what labels answer, so adding
parent classes leaves every leaf label valid, and an import can be told to refuse
any rewording of existing classes.

The hierarchy can be authored or reviewed in [Protégé](https://protege.stanford.edu/)
or any OWL tool:

```bash
hontology ontology export-owl <id> ontology.ttl
hontology ontology import-owl ontology.ttl          # refuses to reword existing classes
```

Each class is an `owl:Class` with its definition and criteria as annotations. A
single parent is `rdfs:subClassOf`; two parents become `rdfs:subClassOf` an
`owl:unionOf`, OWL's spelling of either-or.

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

The **Filtering** page does the same in the UI. It loads either codebook, and a
similarity run ranks codes against each class; nothing is linked until ticked.
Proposing links automatically was tried and let far too much through: on the
supply chain ontology the default selection added about 14 CAMEO codes to every
class, most of them ones no person had chosen, and the filter admits whatever any
link admits. Theme candidates are only event-like themes, those used at least
10,000 times and outside the `TAX_` entity lists (occupations, languages,
species), embedded as readable words. The page also previews what the filter
would keep and reports, per linked code, what it admitted and what that was worth
once labelled (`hontology ingest filter-preview` and `hontology eval filter-report`
from the command line).

A GKG slice is about seventy times the size of an export slice, so a historical
backfill can keep only articles mentioning given countries. A slice records the
scope it was kept for, and is reprocessed rather than skipped when a later request
asks for a place outside it: "ok for Hong Kong" is not "ok for the Netherlands".

### Near-duplicates

Wire stories are republished under many URLs: in a random slice, 23% of articles
repeat another's headline, and an event window is worse. After fetching, bodies
are grouped by MinHash over five-word shingles, and each copy points at one
**representative**. Only representatives are retrieved and judged; a copy reads
its representative's verdicts wherever results are read. A group keeps its
representative when it grows, so verdicts never move and no copy points at
another copy.

```bash
hontology ingest dedup --calendar calendar.csv
```

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

**Several scrapers can run at once without duplicating work.** Each claims its
batch with `SELECT … FOR UPDATE SKIP LOCKED`, so no article is fetched twice, and
holds a host through a Postgres advisory lock while it works on it, so the
per-host spacing stays what one scraper would keep. That is what lets a scraper
run ahead of a calendar run while the judge is busy.

**A connection failure is not taken at its word.** A name that does not resolve
looks the same whether the site is gone or the local network dropped out for a
moment, so a DNS or connection error leaves the article pending for a later
batch and counts the failure; only the third is recorded as final. A 404, a 403
or a timeout is final at once.

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

**Or top-down through the hierarchy.** `hier_batch_v1` judges an article by
descending the class hierarchy: every top-level class in one batched call, then,
for each class answered yes, its children as one call per sibling set. Nothing
below a no, or below a failed call, is asked, and a class with two parents that
both said yes is asked once. The wording is `strict_batch_v1`'s, unchanged; only
the list of classes in each call differs, so a flat and a hierarchical run
answer the same question per leaf. Retrieval serves only as a gate: an article
is judged if retrieval selected any leaf for it, and the descent may reach
leaves retrieval did not pick. Descent state is rebuilt from stored verdicts, so
an interrupted run resumes without asking anything twice, and each call's
tokens are split exactly across the classes it answered.

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

A backfill takes hours, so `run calendar` does not wait for it. An entry is
ready once every slice of its window is ingested in both feeds; it is then
scraped, deduplicated, retrieved and judged into one run while later windows are
still arriving. Finished entries are recorded on the run, so `--run-id` resumes,
and `--prepare-only` stops before the judge to show each window's volume first.
`--budget` caps the pairs judged per window, highest retrieval score first; the
judge never sees which concept the calendar expects, so the cap is blind to the
answer.

```bash
hontology run calendar <id> run.json calendar.csv --prepare-only
hontology run calendar <id> run.json calendar.csv --run-id <run> --budget 200
```

A run scrapes a window and then judges it, so the scraper idles while the judge
works. A second process can scrape the later windows in the meantime; the two
never fetch the same article, and the run waits for any batch the other still
holds in its window before retrieving it:

```bash
hontology ingest scrape --calendar calendar.csv --ontology-id <id>   # beside run calendar
```

The feed itself has gaps: GDELT published nothing from 15 June to 1 July 2025.
A slice the source never published is terminal, so a window inside such a gap
would count as ingested with no articles in it, scoring its event as a miss the
detector never had a chance at. Such a window is set aside instead, recorded on
the run and never scored.

`--candidates-from <run>` makes a run reuse another's retrieval, window by
window, so two arms differ only in how they judge; it refuses to start if any
leaf's wording changed since that run's version.

Scoring reports event recall over the positives, precursor recall, a false-alarm
rate over the controls, and lead time where a precursor was matched before its
disruption's day. Every entry is scored through the stages its documents pass,
`in_feed → passed_filter → fetched → retrieved → judged → matched`, so a miss says where it
happened: an event whose articles never reached the feed needs a different fix
from one the judge rejected.

A match only says that *some* article in the window reported the concept, not
that it reported *this* event: in the pilot, two of eight "detections" were
other export measures and another country's port. So detections are verified
by a person. `eval calendar-review-export` lists the matches awaiting review,
one row per story rather than per copy; `confirmed` is marked yes or no and
read back with `eval calendar-review-import`. Reviews are keyed by entry and
URL, so one review serves every run that matches the same article. Scoring then
reports raw and verified recall side by side; a control whose match turns out
to be a real instance is withdrawn as a calendar error rather than counted.

Only entries a run has actually processed are scored. An entry the run never
reached has no verdicts, which would otherwise read as a miss, or as a quiet
control. Judging cost (pairs, tokens, seconds) is reported per window and per
run, from the verdicts themselves, so arms are compared on the same measure.

**Article-level scores need a labelled sample drawn independently of any run.**
`labels sample` takes every fetched representative in a calendar's windows,
including articles retrieval found nothing in, and fixes one random order in
which every prefix is a stratified sample, so labelling can stop whenever the
intervals are narrow enough without biasing the sample. `labels sample-sheet`
writes the next documents to read; a person lists the concepts that apply to
each (or `none`), and `labels import-documents` writes every other concept as a
negative, which is what makes recall measurable.

```bash
hontology labels sample <baseline-run> calendar.csv --out sample.json
hontology labels sample-sheet <ontology-id> sample.json --out sheet.csv --count 120
hontology labels import-documents <ontology-id> sheet.csv
hontology eval arms <baseline-run> calendar.csv --arm <run> --manifest sample.json
```

In a hierarchy, documents are labelled on leaf classes only; a parent class is
true exactly when one of its leaves is.

**Human labels and machine annotations are kept apart.** A sample labelled by a
machine annotator, such as an LLM labelling blind, is stored as a named
annotation set beside the label bank, never in it: it can cover the very
articles a person labelled, and a score against it measures agreement with that
annotator, not correctness. Every page and `eval arms --annotator <name>` can
score against a set instead of the human labels, and says which it used.

```bash
hontology labels annotations-import <ontology-id> claude.csv --name claude-blind-001 \
    --description "who annotated, how, under what instructions"
hontology eval arms <baseline-run> calendar.csv --arm <run> --manifest sample.json \
    --annotator claude-blind-001
```

An arm can be scored on the sample before, or instead of, judging every
calendar window. `run sample` gives the sample's first documents another run's
retrieval and judges just those, uncapped; rerun with a larger `--first` and the
same `--run-id` as labelling continues, and only the new documents are judged:

```bash
hontology run sample <ontology-id> hier.json sample.json --candidates-from <baseline-run> --first 120
```

`eval arms` puts every arm beside the baseline: calendar results raw and
verified, article-level precision and recall end to end (a pair never judged
counts as no) and judge-only, and judging cost. Intervals resample whole
documents, because one article's labels are correlated and resampling single
labels would understate them. An arm counts as an improvement only when the
paired interval on its F1 difference lies above zero. For a hierarchical arm it
adds recall at each level given a positive parent, yes answers on parent
classes where no leaf below is actually true, and how many of its true positives the
baseline's retrieval had selected at all.

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
