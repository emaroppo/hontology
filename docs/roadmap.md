# Roadmap and future work

Where [hontology](../README.md) stands.

## Built and in progress

- [x] Schema: ontology, taxonomy, vectors, corpus, runs, labels, snapshots
- [x] Pluggable LLM provider layer (Ollama, llama.cpp; Anthropic behind the same protocol)
- [x] Ontology service, import/export, snapshots, API and editor UI
- [x] CAMEO ingest, embeddings, similarity review
- [x] GDELT ingest: watermarking, catch-up, backfill, optional continuous watcher
- [x] Article scraping: extractor chain, quality gate, robots and rate limiting
- [x] Run configuration, both retrieval sources, judge loop with resume
- [x] Ground-truth bank, labelling queue, staleness handling
- [x] Metrics, A/B comparison, consistency checks, regression gate, leaderboard
- [x] GDELT knowledge graph feed, feed-code filter, near-duplicate grouping
- [x] Event calendars: window-by-window runs, verified detections, outage-aware scoring
- [x] Labelled sample with a frozen order, article-level scoring, arms report
- [x] Class hierarchy, OWL interchange, hierarchical judging
- [ ] Flat vs hierarchical comparison: first report at 120 labelled documents

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
- **Concept groups and class relations have no editor in the UI.** Groups are
  reachable only through import; the class hierarchy through import or an OWL
  tool such as Protégé.
- **No machine pre-labelling pass**, though the schema and the adjudication flow
  were built for one.
- **Descriptive per-stage drill-down.** The funnel and detection export both work
  with no labels; score distributions and per-stage inspection do not exist yet.
- **Batched judging's accuracy against per-pair judging is unmeasured.** It is
  the flat baseline of the hierarchy comparison, chosen for cost; the labelled
  sample will measure it, but not against per-pair judging. The three
  aggregation rules are covered only by tests with a mocked provider.
- **Further arms after the hierarchy comparison.** Each is its own pre-stated
  comparison, on the same labelled sample:
  - *leaf wording*: moving criteria from the leaves to their parents, which
    first needs labels tied to a fixed reference definition rather than to the
    prompt's wording, and runs that carry their own wording;
  - *ontology-assisted retrieval*: scoring families rather than single leaves,
    spreading selections across families, and using a class's feed-code links as
    evidence, starting with retrieval's recall on the labelled sample;
  - *budgets*: a per-concept judging budget, and fixed budgets for every arm.
- **One retrieval stage does different jobs in different arms**, which undercuts
  comparing them. The batched and per-pair judges judge exactly the pairs
  retrieval selected, so its ranking and cutoff bound their recall. The
  hierarchical and extraction judges read only *which articles* retrieval
  selected, any article with one class past the cutoff, and then reach classes
  their own way; for them retrieval is an article filter, and with the adaptive
  cutoff it passes nearly everything (24 of 38,151 articles excluded on run
  566). The embedding variant of extraction also ranks leaves against each
  event, a second retrieval the stage version does not cover. Yet every arm
  shares one retrieval version and is scored with the same pair-level metrics.
  To do: make retrieval's role explicit per arm (pair candidates, an article
  gate, or none), version and score each role as what it is (an article gate by
  articles passed and true-match articles kept), and decide whether arms should
  share one article gate so that only the judging differs between them.
- **US events wait for state-level places.** A US-wide window holds far too many
  articles to scrape, so US entries are left out of the evaluation calendar.
- **`precursor_of` drives nothing yet.** Precursors are leaf classes under their
  family; the relation is stored for when it does.
- **Forecasting from detections belongs in a separate library.** hontology would
  provide a point-in-time detections export, each row stamped with when it could
  first have been known, and stay free of any particular target variable.
