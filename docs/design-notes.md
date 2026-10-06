# Design notes

A few decisions in [hontology](../README.md) that are load-bearing and
non-obvious.

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
  one request, with an explicit opt-in to retry. The exception is a connection
  failure, which a network blip produces just as well as a dead host: it gets
  three tries in separate batches before it counts.
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
