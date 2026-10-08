# Removed features

What was taken out of the UI or the API, in which commit, and how to bring it
back. Removing a page section rarely removed what computes it: in each entry,
**Still present** lists the endpoint, client method and evaluation code that
still exist, so reinstating is mostly copying the old page code back.

The old code of any page is one command away; `<commit>^` is the commit just
before the removal:

```bash
git show <commit>^:src/hontology/ui/pages/6_Evaluation.py
```

## Evaluation page

Removed in `4e4e27b` ("Open on a live leaderboard; replace the Evaluation
page"). The page was `src/hontology/ui/pages/6_Evaluation.py`. Recover it with
`git show 4e4e27b^:src/hontology/ui/pages/6_Evaluation.py`. Line numbers below
are in that file.

Most of it moved rather than went:

| Section | Lines | Where it is now |
|---|---|---|
| Labelled sample (end-to-end P/R/F1, cost, disagreements) | 77–138 | Home: Leaderboard and Single run |
| Funnel | 141–173 | Home → Single run, "Funnel" expander, without the parts listed below |
| Detections | 176–220 | Home → Single run, "Detections" expander, without the parts listed below |
| Judge (P/R/F1 with intervals, confusion counts) | 223–244 | Judgement → Evaluation, by judge version, without the confusion chart |
| Retrieval (recall@k) | 247–270 | Retrieval → Evaluation, by retrieval version, computed live |
| Calibration | 272–285 | Judgement → Evaluation, by judge version |
| Compare two runs | 365–389 | Home → Comparison, as the arms comparison on the sample |

Gone without a replacement:

### Breakdown by concept, family, category or locus

- **Lines:** 288–326. A "Slice by" radio, a precision-and-recall chart with intervals, and a table of n, TP, FP, FN, precision, recall and F1 per slice.
- **Still present:**
  - `GET /eval/runs/{id}/breakdown` (`dimension`, `include_machine`) and `Api.run_breakdown`;
  - `evalkit.breakdown.breakdown`;
  - `charts.rates_with_intervals`;
  - `hontology eval breakdown`.
- **To reinstate:** put lines 288–326 in Home → Single run, as an expander, with `selected["id"]` replaced by `run_id`.
- **Caveat:** it scores against the label bank only. To match the rest of Home, `breakdown` would need a labels-file option, as `tuning.truth` gives the other views.

### Error triage with the model's evidence and reasoning

- **Lines:** 329–362. FP and FN counts, then up to 25 errors, each with the model's quoted evidence, the label note and the reasoning trace.
- **Still present:**
  - `GET /eval/runs/{id}/errors` and `Api.run_errors`;
  - `evalkit.errors.triage` and `evalkit.errors.summary`;
  - `hontology eval errors`.
- **To reinstate:** put it in Home → Single run, next to "Where it disagrees with the labels". That expander lists the same kind of errors on the sample, but without the evidence or reasoning.
- **Caveat:** label bank only, like the breakdown.

### DuckDB leaderboard

- **Lines:** 390 to the end. An F1-against-cost scatter and a table, from `warehouse.duckdb`.
- **Still present:**
  - `GET /eval/leaderboard` and `Api.leaderboard`;
  - `evalkit.warehouse` (`record`, `leaderboard`);
  - `charts.leaderboard_scatter`;
  - `hontology eval run --record`, the only code that writes it.
- **Why it went:** nothing but that command wrote to it, so it went stale and never included sample or calendar scores. Home → Leaderboard is computed live instead.
- **To reinstate the chart:** feed `charts.leaderboard_scatter` from the live rows (F1 against tokens).

### Pairwise comparison on the label bank

- **Lines:** 365–389.
- **Still present:**
  - `GET /eval/compare` and `Api.compare_runs`;
  - `evalkit.compare.compare_runs`;
  - `hontology eval compare`.
- **Why it went:** Home → Comparison uses the sample instead, paired the same way, and shows the tables `eval arms` writes.

### Run-level retrieval metrics from the label bank

- **Lines:** 247–268. Retrieval precision, coverage, MRR and cutoff recall for one run.
- **Still present:** `evaluation["retrieval"]` from `GET /eval/runs/{id}` (`Api.evaluate_run`, `evalkit.evaluate.evaluate_run`, `metrics.retrieval_metrics`).
- **Partial replacement:** Retrieval → Evaluation reports recall at the cutoff, recall in the pool and recall@k per version. It has no precision, coverage or MRR.

### Confusion bar chart

- **Line:** 244.
- **Still present:** `charts.confusion_bar`. Judgement → Evaluation has the same counts as numbers.

### Parts dropped from the funnel

- **Lines:** 161–171.
- **What went:**
  - each step's explanatory note (`step["note"]`);
  - the warnings for errored verdicts and for selected pairs never judged (`flow["errored_verdicts"]`, `flow["unjudged_selected"]`).
- **Still present:** both come back from `GET /eval/runs/{id}/funnel`. The expander just doesn't show them.

### Parts dropped from detections

- **Lines:** 188–216.
- **What went:**
  - the "Unverified" count;
  - the detections table;
  - the events CSV download (`Api.detections_csv(id, events=True)`).
- **Still present:** all of them, through the same endpoints.

### Machine and stale label toggles

- **Lines:** 33–50. "Count machine labels" and "Count stale labels".
- **Why they went:** the new views score trusted, current labels only, or a labels file.
- **Still present:** `include_machine` and `include_stale` on `evaluate_run`, `breakdown` and `errors`.

### Liveness and empty-bank warnings

- **Lines:** 64–75.
- **Still present:** `evaluation["liveness"]` from `GET /eval/runs/{id}`.

## Filtering page "Per code" tab

- **Added in** `3674871`, **replaced in** `89fa377` ("Score each stage on its own page, by version").
- **What it was:** a per-code table of documents admitted (each credited to its first matching code), fetched, junk, labelled and positive, from the label bank.
- **Recover it with:** `git show 89fa377^:src/hontology/ui/pages/2_Filtering.py`, function `show_report`.
- **Still present:**
  - `GET /eval/filter-report` and `Api.filter_report`;
  - `evalkit.filter_report.report`;
  - `hontology eval filter-report`.
- **Replaced by:** Filtering → Evaluation counts per link rather than per code, credits a document to every code that admits it, and adds the calendar and the documents only one code admits.

## Code Links page and automatic linking

- **Removed in** `3674871` ("Replace Code Links with a Filtering page; tick GKG theme candidates").
- **Recover it with:** `git show 3674871^:src/hontology/ui/pages/2_Code_Links.py`.
- **The page:** replaced by Filtering → Links. That covers what it did and adds GKG themes.
- **Automatic linking from the API:** `POST /taxonomy/similarity` no longer takes `threshold`, `adaptive`, `rel_margin`, `min_score`, `max_k` or `auto_link`. A run only scores, and proposals are ticked one by one under each class.
  - **Why:** on the supply chain ontology, the default selection added about 14 of CAMEO's 144 event codes to every class.
- **Still present:** `similarity.run_similarity(..., auto_link=True)` and its threshold and adaptive selection (`_select_threshold`, `_select_adaptive`, `_apply_links`), so a script can still link automatically.
- **To reinstate it in the API:** add the fields back to `SimilarityIn` in `api/routers/taxonomy.py` and pass them through.

## Ontology page "Resolve version" button

- **Removed in** `5892f37` ("Show the ontology as a hierarchy, with OWL import and export in the UI").
- **Why:** versions are minted automatically by the next run or label. The header now says which version the live wording is, or that it has been edited.
- **Recover it with:** `git show 5892f37^:src/hontology/ui/pages/1_Ontology.py`.
- **Still present:** `POST /ontologies/{id}/snapshot` and `snapshots.resolve_current`. `Api.snapshot` was removed in the same commit; it was one line calling that endpoint.
