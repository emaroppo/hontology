"""Article-level scoring on the labelled document sample.

Two views of a run, both over documents a person labelled against every concept:

- **End to end:** a pair the run never judged counts as *no*. This is what the
  system as a whole delivers, retrieval and any budget included.
- **Judge only:** just the pairs the run judged, which isolates the judge.

**Why the bootstrap resamples documents.** One article carries a label for every
concept, and those labels are not independent: an article about a typhoon is
negative on forty-odd concepts for the same reason. Resampling single pairs
would treat them as independent and give intervals that are too narrow, so every
interval here resamples whole documents. The difference between two runs is
*paired*: both are scored on the same resampled documents each time.
"""
