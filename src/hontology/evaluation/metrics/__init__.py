"""Metrics, with the uncertainty attached.

A point estimate from a few hundred labels invites over-reading. A ten-point F1
gap on thirty pairs is noise, and reporting it as a finding is how a pipeline
acquires "improvements" that do not survive contact with more data. So every
headline number carries an interval, and comparing two runs uses a *paired* test
over the pairs they both judged.

Intervals are derived from confusion counts rather than stored separately, so
they stay correct when counts are aggregated across runs or slices:

- **precision and recall** are proportions, so they get a **Wilson** score
  interval — closed form, and well behaved near 0 and 1 and at small n where the
  normal approximation produces bounds outside [0, 1].
- **F1** is not a simple proportion, so its interval comes from a **bootstrap**
  over the labelled pairs.
"""
