"""Ontology health checks.

Two failure modes that cost labelling hours to discover downstream, and seconds
to catch here.

**Strength drift.** A concept named "*Successful* negotiation" whose definition
only requires "a concrete commitment" will match a mere promise — and the model
is *right* to do so, because the prompt tells it to follow the definition over
the name. The bug is in the ontology, not the judge, but it surfaces as
inexplicable false positives. The check looks for strength words in a name that
the definition never earns.

**Near-duplicates.** Two concepts whose definitions embed almost identically
cannot be told apart by retrieval, so labels for one contaminate the other and
per-concept metrics become meaningless. Cosine similarity over the stored
concept vectors finds them.

Both are warnings, never errors. An ontology is the user's to author, and a lint
that blocks is a lint that gets ignored.
"""
