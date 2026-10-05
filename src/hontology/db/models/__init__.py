"""All ORM models.

Imported as a unit so that ``Base.metadata`` is complete before Alembic
autogenerates, and so cross-module relationship strings resolve.
"""

from hontology.db.base import Base
from hontology.db.models.corpus import (
    Document,
    FeedArticle,
    FeedEvent,
    FeedSlice,
    IngestWatermark,
)
from hontology.db.models.labels import (
    ALL_SOURCES,
    TRUSTED_SOURCES,
    Observation,
    PairLabel,
)
from hontology.db.models.ontology import (
    Category,
    Concept,
    ConceptGroup,
    ConceptGroupMember,
    Locus,
    Ontology,
    Owner,
)
from hontology.db.models.runs import Candidate, Run, Verdict
from hontology.db.models.snapshot import OntologySnapshot
from hontology.db.models.taxonomy import Code, CodeSystem, ConceptCode
from hontology.db.models.vectors import (
    Embedding,
    EmbeddingModel,
    SimilarityRun,
    SimilarityScore,
)

__all__ = [
    "ALL_SOURCES",
    "TRUSTED_SOURCES",
    "Base",
    "Candidate",
    "Category",
    "Code",
    "CodeSystem",
    "Concept",
    "ConceptCode",
    "ConceptGroup",
    "ConceptGroupMember",
    "Document",
    "FeedArticle",
    "Embedding",
    "EmbeddingModel",
    "FeedEvent",
    "FeedSlice",
    "IngestWatermark",
    "Locus",
    "Observation",
    "Ontology",
    "OntologySnapshot",
    "Owner",
    "PairLabel",
    "Run",
    "SimilarityRun",
    "SimilarityScore",
    "Verdict",
]
