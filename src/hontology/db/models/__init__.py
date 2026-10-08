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
    AnnotationSet,
    CalendarReview,
    MachineAnnotation,
    Observation,
    PairLabel,
)
from hontology.db.models.ontology import (
    Category,
    Concept,
    ConceptGroup,
    ConceptGroupMember,
    ConceptRelation,
    Locus,
    Ontology,
    Owner,
)
from hontology.db.models.runs import Candidate, ExtractedEvent, Run, Verdict
from hontology.db.models.snapshot import LinkSnapshot, OntologySnapshot
from hontology.db.models.taxonomy import Code, CodeSystem, ConceptCode
from hontology.db.models.vectors import (
    Embedding,
    EmbeddingModel,
    SimilarityRun,
    SimilarityScore,
)

__all__ = [
    "ALL_SOURCES",
    "AnnotationSet",
    "CalendarReview",
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
    "ConceptRelation",
    "Document",
    "FeedArticle",
    "Embedding",
    "EmbeddingModel",
    "ExtractedEvent",
    "FeedEvent",
    "FeedSlice",
    "IngestWatermark",
    "Locus",
    "MachineAnnotation",
    "Observation",
    "Ontology",
    "LinkSnapshot",
    "OntologySnapshot",
    "Owner",
    "PairLabel",
    "Run",
    "SimilarityRun",
    "SimilarityScore",
    "Verdict",
]
