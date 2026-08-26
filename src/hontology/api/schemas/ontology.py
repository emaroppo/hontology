"""Request and response shapes for the ontology API."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ConceptIn(BaseModel):
    name: str = Field(min_length=1)
    definition: str | None = None
    # Not documentation: both criteria are inserted into the judge prompt
    # verbatim, and a sharp exclusion is usually the best precision lever there is.
    inclusion_criteria: str | None = None
    exclusion_criteria: str | None = None
    category: str | None = None
    weight: float | None = None


class ConceptPatch(BaseModel):
    name: str | None = None
    definition: str | None = None
    inclusion_criteria: str | None = None
    exclusion_criteria: str | None = None
    category: str | None = None
    weight: float | None = None


class ConceptOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ontology_id: int
    name: str
    definition: str | None = None
    inclusion_criteria: str | None = None
    exclusion_criteria: str | None = None
    weight: float | None = None
    category_id: int | None = None


class OntologyIn(BaseModel):
    slug: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str = Field(min_length=1)
    description: str | None = None


class OntologyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    slug: str
    name: str
    description: str | None = None


class GroupMemberIn(BaseModel):
    concept: str
    weight: float | None = None


class GroupIn(BaseModel):
    name: str
    description: str | None = None
    parent: str | None = None
    members: list[GroupMemberIn] = Field(default_factory=list)


class OntologyImport(BaseModel):
    """A portable ontology. Keyed by name throughout so it merges across databases."""

    export_version: int = 1
    slug: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    name: str
    description: str | None = None
    concepts: list[ConceptIn] = Field(default_factory=list)
    groups: list[GroupIn] = Field(default_factory=list)


class SnapshotOut(BaseModel):
    version: str
    content_hash: str
    n_concepts: int
    created: bool
