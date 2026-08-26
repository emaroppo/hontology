"""Code system endpoints: codebook ingest, similarity runs, link curation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.models import Code, CodeSystem, ConceptCode
from hontology.db.session import get_db
from hontology.ingest import cameo
from hontology.judge.providers.base import ProviderError
from hontology.retrieve import embed, similarity

router = APIRouter(prefix="/taxonomy", tags=["taxonomy"])


class CodeOut(BaseModel):
    id: int
    code: str
    name: str | None = None
    level: str


class LinkOut(BaseModel):
    code: CodeOut
    score: float | None = None
    # None means a human asserted this link, and a recompute will not touch it.
    proposed_by_run: int | None = None
    manual: bool


class CandidateOut(BaseModel):
    code: CodeOut
    score: float
    linked: bool


class SetLinkIn(BaseModel):
    concept_id: int
    code_id: int
    linked: bool


class SimilarityIn(BaseModel):
    ontology_id: int
    level: str = "event"
    concept_fields: str = "name+definition"
    threshold: float = 0.45
    adaptive: bool = True
    rel_margin: float = 0.10
    min_score: float = 0.25
    max_k: int = 15
    model: str | None = None


class SimilarityOut(BaseModel):
    run_id: int
    n_scores: int
    n_linked: int
    n_manual_preserved: int


def _cameo_system(db: Session) -> CodeSystem:
    system = db.scalar(select(CodeSystem).where(CodeSystem.slug == cameo.CAMEO_SLUG))
    if system is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "the CAMEO codebook has not been ingested yet; POST /taxonomy/cameo/ingest",
        )
    return system


@router.post("/cameo/ingest")
def ingest_cameo(db: Session = Depends(get_db)):
    """Fetch and load the public CAMEO codebook. Idempotent."""
    try:
        return cameo.ingest(db)
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as a 502
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"could not fetch the CAMEO codebook: {exc}"
        ) from exc


@router.get("/codes", response_model=list[CodeOut])
def list_codes(level: str | None = None, limit: int = 500, db: Session = Depends(get_db)):
    system = _cameo_system(db)
    query = select(Code).where(Code.system_id == system.id)
    if level:
        query = query.where(Code.level == level)
    return list(db.scalars(query.order_by(Code.code).limit(limit)))


@router.get("/concepts/{concept_id}/links", response_model=list[LinkOut])
def concept_links(concept_id: int, db: Session = Depends(get_db)):
    rows = db.scalars(select(ConceptCode).where(ConceptCode.concept_id == concept_id))
    return [
        LinkOut(
            code=CodeOut.model_validate(row.code, from_attributes=True),
            score=row.similarity_score,
            proposed_by_run=row.similarity_run_id,
            manual=row.similarity_run_id is None,
        )
        for row in rows
    ]


@router.get("/concepts/{concept_id}/candidates", response_model=list[CandidateOut])
def concept_candidates(
    concept_id: int, run_id: int, limit: int = 20, db: Session = Depends(get_db)
):
    """Top-scoring codes from a run, with whether each is currently linked."""
    linked = {
        row.code_id
        for row in db.scalars(select(ConceptCode).where(ConceptCode.concept_id == concept_id))
    }
    return [
        CandidateOut(
            code=CodeOut.model_validate(code, from_attributes=True),
            score=score,
            linked=code.id in linked,
        )
        for code, score in similarity.candidates_for_concept(
            db, run_id, concept_id, limit=limit
        )
    ]


@router.post("/links", status_code=status.HTTP_204_NO_CONTENT)
def set_link(payload: SetLinkIn, db: Session = Depends(get_db)):
    """Tick or untick an association.

    Ticking stores it with no run id, which promotes a machine proposal to a
    human assertion and protects it from the next recompute.
    """
    similarity.set_link(db, payload.concept_id, payload.code_id, linked=payload.linked)


@router.post("/similarity", response_model=SimilarityOut)
def run_similarity(payload: SimilarityIn, db: Session = Depends(get_db)):
    settings = get_settings()
    system = _cameo_system(db)
    provider = embed.get_provider(settings.default_embed_provider, settings.ollama_host)

    try:
        result = similarity.run_similarity(
            db,
            provider,
            payload.model or settings.default_embed_model,
            ontology_id=payload.ontology_id,
            system_id=system.id,
            level=payload.level,
            concept_fields=payload.concept_fields,
            threshold=payload.threshold,
            adaptive=payload.adaptive,
            rel_margin=payload.rel_margin,
            min_score=payload.min_score,
            max_k=payload.max_k,
        )
    except ProviderError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"embedding provider unavailable: {exc}"
        ) from exc

    return SimilarityOut(
        run_id=result.run_id,
        n_scores=result.n_scores,
        n_linked=result.n_linked,
        n_manual_preserved=result.n_manual_preserved,
    )
