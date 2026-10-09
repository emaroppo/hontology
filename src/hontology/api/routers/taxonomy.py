"""Code system endpoints: codebook ingest, similarity runs, link curation."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from hontology.config import get_settings
from hontology.db.models import Code, CodeSystem, Concept, ConceptCode
from hontology.db.session import get_db
from hontology.ingest import cameo, themes
from hontology.judge.providers.base import ProviderError
from hontology.retrieve import embed, similarity

router = APIRouter(prefix="/taxonomy", tags=["taxonomy"])


class CodeOut(BaseModel):
    id: int
    code: str
    name: str | None = None
    level: str
    system: str | None = None  # the code system's slug


def _code_out(code: Code) -> CodeOut:
    return CodeOut(
        id=code.id, code=code.code, name=code.name, level=code.level, system=code.system.slug
    )


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
    """A similarity run scores; it never links. Proposals are ticked one by one
    under each class, so every link is one a person chose."""

    ontology_id: int
    system: str = cameo.CAMEO_SLUG
    # None picks the system's default level: CAMEO's 4-digit events, or the
    # single level of a flat system such as GKG themes.
    level: str | None = None
    concept_fields: str = "name+definition"
    model: str | None = None


class SimilarityOut(BaseModel):
    run_id: int
    n_scores: int


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


@router.post("/themes/ingest")
def ingest_themes(db: Session = Depends(get_db)):
    """Fetch and load the GKG theme lookup. Idempotent."""
    try:
        return themes.ingest(db)
    except Exception as exc:  # noqa: BLE001 - surface any fetch failure to the UI
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"could not fetch the GKG theme lookup: {exc}"
        ) from exc


@router.get("/systems")
def list_systems(db: Session = Depends(get_db)):
    """The code systems loaded, each with its levels, the default first."""
    counts: dict[int, dict[str, int]] = {}
    for system_id, level, n in db.execute(
        select(Code.system_id, Code.level, func.count()).group_by(Code.system_id, Code.level)
    ):
        counts.setdefault(system_id, {})[level] = n
    out = []
    for system in db.scalars(select(CodeSystem).order_by(CodeSystem.slug)):
        levels = counts.get(system.id, {})
        default = _default_level(system.slug, levels)
        out.append(
            {
                "slug": system.slug,
                "name": system.name,
                "levels": [
                    {"level": level, "n_codes": levels[level]}
                    for level in sorted(levels, key=lambda lv: (lv != default, lv))
                ],
            }
        )
    return out


def _default_level(slug: str, levels: dict[str, int]) -> str | None:
    """CAMEO's finest level, 4-digit events; otherwise the most populous level,
    which for a flat system such as GKG themes is its only one."""
    if slug == cameo.CAMEO_SLUG and cameo.LEVELS[4] in levels:
        return cameo.LEVELS[4]
    return max(levels, key=lambda level: levels[level]) if levels else None


@router.get("/codes", response_model=list[CodeOut])
def list_codes(level: str | None = None, limit: int = 500, db: Session = Depends(get_db)):
    system = _cameo_system(db)
    query = select(Code).where(Code.system_id == system.id)
    if level:
        query = query.where(Code.level == level)
    return list(db.scalars(query.order_by(Code.code).limit(limit)))


def _link_out(row: ConceptCode) -> LinkOut:
    return LinkOut(
        code=_code_out(row.code),
        score=row.similarity_score,
        proposed_by_run=row.similarity_run_id,
        manual=row.similarity_run_id is None,
    )


@router.get("/concepts/{concept_id}/links", response_model=list[LinkOut])
def concept_links(concept_id: int, db: Session = Depends(get_db)):
    rows = db.scalars(select(ConceptCode).where(ConceptCode.concept_id == concept_id))
    return [_link_out(row) for row in rows]


@router.get("/ontologies/{ontology_id}/links")
def ontology_links(ontology_id: int, db: Session = Depends(get_db)) -> dict[int, list[LinkOut]]:
    """Every class's links in one call, keyed by concept id; unlinked classes absent."""
    rows = db.scalars(
        select(ConceptCode)
        .join(Concept, Concept.id == ConceptCode.concept_id)
        .where(Concept.ontology_id == ontology_id)
    )
    out: dict[int, list[LinkOut]] = {}
    for row in rows:
        out.setdefault(row.concept_id, []).append(_link_out(row))
    return out


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
            code=_code_out(code),
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
    system = db.scalar(select(CodeSystem).where(CodeSystem.slug == payload.system))
    if system is None:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"the {payload.system!r} codebook has not been ingested yet",
        )
    levels: dict[str, int] = {
        level: n
        for level, n in db.execute(
            select(Code.level, func.count())
            .where(Code.system_id == system.id)
            .group_by(Code.level)
        )
    }
    level = payload.level or _default_level(system.slug, levels)
    if level is None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"{payload.system!r} holds no codes")
    provider = embed.get_provider(settings.default_embed_provider)

    try:
        result = similarity.run_similarity(
            db,
            provider,
            payload.model or settings.default_embed_model,
            ontology_id=payload.ontology_id,
            system_id=system.id,
            level=level,
            concept_fields=payload.concept_fields,
            auto_link=False,
        )
    except ProviderError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, f"embedding provider unavailable: {exc}"
        ) from exc

    return SimilarityOut(run_id=result.run_id, n_scores=result.n_scores)
