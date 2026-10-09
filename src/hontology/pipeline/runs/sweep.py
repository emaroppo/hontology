"""Declarative, resumable parameter sweeps.

Declare the cross-product once — models × prompts × selection settings — and run
every cell. This is the payoff of the stage-key work: each cell resolves to the
same composed keys the runner uses, so a cell whose run already exists is
**skipped**, and cells sharing a candidates key reuse retrieval instead of
re-embedding.

That makes a sweep resumable for free. These runs are long and prone to being
interrupted; restarting one should cost only what is genuinely left to do.

The cross-product is over *axes*, each a list of overrides deep-merged onto a
base config. Expressing it that way rather than as a nested loop keeps a sweep
one committable file, so the experiment grid is reviewable in a diff.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from hontology.db.models import Run
from hontology.ontology import snapshots
from hontology.pipeline.runs import config as run_config
from hontology.pipeline.runs import runner

log = logging.getLogger(__name__)


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Cell:
    name: str
    config: dict
    candidates_key: str
    judge_key: str
    existing_run_id: int | None = None

    @property
    def done(self) -> bool:
        return self.existing_run_id is not None

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "candidates_key": self.candidates_key,
            "judge_key": self.judge_key,
            "existing_run_id": self.existing_run_id,
            "done": self.done,
        }


@dataclass
class Plan:
    cells: list[Cell] = field(default_factory=list)

    @property
    def todo(self) -> list[Cell]:
        return [c for c in self.cells if not c.done]

    def as_dict(self) -> dict:
        shared: dict[str, int] = {}
        for cell in self.cells:
            shared[cell.candidates_key] = shared.get(cell.candidates_key, 0) + 1
        return {
            "cells": [c.as_dict() for c in self.cells],
            "total": len(self.cells),
            "done": len(self.cells) - len(self.todo),
            "todo": len(self.todo),
            # Cells sharing a key embed once between them.
            "distinct_candidate_keys": len(shared),
        }


def expand(base: dict, axes: dict[str, list[dict]]) -> list[tuple[str, dict]]:
    """Cross-product of *axes* merged onto *base*.

    ``axes`` maps an axis name to the list of overrides along it, e.g.
    ``{"prompt": [{"judge": {"prompt_id": "strict_v1"}}, ...]}``. Each result is
    ``(label, config)`` where the label names the chosen point on every axis.
    """
    if not axes:
        return [("base", dict(base))]

    names = list(axes)
    out: list[tuple[str, dict]] = []
    for combination in itertools.product(*(range(len(axes[n])) for n in names)):
        config = dict(base)
        parts: list[str] = []
        for axis_name, index in zip(names, combination, strict=True):
            override = axes[axis_name][index]
            config = deep_merge(config, override)
            parts.append(override.get("_label") or f"{axis_name}{index}")
        # `_label` is metadata for naming, never part of the config.
        config = {k: v for k, v in config.items() if k != "_label"}
        out.append(("-".join(parts), config))
    return out


def plan(
    session: Session,
    ontology_id: int,
    *,
    base: dict,
    axes: dict[str, list[dict]],
    name_prefix: str = "sweep",
) -> Plan:
    """Resolve every cell to its keys and check which already exist."""
    version = snapshots.resolve(
        session, ontology_id, base.get("common", {}).get("ontology_version")
    ).version

    existing = {
        (run.candidates_key, run.judge_key): run.id
        for run in session.scalars(
            select(Run).where(
                Run.ontology_id == ontology_id, Run.status.in_(("done", "candidates"))
            )
        )
    }

    result = Plan()
    for label, config in expand(base, axes):
        normalized = run_config.normalize(config)
        keys = run_config.stage_keys(normalized, version)
        result.cells.append(
            Cell(
                name=f"{name_prefix}-{label}",
                config=config,
                candidates_key=keys["candidates"],
                judge_key=keys["judge"],
                existing_run_id=existing.get((keys["candidates"], keys["judge"])),
            )
        )
    return result


def execute(
    session: Session,
    ontology_id: int,
    sweep_plan: Plan,
    *,
    document_limit: int = 100,
    judge_limit: int | None = None,
    skip_judge: bool = False,
    on_cell: Any = None,
) -> list[dict]:
    """Run every unfinished cell, in order.

    Cells are executed sequentially rather than in parallel: they contend for one
    local model, and running them concurrently would make latency measurements
    meaningless.
    """
    results: list[dict] = []
    for index, cell in enumerate(sweep_plan.todo, start=1):
        log.info("sweep cell %s/%s: %s", index, len(sweep_plan.todo), cell.name)
        run = runner.create_run(
            session, ontology_id=ontology_id, config=cell.config, name=cell.name
        )
        run_id = run.id
        session.commit()

        outcome = runner.execute(
            session,
            run,
            document_limit=document_limit,
            judge_limit=judge_limit,
            skip_judge=skip_judge,
        )
        results.append({"cell": cell.name, "run_id": run_id} | outcome)
        if callable(on_cell):
            on_cell(index, len(sweep_plan.todo), cell.name)
    return results
