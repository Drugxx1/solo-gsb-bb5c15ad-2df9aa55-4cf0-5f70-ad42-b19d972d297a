"""FastAPI application: network prefix policy analysis.

Endpoints
---------
POST   /api/analyze                          analyze a rule set (optionally persist)
POST   /api/analyses/{id}/simulate           preview add/replace/remove changes without saving
GET    /api/analyses                         list saved analysis versions
GET    /api/analyses/{id}                    fetch one version with the full result
GET    /api/analyses/{id}/rules              query rules (filters: status/action/ip_version)
GET    /api/analyses/{id}/findings           query findings (filters: severity/behavioral_only)
GET    /api/analyses/{id}/decision-map       CIDR decision map of this version (query: cidr)
GET    /api/analyses/{id}/redundant-rules    per-rule removability inventory (recomputed LPM diff)
GET    /api/analyses/{id}/export.json        download the result as a JSON file
GET    /api/analyses/{id}/diff/{oid}         effective-action semantic diff of two versions
POST   /api/analyses/{id}/change-guard/{oid} authorize the semantic diff against caller-supplied grants
GET    /health                               liveness probe
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import JSONResponse

from . import __version__
from .analyzer import ANALYZER_VERSION, ValidationError, analyze_rules
from . import db
from .change_guard import ChangeGuardError, guard_changes, parse_guard_entries
from .decision_map import DecisionMapError, build_decision_map, parse_query_cidr
from .diff import ANALYSIS_ID_RE, diff_analyses
from .redundant import redundant_rule_inventory
from .schemas import AnalyzeRequest, ChangeGuardRequest, SimulateRequest
from .simulate import PatchError, apply_patch, rebuild_input_rules


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    yield


app = FastAPI(
    title="Network Prefix Policy Analyzer",
    version=__version__,
    description=(
        "Submit IPv4/IPv6 CIDR rules with priority, action and source tags. "
        "The service computes each rule's effective coverage under longest-prefix "
        "match, identifies fully shadowed / partially overlapping / action-conflicting "
        "rules, and returns reproducible minimal conflict pairs."
    ),
    lifespan=lifespan,
)


@app.exception_handler(ValidationError)
async def _validation_error_handler(_request, exc: ValidationError):
    return JSONResponse(status_code=400, content={"error": "invalid_ruleset", "detail": str(exc)})


@app.exception_handler(PatchError)
async def _patch_error_handler(_request, exc: PatchError):
    # The whole batch is rejected; every offending target is reported at once.
    return JSONResponse(
        status_code=400,
        content={"error": "invalid_simulation", "detail": str(exc), "issues": exc.issues},
    )


@app.exception_handler(DecisionMapError)
async def _decision_map_error_handler(_request, exc: DecisionMapError):
    return JSONResponse(status_code=400, content={"error": "invalid_cidr", "detail": str(exc)})


@app.exception_handler(ChangeGuardError)
async def _change_guard_error_handler(_request, exc: ChangeGuardError):
    # The whole authorization set is rejected; every problem is reported at once.
    return JSONResponse(
        status_code=400,
        content={"error": "invalid_change_guard", "detail": str(exc), "issues": exc.issues},
    )


def _make_id(ruleset_hash: str) -> str:
    return "an_" + ruleset_hash[:16]


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "prefix-policy-analyzer",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__, "analyzer_version": ANALYZER_VERSION}


@app.post("/api/analyze", status_code=200)
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    """Analyze a rule set.

    The analysis is deterministic: identical inputs always produce the same
    ``ruleset_hash`` and the same ``id`` (stored versions are deduplicated).
    """
    raw_rules = [r.model_dump(exclude_defaults=False) for r in req.rules]
    # model_dump turns unset rule_id into None; the analyzer assigns r1, r2...
    result = analyze_rules(raw_rules)

    analysis_id = _make_id(result["ruleset_hash"])
    persisted = False
    if req.save:
        saved_id = db.save_analysis(analysis_id, req.name, result)
        persisted = True
        analysis_id = saved_id

    result["id"] = analysis_id
    result["name"] = req.name
    result["created_at"] = db.utc_now_iso()
    result["persisted"] = persisted
    return result


@app.get("/api/analyses")
def list_analyses(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    total, analyses = db.list_analyses(limit=limit, offset=offset)
    return {"total": total, "analyses": analyses}


@app.get("/api/analyses/{analysis_id}")
def get_analysis(analysis_id: str) -> dict[str, Any]:
    result = db.get_analysis(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"analysis '{analysis_id}' not found")
    return result


@app.post("/api/analyses/{analysis_id}/simulate", status_code=200)
def simulate_changes(analysis_id: str, req: SimulateRequest) -> dict[str, Any]:
    """Preview a batch of ``add`` / ``replace`` / ``remove`` rule changes.

    The analyzer input is rebuilt from the saved version, the validated patch
    is applied, and the LPM analysis is re-run.  Nothing is persisted: the
    response carries the unsaved preview plus its final effective-action
    semantic diff against the base version.
    """
    base = db.get_analysis(analysis_id)
    if base is None:
        raise HTTPException(status_code=404, detail=f"analysis '{analysis_id}' not found")

    add_rules = [r.model_dump() for r in req.add]
    replace_rules = [r.model_dump() for r in req.replace]
    remove_rules = [r.model_dump() for r in req.remove]

    base_input = rebuild_input_rules(base)
    patched_input = apply_patch(base_input, add_rules, replace_rules, remove_rules)

    # Invalid CIDRs/actions inside replacement or added rules surface as the
    # same 400 invalid_ruleset error as POST /api/analyze.
    simulated = analyze_rules(patched_input)
    simulated["id"] = None
    simulated["name"] = None
    simulated["created_at"] = db.utc_now_iso()
    simulated["persisted"] = False

    diff = diff_analyses(base, simulated)
    return {
        "base": {
            "id": analysis_id,
            "ruleset_hash": base.get("ruleset_hash"),
            "analyzer_version": base.get("analyzer_version"),
        },
        "simulated": simulated,
        "diff": diff,
    }


@app.get("/api/analyses/{analysis_id}/rules")
def get_rules(
    analysis_id: str,
    status: str | None = Query(default=None, pattern="^(active|partial|shadowed)$"),
    action: str | None = Query(default=None, pattern="^(permit|deny)$"),
    ip_version: int | None = Query(default=None, alias="ip", ge=4, le=6),
) -> dict[str, Any]:
    rows = db.query_rules(
        analysis_id, status=status, action=action, ip_version=ip_version
    )
    if rows is None:
        raise HTTPException(status_code=404, detail=f"analysis '{analysis_id}' not found")
    return {"analysis_id": analysis_id, "count": len(rows), "rules": rows}


@app.get("/api/analyses/{analysis_id}/findings")
def get_findings(
    analysis_id: str,
    severity: str | None = Query(default=None, pattern="^(conflict|warning|info)$"),
    behavioral_only: bool = Query(default=False),
) -> dict[str, Any]:
    rows = db.query_findings(
        analysis_id, severity=severity, behavioral_only=behavioral_only
    )
    if rows is None:
        raise HTTPException(status_code=404, detail=f"analysis '{analysis_id}' not found")
    return {"analysis_id": analysis_id, "count": len(rows), "findings": rows}


@app.get("/api/analyses/{analysis_id}/decision-map")
def get_decision_map(
    analysis_id: str,
    cidr: str = Query(..., description="Strict IPv4/IPv6 CIDR query range"),
) -> dict[str, Any]:
    """Decision map of a *saved* version over one CIDR query.

    Read-only: the version's stored rules and their winning effective
    coverage are partitioned into disjoint, exhaustive minimal CIDR blocks.
    Each block reports its address count, final action (``none`` when nothing
    matches), winning rule/source tag, and the full longest-prefix-first
    match chain (losing duplicates included). No analysis is created.
    """
    # The query is validated before looking the version up, mirroring the
    # diff endpoint's format-before-existence ordering.
    query = parse_query_cidr(cidr)

    result = db.get_analysis(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"analysis '{analysis_id}' not found")

    return build_decision_map(result, query)


@app.get("/api/analyses/{analysis_id}/redundant-rules")
def get_redundant_rules(analysis_id: str) -> dict[str, Any]:
    """Inventory the rules that can be individually deleted without effect.

    Every rule of the saved version is removed in turn, the longest-prefix
    analysis is recomputed on the remainder, and the final effective actions
    (``permit``/``deny``/``none``) are compared against the original version
    for both address families.  A rule is ``safe_to_remove`` only when no
    address changes action; ``changed_addresses`` is the exact number of
    addresses whose action would change.  Read-only: nothing is persisted.
    """
    result = db.get_analysis(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"analysis '{analysis_id}' not found")
    return redundant_rule_inventory(result)


@app.get("/api/analyses/{analysis_id}/export.json")
def export_analysis(analysis_id: str) -> Response:
    result = db.get_analysis(analysis_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"analysis '{analysis_id}' not found")
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    return Response(
        content=payload,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{analysis_id}.json"'},
    )


def _validate_analysis_id_pair(base_id: str, target_id: str) -> None:
    """Format/identity checks shared by the diff and change-guard endpoints."""
    bad = [i for i in (base_id, target_id) if not ANALYSIS_ID_RE.match(i)]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=f"invalid analysis id format: {', '.join(sorted(set(bad)))}; "
            "expected 'an_' followed by 16 lowercase hexadecimal characters",
        )
    if base_id == target_id:
        raise HTTPException(status_code=400, detail="base and target ids must differ")


@app.get("/api/analyses/{base_id}/diff/{target_id}")
def diff_versions(base_id: str, target_id: str) -> dict[str, Any]:
    """Compare final effective actions of two saved versions.

    Every address covered by either version is classified by its effective
    action in each version (``permit``/``deny``, ``none`` when no rule
    matches).  Only addresses whose action changed are returned, as minimal
    CIDR ranges grouped by the six change types; adjacent ranges of the same
    type are merged.
    """
    _validate_analysis_id_pair(base_id, target_id)

    base = db.get_analysis(base_id)
    if base is None:
        raise HTTPException(status_code=404, detail=f"analysis '{base_id}' not found")
    target = db.get_analysis(target_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"analysis '{target_id}' not found")

    return diff_analyses(base, target)


@app.post("/api/analyses/{base_id}/change-guard/{target_id}", status_code=200)
def change_guard_versions(base_id: str, target_id: str, req: ChangeGuardRequest) -> dict[str, Any]:
    """Authorize the base→target semantic diff against caller-supplied grants.

    Each authorization entry grants one action transition
    (``permit``/``deny``/``none`` → a *different* one) on one strict CIDR.
    A changed address is authorized only when covered by a grant of the same
    IP family and the same transition; grants of one transition are unioned
    (nesting/duplicates allowed), while overlapping grants of different
    transitions are rejected with ``400``.  The response reports ``passed``
    and, per transition, the unauthorized surface as minimal CIDRs with exact
    address counts.  Nothing is persisted.
    """
    _validate_analysis_id_pair(base_id, target_id)

    # The authorization set is validated before the versions are looked up,
    # mirroring the decision-map endpoint's format-before-existence ordering.
    grants = parse_guard_entries([e.model_dump() for e in req.authorizations])

    base = db.get_analysis(base_id)
    if base is None:
        raise HTTPException(status_code=404, detail=f"analysis '{base_id}' not found")
    target = db.get_analysis(target_id)
    if target is None:
        raise HTTPException(status_code=404, detail=f"analysis '{target_id}' not found")

    return guard_changes(base, target, grants)
