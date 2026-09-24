"""End-to-end tests for the POST /api/analyses/{id}/merge-precheck endpoint."""

from __future__ import annotations

import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from app import db


@pytest.fixture()
def client() -> TestClient:
    fd, path = tempfile.mkstemp(suffix=".db", prefix="ppa-test-merge-")
    os.close(fd)
    os.remove(path)  # let sqlite create it
    db.reset_db(path)
    from app.main import app

    with TestClient(app) as c:
        yield c
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(path + suffix)
        except OSError:
            pass


def _save(client: TestClient, rules: list[dict], name: str | None = None) -> str:
    body: dict = {"rules": rules}
    if name is not None:
        body["name"] = name
    r = client.post("/api/analyze", json=body)
    assert r.status_code == 200, r.text
    assert r.json()["persisted"] is True
    return r.json()["id"]


def _merge(client: TestClient, aid: str, a: dict, b: dict):
    return client.post(f"/api/analyses/{aid}/merge-precheck", json={"a": a, "b": b})


# ---------------------------------------------------------------------------
# Compatible merges
# ---------------------------------------------------------------------------
def test_disjoint_changes_merge_compatibly(client: TestClient) -> None:
    # Base: permit /8 with a deny /16 inside.  A adds a deny /24 in one place,
    # B adds a permit /16 on previously uncovered space; the changed surfaces
    # never overlap.
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"},
        {"rule_id": "b", "cidr": "10.1.0.0/16", "action": "deny"},
    ])
    r = _merge(client, aid,
               {"add": [{"rule_id": "x", "cidr": "10.9.9.0/24", "action": "deny"}]},
               {"add": [{"rule_id": "y", "cidr": "192.168.0.0/16", "action": "deny"}]})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["status"] == "compatible"
    assert payload["compatible"] is True
    assert payload["structural_conflicts"] == []
    assert payload["conflicts"] == {"divergence": [], "merge_preservation": []}

    merged = payload["merged"]
    assert merged["persisted"] is False
    assert merged["id"] is None
    ids = [rule["rule_id"] for rule in merged["rules"]]
    assert ids == ["a", "b", "x", "y"]  # A's add before B's (A applied first)

    # The merged diff against the base carries both independent additions.
    types = {(c["ip_version"], c["change_type"]): c for c in payload["diff"]["changes"]}
    assert types[(4, "permit_to_deny")]["cidrs"] == ["10.9.9.0/24"]
    assert types[(4, "none_to_deny")]["cidrs"] == ["192.168.0.0/16"]

    # Side metadata reports the per-batch operation counts.
    assert payload["base"]["id"] == aid
    assert payload["sides"]["a"]["operations"] == {"add": 1, "replace": 0, "remove": 0}
    assert payload["sides"]["b"]["operations"] == {"add": 1, "replace": 0, "remove": 0}
    assert payload["sides"]["a"]["ruleset_hash"] != payload["sides"]["b"]["ruleset_hash"]


def test_same_action_on_overlapping_changed_range_is_compatible(client: TestClient) -> None:
    # Both sides permit previously-uncovered 11/16 (different new rule ids);
    # their changed ranges intersect but the actions agree -> no divergence.
    aid = _save(client, [{"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"}])
    r = _merge(client, aid,
               {"add": [{"rule_id": "x", "cidr": "11.0.0.0/16", "action": "permit"}]},
               {"add": [{"rule_id": "y", "cidr": "11.0.0.0/16", "action": "permit"}]})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["status"] == "compatible"
    assert payload["conflicts"]["divergence"] == []
    assert payload["conflicts"]["merge_preservation"] == []
    # One of the two same-CIDR permits wins; the end result forwards permit.
    assert {(c["change_type"], tuple(c["cidrs"])) for c in payload["diff"]["changes"]} == {
        ("none_to_permit", ("11.0.0.0/16",))
    }


def test_merge_preserves_each_side_change_when_rules_cooperate(client: TestClient) -> None:
    # A claims previously-uncovered 11/16 as permit; B adds a deny subnet in a
    # different part of uncovered space.  Neither batch changes an address the
    # other changes, and the merged replay keeps both sides' new actions.
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"},
        {"rule_id": "b", "cidr": "10.1.0.0/16", "action": "deny"},
    ])
    r = _merge(client, aid,
               {"add": [{"rule_id": "x", "cidr": "11.0.0.0/16", "action": "permit"}]},
               {"add": [{"rule_id": "y", "cidr": "12.0.9.0/24", "action": "deny"}]})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["status"] == "compatible"
    types = {(c["change_type"], tuple(c["cidrs"])) for c in payload["diff"]["changes"]}
    assert ("none_to_permit", ("11.0.0.0/16",)) in types
    assert ("none_to_deny", ("12.0.9.0/24",)) in types
    # Both new rules are effective in the merged analysis.
    effective = {
        rule["rule_id"]: rule["effective_cidrs"] for rule in payload["merged"]["rules"]
    }
    assert effective["x"] == ["11.0.0.0/16"]
    assert effective["y"] == ["12.0.9.0/24"]


# ---------------------------------------------------------------------------
# Structural conflicts
# ---------------------------------------------------------------------------
def test_same_rule_id_targeted_by_both_batches_is_structural_conflict(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"},
        {"rule_id": "b", "cidr": "10.1.0.0/16", "action": "deny"},
    ])
    r = _merge(client, aid,
               {"remove": [{"rule_id": "b"}]},
               {"replace": [{"rule_id": "b", "cidr": "10.1.0.0/16", "action": "permit"}]})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["status"] == "structural_conflict"
    assert payload["compatible"] is False
    assert payload["structural_conflicts"] == [{
        "rule_id": "b", "side_a_operation": "remove", "side_b_operation": "replace",
    }]
    # The merge is not attempted when the rule structure clashes.
    assert payload["merged"] is None
    assert payload["diff"] is None
    assert payload["conflicts"]["merge_preservation"] == []


def test_two_new_rules_of_same_id_from_both_sides_is_structural_conflict(client: TestClient) -> None:
    aid = _save(client, [{"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"}])
    r = _merge(client, aid,
               {"add": [{"rule_id": "x", "cidr": "11.0.0.0/16", "action": "deny"}]},
               {"add": [{"rule_id": "x", "cidr": "12.0.0.0/16", "action": "deny"}]})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["status"] == "structural_conflict"
    assert payload["structural_conflicts"] == [{
        "rule_id": "x", "side_a_operation": "add", "side_b_operation": "add",
    }]
    assert payload["merged"] is None


# ---------------------------------------------------------------------------
# Effective-action conflicts
# ---------------------------------------------------------------------------
def test_divergence_reports_minimal_cidr_base_and_both_side_actions(client: TestClient) -> None:
    # Base only forwards 10/8; both sides claim previously-uncovered 11/16,
    # A as permit, B as deny.  The exact intersection (11/16) is a divergence.
    aid = _save(client, [{"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"}])
    r = _merge(client, aid,
               {"add": [{"rule_id": "x", "cidr": "11.0.0.0/16", "action": "permit"}]},
               {"add": [{"rule_id": "y", "cidr": "11.0.0.0/16", "action": "deny"}]})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["status"] == "action_conflict"
    assert payload["structural_conflicts"] == []

    divergence = payload["conflicts"]["divergence"]
    assert len(divergence) == 1
    conflict = divergence[0]
    assert conflict["ip_version"] == 4
    assert conflict["base_action"] is None
    assert conflict["side_a_action"] == "permit"
    assert conflict["side_b_action"] == "deny"
    assert conflict["cidrs"] == ["11.0.0.0/16"]
    assert conflict["address_count"] == 2 ** 16
    assert conflict["range_count"] == 1

    # A then B keeps both rules; same-CIDR LPM ties break by submission order,
    # so A's earlier permit wins over B's later deny: B's deny expectation is
    # the merge-preservation violation.
    violations = {
        (v["side"], v["expected_action"], v["merged_action"]): v
        for v in payload["conflicts"]["merge_preservation"]
    }
    assert ("b", "deny", "permit") in violations
    assert violations[("b", "deny", "permit")]["cidrs"] == ["11.0.0.0/16"]
    assert ("a", "permit", "permit") not in violations

    # The merged analysis and its base diff are still returned for reference.
    assert payload["merged"] is not None
    assert payload["diff"] is not None
    assert payload["diff"]["target"]["id"] is None


def test_divergence_on_partial_overlap_returns_intersection_only(client: TestClient) -> None:
    # Base forwards nothing on 11/16 space.  A claims a /16 permit, B claims a
    # /24 deny strictly inside it.  Only the /24 intersection diverges; the
    # remainder of A's /16 stays permit after the A-then-B merge.
    aid = _save(client, [{"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"}])
    r = _merge(client, aid,
               {"add": [{"rule_id": "x", "cidr": "11.0.0.0/16", "action": "permit"}]},
               {"add": [{"rule_id": "y", "cidr": "11.0.1.0/24", "action": "deny"}]})
    assert r.status_code == 200, r.text
    divergence = r.json()["conflicts"]["divergence"]
    assert len(divergence) == 1
    assert divergence[0]["cidrs"] == ["11.0.1.0/24"]
    assert divergence[0]["address_count"] == 2 ** 8
    assert divergence[0]["base_action"] is None
    assert divergence[0]["side_a_action"] == "permit"
    assert divergence[0]["side_b_action"] == "deny"


def test_preservation_violation_without_divergence(client: TestClient) -> None:
    # Base: permit p /8 -> deny d /16 -> permit g /24 (g forwards its /24).
    # A removes g: the deny d reclaims 10.1.1.0/24 (permit -> deny on A).
    # B removes d: g still forwards the /24 permit, so B's own replay does not
    # change 10.1.1.0/24 relative to the base (it lies outside B's changed
    # surface) -> no A∩B divergence there.  In the A-then-B merge both d and g
    # are gone, so the /24 falls back to p's permit: A's deny is not kept.
    aid = _save(client, [
        {"rule_id": "p", "cidr": "10.0.0.0/8", "action": "permit"},
        {"rule_id": "d", "cidr": "10.1.0.0/16", "action": "deny"},
        {"rule_id": "g", "cidr": "10.1.1.0/24", "action": "permit"},
    ])
    r = _merge(client, aid,
               {"remove": [{"rule_id": "g"}]},
               {"remove": [{"rule_id": "d"}]})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["status"] == "action_conflict"
    # B's changed surface is /16 minus /24; A's is exactly /24, so the two
    # surfaces do not intersect -> no divergence entry.
    assert payload["conflicts"]["divergence"] == []

    violations = payload["conflicts"]["merge_preservation"]
    a_violations = [v for v in violations if v["side"] == "a"]
    assert len(a_violations) == 1
    v = a_violations[0]
    assert v["cidrs"] == ["10.1.1.0/24"]
    assert v["base_action"] == "permit"
    assert v["expected_action"] == "deny"
    assert v["merged_action"] == "permit"
    assert v["address_count"] == 2 ** 8


def test_diverging_addresses_collapse_to_minimal_cidrs(client: TestClient) -> None:
    # Two adjacent /24 siblings diverge with the same base/A/B actions; they
    # must be reported as one minimal /23 rather than two /24s.
    aid = _save(client, [{"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"}])
    r = _merge(client, aid,
               {"add": [
                   {"rule_id": "x1", "cidr": "11.0.0.0/24", "action": "permit"},
                   {"rule_id": "x2", "cidr": "11.0.1.0/24", "action": "permit"},
               ]},
               {"add": [
                   {"rule_id": "y1", "cidr": "11.0.0.0/24", "action": "deny"},
                   {"rule_id": "y2", "cidr": "11.0.1.0/24", "action": "deny"},
               ]})
    assert r.status_code == 200, r.text
    divergence = r.json()["conflicts"]["divergence"]
    assert len(divergence) == 1
    assert divergence[0]["cidrs"] == ["11.0.0.0/23"]
    assert divergence[0]["range_count"] == 1
    assert divergence[0]["address_count"] == 2 ** 9


def test_ipv6_divergence_handled_exactly(client: TestClient) -> None:
    aid = _save(client, [{"rule_id": "a", "cidr": "2001:db8::/32", "action": "permit"}])
    r = _merge(client, aid,
               {"add": [{"rule_id": "x", "cidr": "2001:db9::/48", "action": "permit"}]},
               {"add": [{"rule_id": "y", "cidr": "2001:db9::/48", "action": "deny"}]})
    assert r.status_code == 200, r.text
    divergence = r.json()["conflicts"]["divergence"]
    assert len(divergence) == 1
    assert divergence[0]["ip_version"] == 6
    assert divergence[0]["cidrs"] == ["2001:db9::/48"]
    assert divergence[0]["address_count"] == 2 ** 80
    assert divergence[0]["base_action"] is None


def test_ipv4_and_ipv6_ranges_never_overlay(client: TestClient) -> None:
    # An IPv4 change on A and an IPv6 change on B cannot intersect even
    # though both are "the whole" of their respective rule CIDRs.
    aid = _save(client, [{"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"}])
    r = _merge(client, aid,
               {"add": [{"rule_id": "x", "cidr": "11.0.0.0/16", "action": "deny"}]},
               {"add": [{"rule_id": "y", "cidr": "2001:db9::/48", "action": "deny"}]})
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["status"] == "compatible"
    assert payload["conflicts"]["divergence"] == []


# ---------------------------------------------------------------------------
# Batch validation / error contract
# ---------------------------------------------------------------------------
def test_invalid_batches_report_every_issue_per_side(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"},
        {"rule_id": "b", "cidr": "10.1.0.0/16", "action": "deny"},
    ])
    r = _merge(client, aid,
               {"remove": [{"rule_id": "ghost"}]},
               {"add": [{"rule_id": "a", "cidr": "10.2.0.0/16", "action": "deny"}]})
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "invalid_simulation"
    assert len(body["issues"]) == 2
    assert body["issues"][0].startswith("batch A:")
    assert "'ghost'" in body["issues"][0]
    assert body["issues"][1].startswith("batch B:")
    assert "'a'" in body["issues"][1]


def test_empty_batch_is_rejected_with_side_label(client: TestClient) -> None:
    aid = _save(client, [{"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"}])
    r = _merge(client, aid, {}, {"remove": [{"rule_id": "a"}]})
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "invalid_simulation"
    assert body["issues"] == [
        "batch A: at least one operation (add/replace/remove) is required"
    ]


def test_invalid_rule_cidr_surfaces_as_invalid_ruleset(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"},
        {"rule_id": "b", "cidr": "10.1.0.0/16", "action": "deny"},
    ])
    r = _merge(client, aid,
               {"add": [{"rule_id": "x", "cidr": "10.2.0.5/24", "action": "deny"}]},
               {"remove": [{"rule_id": "b"}]})
    assert r.status_code == 400
    assert r.json()["error"] == "invalid_ruleset"
    assert "invalid CIDR" in r.json()["detail"]


def test_batches_that_jointly_remove_every_rule_are_rejected(client: TestClient) -> None:
    # Each batch on its own leaves >=1 rule, but A then B deletes them all.
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"},
        {"rule_id": "b", "cidr": "10.1.0.0/16", "action": "deny"},
    ])
    r = _merge(client, aid,
               {"remove": [{"rule_id": "a"}]},
               {"remove": [{"rule_id": "b"}]})
    assert r.status_code == 400
    body = r.json()
    assert body["error"] == "invalid_simulation"
    assert body["detail"].startswith("merged result:")
    assert "at least one rule" in body["detail"]
    # Nothing persisted.
    assert len(client.get(f"/api/analyses/{aid}").json()["rules"]) == 2


def test_schema_level_errors_return_422(client: TestClient) -> None:
    aid = _save(client, [{"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"}])

    # Missing side B entirely.
    r = client.post(
        f"/api/analyses/{aid}/merge-precheck", json={"a": {"remove": [{"rule_id": "a"}]}}
    )
    assert r.status_code == 422

    # Extra/unknown section.
    r = client.post(f"/api/analyses/{aid}/merge-precheck", json={
        "a": {"remove": [{"rule_id": "a"}]},
        "b": {},
        "c": {},
    })
    assert r.status_code == 422

    # Unknown action inside an operation.
    r = client.post(f"/api/analyses/{aid}/merge-precheck", json={
        "a": {"add": [{"rule_id": "x", "cidr": "11.0.0.0/16", "action": "maybe"}]},
        "b": {"remove": [{"rule_id": "a"}]},
    })
    assert r.status_code == 422


def test_unknown_analysis_returns_404(client: TestClient) -> None:
    r = client.post(
        "/api/analyses/an_0000000000000000/merge-precheck",
        json={
            "a": {"remove": [{"rule_id": "a"}]},
            "b": {"remove": [{"rule_id": "b"}]},
        },
    )
    assert r.status_code == 404
    assert "an_0000000000000000" in r.json()["detail"]


def test_precheck_never_persists(client: TestClient) -> None:
    aid = _save(client, [
        {"rule_id": "a", "cidr": "10.0.0.0/8", "action": "permit"},
        {"rule_id": "b", "cidr": "10.1.0.0/16", "action": "deny"},
    ])

    # Compatible run.
    r = _merge(client, aid,
               {"add": [{"rule_id": "x", "cidr": "11.0.0.0/16", "action": "permit"}]},
               {"add": [{"rule_id": "y", "cidr": "12.0.0.0/16", "action": "deny"}]})
    assert r.status_code == 200

    # Conflicting run.
    r = _merge(client, aid,
               {"remove": [{"rule_id": "a"}]},
               {"remove": [{"rule_id": "b"}]})
    assert r.status_code == 400

    # Only the one base version exists; the base itself is untouched.
    listing = client.get("/api/analyses").json()
    assert listing["total"] == 1
    assert listing["analyses"][0]["id"] == aid
    assert {x["rule_id"] for x in client.get(f"/api/analyses/{aid}").json()["rules"]} == {"a", "b"}
